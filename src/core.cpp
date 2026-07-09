#include "vecdb/core.hpp"

#include <immintrin.h>  //hardware intrinsics header

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>  // for numeric_limits
#include <queue>
#include <stdexcept>
#include <unordered_set>

namespace vecdb {
VectorEngine::VectorEngine(size_t dimensions, size_t M)
    : dimensions_(dimensions),
      M_(M),
      M_max0_(M * 2),
      max_layer_(-1),  //-1 -> graph is empty
      ep_index_(-1),
      generator_(42)  // fixed seed
{
  // The HNSW paper defines the optimal multiplier to scale the exponential
  // decay
  mult_ = 1.0 / std::log(static_cast<double>(M_));
}
// dice roll
int VectorEngine::generate_random_layer() {
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  double r = distribution(generator_);

  // prevent taking natural log to abs zero
  if (r == 0.0) r = std::numeric_limits<double>::min();

  // Formula: floor(-ln(uniform_random) * multiplier)
  return static_cast<int>(-std::log(r) * mult_);
}

void VectorEngine::insert(VectorId id, const std::vector<float>& data) {
  if (data.size() != dimensions_) {
    throw std::invalid_argument(
        "Vector dimensions do not match databse dimensiosn.");
  }
  // L2 normalization
  //  calc squared sum of all the elements
  float sq_sum = 0.0f;
  for (float val : data) {
    sq_sum += val * val;
  }

  // calc magnitude
  float magnitude = std::sqrt(sq_sum);

  // prevent div by 0 if someone enteres an empty vector
  if (magnitude == 0.0f) {
    magnitude = 1.0f;
  }

  // divide every elemet by the magnitude and append to flat array
  std::vector<float> norm_data;
  norm_data.reserve(dimensions_);
  for (float val : data) {
    norm_data.push_back(val / magnitude);
  }

  ids_.push_back(id);
  raw_data_.insert(raw_data_.end(), norm_data.begin(), norm_data.end());

  // graph prep
  int level = generate_random_layer();

  // dynamically expand the graph if we encounter a higher ID
  if (id >= nodes_.size()) {
    nodes_.resize(id + 1);
  }
  nodes_[id].neighbors.resize(static_cast<size_t>(level + 1));

  // First node inserted becomes the "King" of the empty graph
  if (max_layer_ == -1) {
    max_layer_ = level;
    ep_index_ = static_cast<int>(id);
    return;
  }

  // HNSW ROUNTING
  VectorId curr_node = static_cast<VectorId>(ep_index_);

  // Phase 1 : Drop (Descend layers without making connections)
  for (int lc = max_layer_; lc > level; --lc) {
    size_t max_connections = (lc == 0) ? M_max0_ : M_;
    size_t ef_construction = 32;

    auto W = search_layer(norm_data, curr_node, ef_construction, lc);

    // Extract from Max-heap (they come out worst to best)
    std::vector<SearchResult> neighbors;
    neighbors.reserve(W.size());
    while (!W.empty()) {
      neighbors.push_back(W.top());
      W.pop();
    }

    // The absolute closest node becomes our entry point for the next layer down
    curr_node = neighbors.back().id;

    // wire the connection
    size_t num_to_connect = std::min(neighbors.size(), max_connections);

    // Iterate backwords (from closest to furthest)
    for (auto it = neighbors.rbegin();
         it != neighbors.rend() && num_to_connect > 0; ++it, --num_to_connect) {
      VectorId neighbors_id = it->id;

      // Bidirectional wiring
      nodes_[id].neighbors[static_cast<size_t>(lc)].push_back(neighbors_id);
      nodes_[neighbors_id].neighbors[static_cast<size_t>(lc)].push_back(id);

      // Naive Pruning : Prevent infinite edge growth for existing neighbors
      if (nodes_[neighbors_id].neighbors[static_cast<size_t>(lc)].size() >
          max_connections) {
        nodes_[neighbors_id].neighbors[static_cast<size_t>(lc)].erase(
            nodes_[neighbors_id].neighbors[static_cast<size_t>(lc)].begin());
      }
    }
  }

  // If this new node rolled the highest level ever seen, it becomes the new
  // King
  if (level > max_layer_) {
    max_layer_ = level;
    ep_index_ = static_cast<int>(id);
  }
}

inline float simd_dot_product(const float* a, const float* b, size_t size) {
  __m256 sum_vec = _mm256_setzero_ps();

  size_t i = 0;
  for (; i + 7 < size; i += 8) {
    __m256 vec_a = _mm256_loadu_ps(a + i);
    __m256 vec_b = _mm256_loadu_ps(b + i);

    sum_vec = _mm256_fmadd_ps(vec_a, vec_b, sum_vec);
  }

  // We have 8 partial sums in our AVX register. We need to horizontally add
  // them together. This is standard AVX boilerplate to collapse a 256-bit
  // register into a single float.
  __m128 sum_half = _mm_add_ps(_mm256_castps256_ps128(sum_vec),
                               _mm256_extractf128_ps(sum_vec, 1));
  __m128 sum_quad = _mm_hadd_ps(sum_half, sum_half);
  __m128 sum_single = _mm_hadd_ps(sum_quad, sum_quad);

  float final_sum = _mm_cvtss_f32(sum_single);

  // Handle any leftovers (e.g., if dimensions was 130 instead of a clean
  // multiple of 8)
  for (; i < size; ++i) {
    final_sum += a[i] * b[i];
  }

  return final_sum;
}
std::vector<SearchResult> VectorEngine::search(const std::vector<float>& query,
                                               size_t k) const {
  if (query.size() != dimensions_) {
    throw std::invalid_argument(
        "Query dimensions do not match databse dimensions.");
  }

  if (ids_.empty() || k == 0) return {};

  // Normalize the user's query vector
  float sq_sum = 0.0f;
  for (float val : query) {
    sq_sum += val * val;
  }
  float magnitude = std::sqrt(sq_sum);
  if (magnitude == 0.0f) magnitude = 1.0f;

  std::vector<float> norm_query;
  norm_query.reserve(dimensions_);
  for (float val : query) {
    norm_query.push_back(val / magnitude);
  }

  VectorId curr_node = static_cast<VectorId>(ep_index_);

  // 1. Highway navigation
  for (int lc = max_layer_; lc > 0; --lc) {
    auto W = search_layer(norm_query, curr_node, 1, lc);
    curr_node = W.top().id;
  }

  // 2. Driveway navigation: Seach layer 0 for the top K res
  //  'ef_search' controls the accuracy vs speed tradeoff
  size_t ef_search = std::max(k, static_cast<size_t>(64));
  auto W = search_layer(norm_query, curr_node, ef_search, 0);

  // Extract and format the final res
  std::vector<SearchResult> results;
  while (!W.empty()) {
    results.push_back(W.top());
    W.pop();
  }

  // Reverse bcuz Max-heap pops worst first
  std::reverse(results.begin(), results.end());

  // trim down to exactly K res
  if (results.size() > k) {
    results.resize(k);
  }
  return results;
}
// A unique identifier: "VEC1" encoded as a 32-bit hex integer
const uint32_t VECDB_MAGIC_NUMBER = 0x31434556;

void VectorEngine::save(const std::string& file_path) const {
  std::ofstream file(file_path, std::ios::binary);
  if (!file) {
    throw DatabaseError("Failed to open file for writing: " + file_path);
  }
  // 1. Write the Header (Magic Number, Vector Count, Dimensions)
  file.write(reinterpret_cast<const char*>(&VECDB_MAGIC_NUMBER),
             sizeof(uint32_t));

  size_t num_vectors = ids_.size();
  file.write(reinterpret_cast<const char*>(&num_vectors), sizeof(size_t));
  file.write(reinterpret_cast<const char*>(&dimensions_), sizeof(size_t));

  // 2. Dump the IDs array directly from RAM to SSD
  if (num_vectors > 0) {
    file.write(reinterpret_cast<const char*>(ids_.data()),
               static_cast<std::streamsize>(num_vectors * sizeof(VectorId)));
  }

  // 3. Dump the float array directly from RAM to SSD
  if (!raw_data_.empty()) {
    file.write(reinterpret_cast<const char*>(raw_data_.data()),
               static_cast<std::streamsize>(raw_data_.size() * sizeof(float)));
  }

  // Verify the write stream didn't fail midway
  if (!file.good()) {
    throw DatabaseError("An OS error occurred while writing data to disk.");
  }
}

void VectorEngine::load(const std::string& file_path) {
  std::ifstream file(file_path, std::ios::binary);
  if (!file) {
    throw DatabaseError("Failed to open file for reading: " + file_path);
  }

  // 1. Security Check: Verify the Magic Number
  uint32_t magic = 0;
  file.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
  // if (magic != VECDB_MAGIC_NUMBERR) {
  //   throw DatabaseError(
  //       "Corrupted or invalid file format. Magic number mismatch");
  // }
  if (magic != VECDB_MAGIC_NUMBER) {
    std::string error_msg = "Magic number mismatch!\n";
    error_msg += "Expected: " + std::to_string(VECDB_MAGIC_NUMBER) + "\n";
    error_msg += "Got: " + std::to_string(magic) + "\n";
    error_msg += "Bytes actually read: " + std::to_string(file.gcount()) + "\n";

    throw DatabaseError(error_msg);
  }

  // 2. Read Metadata
  size_t num_vectors = 0;
  size_t file_dimensions = 0;
  file.read(reinterpret_cast<char*>(&num_vectors), sizeof(size_t));
  file.read(reinterpret_cast<char*>(&file_dimensions), sizeof(size_t));

  if (file_dimensions != dimensions_) {
    throw DatabaseError(
        "Dimensions mismatch! Engine is configured for a different vector "
        "size.");
  }

  // 3. The Performance Hack: Pre-allocate all memory instantly
  // By resizing before reading, we guarantee contiguous memory and avoid
  // the CPU wasting time dynamically expanding the vectors during the load.

  ids_.resize(num_vectors);
  raw_data_.resize(num_vectors * dimensions_);

  // 4. Blast the bytes straight from the SSD into the pre-allocated RAM
  if (num_vectors > 0) {
    // Read expects a char*, not a const char*, because it has to modify the
    // memory
    file.read(reinterpret_cast<char*>(ids_.data()),
              static_cast<std::streamsize>(num_vectors * sizeof(VectorId)));
    file.read(reinterpret_cast<char*>(raw_data_.data()),
              static_cast<std::streamsize>(raw_data_.size() * sizeof(float)));
  }
}

std::priority_queue<SearchResult, std::vector<SearchResult>, ComapreByDistance>
VectorEngine::search_layer(const std::vector<float>& query, VectorId ep,
                           size_t ef, int layer) const {
  // track where we have been
  std::unordered_set<VectorId> visited;
  visited.insert(ep);

  // Min-heap for nodes we need to evaluate
  std::priority_queue<SearchResult, std::vector<SearchResult>,
                      ComapreByDistanceMin>
      candidates;

  // Max-heap for the best results we have found so fat
  std::priority_queue<SearchResult, std::vector<SearchResult>,
                      ComapreByDistance>
      top_results;

  // calc distance for the entry point
  const float* query_ptr = query.data();
  const float* ep_ptr = &raw_data_[ep * dimensions_];
  float ep_dist = 1.0f - simd_dot_product(ep_ptr, query_ptr, dimensions_);

  candidates.push({ep, ep_dist});
  top_results.push({ep, ep_dist});

  while (!candidates.empty()) {
    SearchResult current = candidates.top();
    candidates.pop();

    // GREEDY STOP CONDITION:
    // If our best candidate to explore is further away than the WORST result
    // in our top_results heap, it means we have hit a local minimum. Stop
    // searching.
    if (current.distance > top_results.top().distance) {
      break;
    }

    // Look at all the neighbors of this current node on this specific layer
    for (VectorId neighbor :
         nodes_[current.id].neighbors[static_cast<size_t>(layer)]) {
      if (visited.find(neighbor) == visited.end()) {
        visited.insert(neighbor);

        const float* neighbor_ptr = &raw_data_[neighbor * dimensions_];
        float dist =
            1.0f - simd_dot_product(neighbor_ptr, query_ptr, dimensions_);

        // If our top_results aren't full yet, OR this neighbor is better than
        // our worst result
        if (top_results.size() < ef || dist < top_results.top().distance) {
          candidates.push({neighbor, dist});
          top_results.push({neighbor, dist});

          // Kick out the worst res if we exceed out "ef" limit

          if (top_results.size() > ef) {
            top_results.pop();
          }
        }
      }
    }
  }
  return top_results;
}
}  // namespace vecdb