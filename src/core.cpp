#include "vecdb/core.hpp"

#include <immintrin.h>  //hardware intrinsics header

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>  // for numeric_limits
#include <queue>
#include <stdexcept>

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

  ids_.push_back(id);

  // divide every elemet by the magnitude and append to flat array
  for (float val : data) {
    raw_data_.push_back(val / magnitude);
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

  auto cmp = [](const SearchResult& left, const SearchResult& right) {
    return left.distance < right.distance;
  };
  std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(cmp)>
      max_heap(cmp);

  size_t num_vectors = ids_.size();
  const float* query_ptr = query.data();

  for (size_t i = 0; i < num_vectors; ++i) {
    const float* vec_ptr = &raw_data_[i * dimensions_];

    float dot_product = simd_dot_product(vec_ptr, query_ptr, dimensions_);

    float distance = 1.0f - dot_product;

    if (max_heap.size() < k) {
      max_heap.push({ids_[i], distance});
    } else if (distance < max_heap.top().distance) {
      max_heap.pop();
      max_heap.push({ids_[i], distance});
    }
  }

  std::vector<SearchResult> results;
  results.reserve(max_heap.size());
  while (!max_heap.empty()) {
    results.push_back(max_heap.top());
    max_heap.pop();
  }
  std::reverse(results.begin(), results.end());
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
}  // namespace vecdb