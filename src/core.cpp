#include "vecdb/core.hpp"

#include <immintrin.h>  // hardware intrinsics (AVX2)

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>    // std::rename
#include <fstream>
#include <limits>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <stdexcept>
#include <thread>
#include <unordered_set>

namespace vecdb {

// ---------------------------------------------------------------------------
// Magic number: "VEC1" as a 32-bit little-endian constant.
// ---------------------------------------------------------------------------
static const uint32_t VECDB_MAGIC_NUMBER = 0x31434556;

// ---------------------------------------------------------------------------
// Forward declaration — defined after insert() so it can be inlined.
// ---------------------------------------------------------------------------
inline float simd_dot_product(const float* a, const float* b, size_t size);

// ===========================================================================
// Constructor
// ===========================================================================
VectorEngine::VectorEngine(size_t dimensions, size_t M)
    : dimensions_(dimensions),
      M_(M),
      M_max0_(M * 2),
      max_layer_(-1),
      ep_index_(-1),
      generator_(42)
{
  mult_ = 1.0 / std::log(static_cast<double>(M_));
}

// ===========================================================================
// Layer dice-roll
// ===========================================================================
int VectorEngine::generate_random_layer() {
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  double r = distribution(generator_);
  if (r == 0.0) r = std::numeric_limits<double>::min();
  return static_cast<int>(-std::log(r) * mult_);
}

// ===========================================================================
// 7.1 INSERT — stores metadata alongside vector data
// ===========================================================================
void VectorEngine::insert(VectorId id, const std::vector<float>& data,
                          const nlohmann::json& meta) {
  if (data.size() != dimensions_) {
    throw std::invalid_argument(
        "Vector dimensions do not match database dimensions.");
  }

  // L2 normalisation — CPU work done BEFORE acquiring the lock.
  float sq_sum = 0.0f;
  for (float val : data) sq_sum += val * val;
  float magnitude = std::sqrt(sq_sum);
  if (magnitude == 0.0f) magnitude = 1.0f;

  std::vector<float> norm_data;
  norm_data.reserve(dimensions_);
  for (float val : data) norm_data.push_back(val / magnitude);

  // --- Exclusive write lock: everything below touches shared state -----------
  std::unique_lock lock(mutex_);

  ids_.push_back(id);
  id_to_index_[id] = ids_.size() - 1;
  raw_data_.insert(raw_data_.end(), norm_data.begin(), norm_data.end());

  // Store metadata only if non-empty.
  if (!meta.is_null() && !meta.empty()) {
    metadata_[id] = meta;
  }

  // HNSW graph wiring
  int level = generate_random_layer();

  if (id >= nodes_.size()) nodes_.resize(id + 1);
  nodes_[id].neighbors.resize(static_cast<size_t>(level + 1));

  // First node becomes the entry point of the empty graph.
  if (max_layer_ == -1) {
    max_layer_ = level;
    ep_index_  = static_cast<int>(id);
    return;
  }

  VectorId curr_node = static_cast<VectorId>(ep_index_);

  // Phase 1: Greedy drop — descend from max_layer to level+1 (no edge wiring).
  for (int lc = max_layer_; lc > level; --lc) {
    auto W   = search_layer(norm_data, curr_node, 1, lc);
    curr_node = W.top().id;
  }

  // Phase 2: Connect — wire bidirectional edges for every layer down to 0.
  int connect_from = std::min(max_layer_, level);
  for (int lc = connect_from; lc >= 0; --lc) {
    size_t max_connections = (lc == 0) ? M_max0_ : M_;
    size_t ef_construction = 128;

    auto W = search_layer(norm_data, curr_node, ef_construction, lc);

    std::vector<SearchResult> neighbors;
    neighbors.reserve(W.size());
    while (!W.empty()) { neighbors.push_back(W.top()); W.pop(); }

    if (!neighbors.empty()) curr_node = neighbors.back().id;

    size_t num_to_connect = std::min(neighbors.size(), max_connections);
    for (auto it = neighbors.rbegin();
         it != neighbors.rend() && num_to_connect > 0; ++it, --num_to_connect) {
      VectorId nbr_id = it->id;

      if (static_cast<size_t>(lc) >= nodes_[id].neighbors.size())
        nodes_[id].neighbors.resize(static_cast<size_t>(lc + 1));
      if (static_cast<size_t>(lc) >= nodes_[nbr_id].neighbors.size())
        nodes_[nbr_id].neighbors.resize(static_cast<size_t>(lc + 1));

      nodes_[id].neighbors[static_cast<size_t>(lc)].push_back(nbr_id);
      nodes_[nbr_id].neighbors[static_cast<size_t>(lc)].push_back(id);

      // Prune: if neighbor now has too many connections, drop the worst one.
      auto& nbr_list = nodes_[nbr_id].neighbors[static_cast<size_t>(lc)];
      if (nbr_list.size() > max_connections) {
        size_t worst_pos  = 0;
        float  worst_dist = 0.0f;
        size_t nbr_idx    = id_to_index_.at(nbr_id);
        const float* nbr_ptr = &raw_data_[nbr_idx * dimensions_];
        for (size_t ni = 0; ni < nbr_list.size(); ++ni) {
          size_t    cand_idx = id_to_index_.at(nbr_list[ni]);
          const float* cand_ptr = &raw_data_[cand_idx * dimensions_];
          float d = 1.0f - simd_dot_product(nbr_ptr, cand_ptr, dimensions_);
          if (d > worst_dist) { worst_dist = d; worst_pos = ni; }
        }
        nbr_list.erase(nbr_list.begin() +
                       static_cast<std::ptrdiff_t>(worst_pos));
      }
    }
  }

  if (level > max_layer_) {
    max_layer_ = level;
    ep_index_  = static_cast<int>(id);
  }
}

// ===========================================================================
// AVX2 dot product — unchanged from Phase 6
// ===========================================================================
inline float simd_dot_product(const float* a, const float* b, size_t size) {
  __m256 sum_vec = _mm256_setzero_ps();

  size_t i = 0;
  for (; i + 7 < size; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    sum_vec = _mm256_fmadd_ps(va, vb, sum_vec);
  }

  __m128 sum_half   = _mm_add_ps(_mm256_castps256_ps128(sum_vec),
                                  _mm256_extractf128_ps(sum_vec, 1));
  __m128 sum_quad   = _mm_hadd_ps(sum_half, sum_half);
  __m128 sum_single = _mm_hadd_ps(sum_quad, sum_quad);
  float  final_sum  = _mm_cvtss_f32(sum_single);

  for (; i < size; ++i) final_sum += a[i] * b[i];
  return final_sum;
}

// ===========================================================================
// 7.1 + 7.2 SEARCH — post-filter by metadata AND exclude tombstones
// ===========================================================================
std::vector<SearchResult> VectorEngine::search(const std::vector<float>& query,
                                                size_t k,
                                                MetadataFilter filter) const {
  if (query.size() != dimensions_)
    throw std::invalid_argument(
        "Query dimensions do not match database dimensions.");

  std::shared_lock lock(mutex_);

  if (ids_.empty() || k == 0) return {};

  // Normalise query
  float sq_sum = 0.0f;
  for (float val : query) sq_sum += val * val;
  float magnitude = std::sqrt(sq_sum);
  if (magnitude == 0.0f) magnitude = 1.0f;

  std::vector<float> norm_query;
  norm_query.reserve(dimensions_);
  for (float val : query) norm_query.push_back(val / magnitude);

  VectorId curr_node = static_cast<VectorId>(ep_index_);

  // 1. Highway navigation: greedy descent on upper layers
  for (int lc = max_layer_; lc > 0; --lc) {
    auto W = search_layer(norm_query, curr_node, 1, lc);
    curr_node = W.top().id;
  }

  // 2. Driveway: search layer 0 with a generous ef to compensate for filtering.
  //    Over-fetch factor: 4× to ensure we have enough candidates after filtering.
  size_t ef_search = std::max(k * 4, static_cast<size_t>(200));
  auto W = search_layer(norm_query, curr_node, ef_search, 0);

  // Flatten max-heap (worst → best order) then reverse to best-first.
  std::vector<SearchResult> candidates;
  candidates.reserve(W.size());
  while (!W.empty()) { candidates.push_back(W.top()); W.pop(); }
  std::reverse(candidates.begin(), candidates.end());

  // 3. Post-filter: exclude tombstones and non-matching metadata.
  std::vector<SearchResult> results;
  results.reserve(k);

  for (auto& r : candidates) {
    if (results.size() >= k) break;

    // 7.2 — skip tombstoned vectors
    if (deleted_.count(r.id)) continue;

    // 7.1 — skip vectors that don't match the metadata filter
    if (filter.active) {
      auto it = metadata_.find(r.id);
      if (it == metadata_.end()) continue;          // no metadata → no match
      if (!it->second.contains(filter.key)) continue;
      if (it->second[filter.key] != filter.value)  continue;
    }

    // Attach metadata to the result for the caller
    auto meta_it = metadata_.find(r.id);
    if (meta_it != metadata_.end()) r.metadata = meta_it->second;

    results.push_back(r);
  }

  return results;
}

// ===========================================================================
// 7.2 REMOVE — soft-delete (tombstone)
// ===========================================================================
void VectorEngine::remove(VectorId id) {
  std::unique_lock lock(mutex_);

  if (id_to_index_.find(id) == id_to_index_.end())
    throw std::invalid_argument("VectorId " + std::to_string(id) +
                                " does not exist in the database.");

  deleted_.insert(id);

  // If we just tombstoned the entry point, find a new one.
  if (ep_index_ >= 0 && static_cast<VectorId>(ep_index_) == id) {
    heal_entry_point_locked();
  }
}

// ---------------------------------------------------------------------------
// Heal the entry point: walk layer-0 neighbors of the current ep until we
// find a live (non-tombstoned) node. Must be called with the exclusive lock.
// ---------------------------------------------------------------------------
void VectorEngine::heal_entry_point_locked() {
  if (ids_.empty()) {
    ep_index_  = -1;
    max_layer_ = -1;
    return;
  }

  // BFS over all inserted IDs to find any live node.
  for (VectorId vid : ids_) {
    if (!deleted_.count(vid)) {
      ep_index_ = static_cast<int>(vid);
      // Recalculate max_layer_ from this node's layer count.
      if (vid < nodes_.size() && !nodes_[vid].neighbors.empty()) {
        max_layer_ = static_cast<int>(nodes_[vid].neighbors.size()) - 1;
      }
      return;
    }
  }

  // All vectors are deleted — truly empty graph.
  ep_index_  = -1;
  max_layer_ = -1;
}

// ===========================================================================
// SEARCH_LAYER — single-layer HNSW beam search (unchanged algorithm)
// ===========================================================================
std::priority_queue<SearchResult, std::vector<SearchResult>, ComapreByDistance>
VectorEngine::search_layer(const std::vector<float>& query, VectorId ep,
                           size_t ef, int layer) const {
  std::unordered_set<VectorId> visited;
  visited.insert(ep);

  std::priority_queue<SearchResult, std::vector<SearchResult>,
                      ComapreByDistanceMin> candidates;
  std::priority_queue<SearchResult, std::vector<SearchResult>,
                      ComapreByDistance>    top_results;

  const float* query_ptr = query.data();
  size_t ep_idx = id_to_index_.at(ep);
  const float* ep_ptr = &raw_data_[ep_idx * dimensions_];
  float ep_dist = 1.0f - simd_dot_product(ep_ptr, query_ptr, dimensions_);

  candidates.push({.id = ep, .distance = ep_dist});
  top_results.push({.id = ep, .distance = ep_dist});

  while (!candidates.empty()) {
    SearchResult current = candidates.top();
    candidates.pop();

    if (current.distance > top_results.top().distance) break;

    if (static_cast<size_t>(layer) < nodes_[current.id].neighbors.size()) {
      for (VectorId nbr : nodes_[current.id].neighbors[static_cast<size_t>(layer)]) {
        if (visited.find(nbr) == visited.end()) {
          visited.insert(nbr);

          const float* nbr_ptr =
              &raw_data_[id_to_index_.at(nbr) * dimensions_];
          float dist = 1.0f - simd_dot_product(nbr_ptr, query_ptr, dimensions_);

          if (top_results.size() < ef || dist < top_results.top().distance) {
            candidates.push({.id = nbr, .distance = dist});
            top_results.push({.id = nbr, .distance = dist});
            if (top_results.size() > ef) top_results.pop();
          }
        }
      }
    }
  }
  return top_results;
}

// ===========================================================================
// PERSISTENCE HELPERS
// ===========================================================================

// ---------------------------------------------------------------------------
// Low-level write routine — operates on raw references so both save() and
// write_snapshot() can share the exact same serialisation logic.
// ---------------------------------------------------------------------------
static void write_engine_data(
    std::ofstream& file,
    const std::vector<VectorId>&                        ids,
    const std::vector<float>&                           raw_data,
    const std::unordered_map<VectorId, size_t>&         id_to_index,
    const std::vector<HnswNode>&                        nodes,
    const std::unordered_map<VectorId, nlohmann::json>& metadata,
    const std::unordered_set<VectorId>&                 deleted,
    int    max_layer,
    int    ep_index,
    size_t dimensions)
{
  // 1. Magic + version
  file.write(reinterpret_cast<const char*>(&VECDB_MAGIC_NUMBER), sizeof(uint32_t));
  file.write(reinterpret_cast<const char*>(&VECDB_FORMAT_VERSION), sizeof(uint8_t));

  // 2. Core metadata
  size_t num_vectors = ids.size();
  file.write(reinterpret_cast<const char*>(&num_vectors),  sizeof(size_t));
  file.write(reinterpret_cast<const char*>(&dimensions),   sizeof(size_t));

  // 3. IDs + raw float data
  if (num_vectors > 0) {
    file.write(reinterpret_cast<const char*>(ids.data()),
               static_cast<std::streamsize>(num_vectors * sizeof(VectorId)));
    file.write(reinterpret_cast<const char*>(raw_data.data()),
               static_cast<std::streamsize>(raw_data.size() * sizeof(float)));
  }

  // 4. Graph state
  file.write(reinterpret_cast<const char*>(&max_layer), sizeof(int));
  file.write(reinterpret_cast<const char*>(&ep_index),  sizeof(int));

  // 5. id_to_index map
  size_t map_size = id_to_index.size();
  file.write(reinterpret_cast<const char*>(&map_size), sizeof(size_t));
  for (const auto& [vid, idx] : id_to_index) {
    file.write(reinterpret_cast<const char*>(&vid), sizeof(VectorId));
    file.write(reinterpret_cast<const char*>(&idx), sizeof(size_t));
  }

  // 6. HNSW graph topology
  size_t nodes_size = nodes.size();
  file.write(reinterpret_cast<const char*>(&nodes_size), sizeof(size_t));
  for (const auto& node : nodes) {
    size_t num_layers = node.neighbors.size();
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(size_t));
    for (const auto& layer : node.neighbors) {
      size_t num_nbrs = layer.size();
      file.write(reinterpret_cast<const char*>(&num_nbrs), sizeof(size_t));
      if (num_nbrs > 0)
        file.write(reinterpret_cast<const char*>(layer.data()),
                   static_cast<std::streamsize>(num_nbrs * sizeof(VectorId)));
    }
  }

  // 7. Metadata section (Phase 7.1)
  size_t meta_count = metadata.size();
  file.write(reinterpret_cast<const char*>(&meta_count), sizeof(size_t));
  for (const auto& [vid, meta_json] : metadata) {
    file.write(reinterpret_cast<const char*>(&vid), sizeof(VectorId));
    std::string s = meta_json.dump();
    size_t slen = s.size();
    file.write(reinterpret_cast<const char*>(&slen), sizeof(size_t));
    file.write(s.data(), static_cast<std::streamsize>(slen));
  }

  // 8. Deleted set section (Phase 7.2)
  size_t del_count = deleted.size();
  file.write(reinterpret_cast<const char*>(&del_count), sizeof(size_t));
  for (VectorId vid : deleted)
    file.write(reinterpret_cast<const char*>(&vid), sizeof(VectorId));
}

// ===========================================================================
// SAVE (synchronous) — holds a shared read-lock during the write
// ===========================================================================
void VectorEngine::save(const std::string& file_path) const {
  std::shared_lock lock(mutex_);

  std::ofstream file(file_path, std::ios::binary);
  if (!file) throw DatabaseError("Failed to open file for writing: " + file_path);

  write_engine_data(file, ids_, raw_data_, id_to_index_, nodes_,
                    metadata_, deleted_, max_layer_, ep_index_, dimensions_);

  if (!file.good())
    throw DatabaseError("An OS error occurred while writing data to disk.");
}

// ===========================================================================
// TAKE_SNAPSHOT — deep copy under a brief read-lock (Phase 7.3)
// ===========================================================================
EngineSnapshot VectorEngine::take_snapshot() const {
  std::shared_lock lock(mutex_);

  EngineSnapshot snap;
  snap.ids         = ids_;
  snap.raw_data    = raw_data_;
  snap.id_to_index = id_to_index_;
  snap.nodes       = nodes_;
  snap.metadata    = metadata_;
  snap.deleted     = deleted_;
  snap.max_layer   = max_layer_;
  snap.ep_index    = ep_index_;
  snap.dimensions  = dimensions_;
  return snap;
}

// ===========================================================================
// WRITE_SNAPSHOT — static worker: writes snapshot to .tmp then renames (Phase 7.3)
// ===========================================================================
void VectorEngine::write_snapshot(EngineSnapshot snap, std::string file_path) {
  std::string tmp_path = file_path + ".tmp";

  std::ofstream file(tmp_path, std::ios::binary);
  if (!file) {
    // Can't throw here (we're in a detached thread) — just return.
    return;
  }

  write_engine_data(file, snap.ids, snap.raw_data, snap.id_to_index,
                    snap.nodes, snap.metadata, snap.deleted,
                    snap.max_layer, snap.ep_index, snap.dimensions);

  file.close();

  // Atomic rename: on Linux this is guaranteed to be crash-safe.
  std::rename(tmp_path.c_str(), file_path.c_str());
}

// ===========================================================================
// SAVE_ASYNC — snapshots then launches background writer (Phase 7.3)
// ===========================================================================
bool VectorEngine::save_async(const std::string& file_path) {
  // If a save is already running, refuse to launch another.
  bool expected = false;
  if (!save_in_progress_.compare_exchange_strong(expected, true)) {
    return false;  // save already in progress
  }

  EngineSnapshot snap = take_snapshot();

  // Detach a background thread. It resets the flag when done.
  std::thread([this, snap = std::move(snap), file_path]() mutable {
    write_snapshot(std::move(snap), file_path);
    save_in_progress_.store(false);
  }).detach();

  return true;
}

// ===========================================================================
// LOAD — backward-compatible across format versions 0, 1, 2
// ===========================================================================
void VectorEngine::load(const std::string& file_path) {
  std::unique_lock lock(mutex_);

  std::ifstream file(file_path, std::ios::binary);
  if (!file) throw DatabaseError("Failed to open file for reading: " + file_path);

  // 1. Magic number check
  uint32_t magic = 0;
  file.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
  if (magic != VECDB_MAGIC_NUMBER) {
    std::string msg = "Magic number mismatch! Expected: " +
                      std::to_string(VECDB_MAGIC_NUMBER) +
                      " Got: " + std::to_string(magic);
    throw DatabaseError(msg);
  }

  // 2. Read the format version byte (absent in version 0)
  uint8_t format_version = 0;
  char version_peek;
  // Peek at the next byte: if it's 0 or 1 and a size_t follows cleanly,
  // assume version 0 (legacy). For simplicity we always read it; version 0
  // files happened to write num_vectors here as a size_t starting with
  // a non-zero byte on typical systems, so we check if the value ≤ 2.
  file.read(&version_peek, sizeof(char));
  uint8_t peeked = static_cast<uint8_t>(version_peek);
  if (peeked <= 2) {
    // Looks like a version byte (0, 1, or 2)
    format_version = peeked;
  } else {
    // Legacy format 0: the byte we read was actually the first byte of
    // num_vectors. Seek back.
    file.seekg(-static_cast<std::streamoff>(sizeof(char)), std::ios::cur);
    format_version = 0;
  }

  // 3. Core metadata
  size_t num_vectors   = 0;
  size_t file_dims     = 0;
  file.read(reinterpret_cast<char*>(&num_vectors), sizeof(size_t));
  file.read(reinterpret_cast<char*>(&file_dims),   sizeof(size_t));

  if (file_dims != dimensions_)
    throw DatabaseError(
        "Dimensions mismatch! Engine configured for " +
        std::to_string(dimensions_) + "D, file contains " +
        std::to_string(file_dims) + "D vectors.");

  // 4. IDs + raw float data
  ids_.resize(num_vectors);
  raw_data_.resize(num_vectors * dimensions_);

  if (num_vectors > 0) {
    file.read(reinterpret_cast<char*>(ids_.data()),
              static_cast<std::streamsize>(num_vectors * sizeof(VectorId)));
    file.read(reinterpret_cast<char*>(raw_data_.data()),
              static_cast<std::streamsize>(raw_data_.size() * sizeof(float)));
  }

  // 5. Graph state
  file.read(reinterpret_cast<char*>(&max_layer_), sizeof(int));
  file.read(reinterpret_cast<char*>(&ep_index_),  sizeof(int));

  // 6. id_to_index map
  size_t map_size = 0;
  file.read(reinterpret_cast<char*>(&map_size), sizeof(size_t));
  id_to_index_.clear();
  id_to_index_.reserve(map_size);
  for (size_t i = 0; i < map_size; ++i) {
    VectorId vid; size_t idx;
    file.read(reinterpret_cast<char*>(&vid), sizeof(VectorId));
    file.read(reinterpret_cast<char*>(&idx), sizeof(size_t));
    id_to_index_[vid] = idx;
  }

  // 7. HNSW graph topology
  size_t nodes_size = 0;
  file.read(reinterpret_cast<char*>(&nodes_size), sizeof(size_t));
  nodes_.resize(nodes_size);
  for (size_t i = 0; i < nodes_size; ++i) {
    size_t num_layers = 0;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(size_t));
    nodes_[i].neighbors.resize(num_layers);
    for (size_t l = 0; l < num_layers; ++l) {
      size_t num_nbrs = 0;
      file.read(reinterpret_cast<char*>(&num_nbrs), sizeof(size_t));
      if (num_nbrs > 0) {
        nodes_[i].neighbors[l].resize(num_nbrs);
        file.read(
            reinterpret_cast<char*>(nodes_[i].neighbors[l].data()),
            static_cast<std::streamsize>(num_nbrs * sizeof(VectorId)));
      }
    }
  }

  // 8. Metadata section (format_version >= 1)
  metadata_.clear();
  if (format_version >= 1) {
    size_t meta_count = 0;
    file.read(reinterpret_cast<char*>(&meta_count), sizeof(size_t));
    for (size_t i = 0; i < meta_count; ++i) {
      VectorId vid; size_t slen;
      file.read(reinterpret_cast<char*>(&vid),  sizeof(VectorId));
      file.read(reinterpret_cast<char*>(&slen), sizeof(size_t));
      std::string s(slen, '\0');
      file.read(s.data(), static_cast<std::streamsize>(slen));
      metadata_[vid] = nlohmann::json::parse(s);
    }
  }

  // 9. Deleted set (format_version >= 2)
  deleted_.clear();
  if (format_version >= 2) {
    size_t del_count = 0;
    file.read(reinterpret_cast<char*>(&del_count), sizeof(size_t));
    for (size_t i = 0; i < del_count; ++i) {
      VectorId vid;
      file.read(reinterpret_cast<char*>(&vid), sizeof(VectorId));
      deleted_.insert(vid);
    }
  }
}

}  // namespace vecdb