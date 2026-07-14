#pragma once

#include <cstdint>
#include <atomic>
#include <queue>
#include <random>         // for HNSW probability generation
#include <shared_mutex>   // for reader-writer thread safety
#include <stdexcept>      // for std::runtime_error
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

namespace vecdb {

using VectorId = uint32_t;

// ---------------------------------------------------------------------------
// Binary format version — bump whenever the .vec file layout changes.
// load() branches on this for backward-compatible reading.
//   0 = Phase 6 (original)
//   1 = Phase 7.1 (+ metadata section)
//   2 = Phase 7.2 (+ deleted-set section)
// ---------------------------------------------------------------------------
constexpr uint8_t VECDB_FORMAT_VERSION = 2;

struct SearchResult {
  VectorId id;
  float distance;
  nlohmann::json metadata = nlohmann::json{};  // empty by default; populated in search()
};

class DatabaseError : public std::runtime_error {
 public:
  explicit DatabaseError(const std::string& message)
      : std::runtime_error(message) {}
};

struct HnswNode {
  // list of connected VectorIds on that specific layer
  std::vector<std::vector<VectorId>> neighbors;
};

struct ComapreByDistance {
  // Max-heap: keeps the LARGEST distance at the top
  bool operator()(const SearchResult& a, const SearchResult& b) const {
    return a.distance < b.distance;
  }
};

struct ComapreByDistanceMin {
  // Min-heap: keeps the SMALLEST distance at the top
  bool operator()(const SearchResult& a, const SearchResult& b) const {
    return a.distance > b.distance;
  }
};

// ---------------------------------------------------------------------------
// MetadataFilter — simple key=value equality predicate for search().
// Set active=false (default) to skip filtering entirely.
// ---------------------------------------------------------------------------
struct MetadataFilter {
  std::string key;
  nlohmann::json value;
  bool active = false;
};

// ---------------------------------------------------------------------------
// EngineSnapshot — a deep copy of all engine state for async disk writes.
// Phase 7.3: take_snapshot() populates this under a brief read-lock;
// write_snapshot() then flushes it to disk without holding any lock.
// ---------------------------------------------------------------------------
struct EngineSnapshot {
  std::vector<VectorId>                         ids;
  std::vector<float>                            raw_data;
  std::unordered_map<VectorId, size_t>          id_to_index;
  std::vector<HnswNode>                         nodes;
  std::unordered_map<VectorId, nlohmann::json>  metadata;
  std::unordered_set<VectorId>                  deleted;
  int    max_layer;
  int    ep_index;
  size_t dimensions;
};

class VectorEngine {
 public:
  // M = max connections per layer (default 16 matches HNSW paper recommendation)
  explicit VectorEngine(size_t dimensions, size_t M = 16);

  // --- Core CRUD -----------------------------------------------------------

  // Insert a vector with optional JSON metadata.
  void insert(VectorId id, const std::vector<float>& data,
              const nlohmann::json& meta = nlohmann::json{});

  // Search for the k nearest neighbours. Optionally filter results by metadata.
  std::vector<SearchResult> search(const std::vector<float>& query, size_t k,
                                   MetadataFilter filter = {}) const;

  // Soft-delete: mark a vector as tombstoned. It stays in the graph for
  // routing but is excluded from all search results.
  void remove(VectorId id);

  // --- Persistence ---------------------------------------------------------

  // Synchronous save (safe for tests / shutdown). Holds a read-lock during IO.
  void save(const std::string& file_path) const;

  // Load from disk. Supports format versions 0, 1 and 2.
  void load(const std::string& file_path);

  // Async save (Phase 7.3): snapshots state quickly then writes in background.
  // Returns false immediately if a save is already in progress.
  bool save_async(const std::string& file_path);

  // Static worker called by the background thread.
  static void write_snapshot(EngineSnapshot snap, std::string file_path);

  // Take a deep-copy snapshot of all state under a brief read-lock.
  EngineSnapshot take_snapshot() const;

  // --- Introspection -------------------------------------------------------

  // Returns count of live (non-tombstoned) vectors.
  size_t size() const {
    std::shared_lock lock(mutex_);
    return ids_.size() - deleted_.size();
  }

  // Returns total inserted vectors (including tombstoned ones).
  size_t total_size() const {
    std::shared_lock lock(mutex_);
    return ids_.size();
  }

  size_t deleted_count() const {
    std::shared_lock lock(mutex_);
    return deleted_.size();
  }

  size_t dimensions() const { return dimensions_; }

  bool is_save_in_progress() const { return save_in_progress_.load(); }

  bool is_deleted(VectorId id) const {
    std::shared_lock lock(mutex_);
    return deleted_.count(id) > 0;
  }

  int generate_random_layer();

 private:
  size_t dimensions_;

  // Structure of Arrays (SoA) flat layout
  std::vector<VectorId> ids_;
  std::vector<float>    raw_data_;   // single massive 1D float array

  // Maps external VectorId → sequential insertion index in raw_data_.
  // Decouples logical IDs from physical storage slots (non-sequential IDs ok).
  std::unordered_map<VectorId, size_t> id_to_index_;

  // Per-vector metadata blobs (optional, may be empty json object {})
  std::unordered_map<VectorId, nlohmann::json> metadata_;

  // Soft-delete tombstone set: vectors here are excluded from search results
  // but their HNSW edges remain intact for graph traversal.
  std::unordered_set<VectorId> deleted_;

  // HNSW graph parameters
  size_t M_;       // max connections per node per layer
  size_t M_max0_;  // max connections for layer 0 (= 2*M per paper)
  double mult_;    // 1 / ln(M) — scaling multiplier for level generation

  int max_layer_;  // highest layer currently in the graph (-1 = empty)
  int ep_index_;   // entry-point node index (top of the graph)

  std::vector<HnswNode> nodes_;     // adjacency list (indexed by VectorId)
  std::mt19937          generator_; // for layer dice-roll

  // Reader-writer lock: concurrent searches ok; insert/remove are exclusive.
  // mutable so const methods (search, save) can lock it.
  mutable std::shared_mutex mutex_;

  // Phase 7.3: true while a background save thread is running.
  std::atomic<bool> save_in_progress_{false};

  // Internal helper: crawls one HNSW layer and returns the `ef` closest nodes.
  std::priority_queue<SearchResult, std::vector<SearchResult>, ComapreByDistance>
  search_layer(const std::vector<float>& query, VectorId ep, size_t ef,
               int layer) const;

  // Heal the entry point if it has been tombstoned. Called inside remove().
  // Caller must hold the exclusive lock.
  void heal_entry_point_locked();
};

}  // namespace vecdb