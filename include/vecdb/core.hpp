#pragma once

#include <cstdint>
#include <queue>
#include <random>        // for HNSW probability generation
#include <shared_mutex>  // for reader-writer thread safety
#include <stdexcept>     // for std::runtime_error
#include <string>
#include <unordered_map>
#include <vector>

namespace vecdb {

using VectorId = uint32_t;

struct SearchResult {
  VectorId id;
  float distance;
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

class VectorEngine {
 public:
  // We add M (max connections) to the constructor, defaulting to 16
  explicit VectorEngine(size_t dimensions, size_t M = 16);

  void insert(VectorId id, const std::vector<float>& data);
  std::vector<SearchResult> search(const std::vector<float>& query,
                                   size_t k) const;

  // Persistence methods
  void save(const std::string& file_path) const;
  void load(const std::string& file_path);

  size_t size() const {
    std::shared_lock lock(mutex_);
    return ids_.size();
  }
  size_t dimensions() const { return dimensions_; }

  int generate_random_layer();

 private:
  size_t dimensions_;

  // Structure of Arrays (SoA) Flat Layout
  std::vector<VectorId> ids_;
  std::vector<float> raw_data_;  // A single massive 1D array of floats

  // Maps an external VectorId to its sequential insertion index in raw_data_.
  // This decouples the logical ID from the physical storage slot, allowing
  // non-sequential IDs (e.g. 100, 200) to be inserted safely.
  std::unordered_map<VectorId, size_t> id_to_index_;

  // HNSW graph state
  size_t M_;       // max connection a node can have per layer
  size_t M_max0_;  // max connections for layer 0
  double mult_;    // the scaling multiplier: 1/ln(M)

  int max_layer_;  // the highest layer currently exsiting in the whole graph
  int ep_index_;   // entry point index (index of the node at very top)

  std::vector<HnswNode> nodes_;  // adj list
  std::mt19937 generator_;       // for dice roll

  // Reader-writer lock: multiple concurrent searches are safe; inserts are
  // exclusive. mutable so const methods (search, save) can lock it too.
  mutable std::shared_mutex mutex_;

  // Crawls a single layer of the graph and returns the 'ef' closest neighbors
  std::priority_queue<SearchResult, std::vector<SearchResult>,
                      ComapreByDistance>
  search_layer(const std::vector<float>& query, VectorId ep, size_t ef,
               int layer) const;
};

}  // namespace vecdb