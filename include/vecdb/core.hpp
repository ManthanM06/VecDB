#pragma once

#include <cstdint>
#include <vector>

namespace vecdb {
// Explicitly define an ID type for readability and future scaling (e.g.,
// transitioning to 64-bit IDs)
using VectorId = uint32_t;

// a single vector definition (ID + floating point data)
struct Vector {
  VectorId id;
  std::vector<float> data;
};

// struture to return search results to the user
struct SearchResult {
  VectorId id;
  float distance;
};

class VectorEngine {
 public:
  explicit VectorEngine(size_t dimensions);

  void insert(VectorId id, const std::vector<float>& data);

  std::vector<SearchResult> search(const std::vector<float>& query,
                                   size_t k) const;

  size_t size() const { return vectors_.size(); }
  size_t dimensions() const { return dimensions_; }

 private:
  size_t dimensions_;
  std::vector<Vector> vectors_;
};

}  // namespace vecdb