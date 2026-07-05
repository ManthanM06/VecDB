#pragma once

#include <cstdint>
#include <vector>

namespace vecdb {

using VectorId = uint32_t;

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

  // Notice size() now checks ids_.size()
  size_t size() const { return ids_.size(); }
  size_t dimensions() const { return dimensions_; }

 private:
  size_t dimensions_;

  // Structure of Arrays (SoA) Flat Layout
  std::vector<VectorId> ids_;
  std::vector<float> raw_data_;  // A single massive 1D array of floats
};

}  // namespace vecdb