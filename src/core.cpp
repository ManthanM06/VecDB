#include "vecdb/core.hpp"

#include <algorithm>
#include <numeric>
#include <queue>
#include <stdexcept>

namespace vecdb {

VectorEngine::VectorEngine(size_t dimensions) : dimensions_(dimensions) {}

void VectorEngine::insert(VectorId id, const std::vector<float>& data) {
  if (data.size() != dimensions_) {
    throw std::invalid_argument(
        "Vector dimensions do not match database dimensions.");
  }
  vectors_.push_back({id, data});
}

std::vector<SearchResult> VectorEngine::search(const std::vector<float>& query,
                                               size_t k) const {
  if (query.size() != dimensions_) {
    throw std::invalid_argument(
        "Query dimensions do not match database dimensions.");
  }
  if (vectors_.empty() || k == 0) {
    return {};
  }

  // A Max-Heap keeps the largest distance at the top.
  // We want to keep the K SMALLEST distances.
  auto cmp = [](const SearchResult& left, const SearchResult& right) {
    return left.distance < right.distance;
  };
  std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(cmp)>
      max_heap(cmp);

  // Brute-force linear scan
  for (const auto& vec : vectors_) {
    // Calculate Dot Product using standard C++
    float dot_product = std::inner_product(vec.data.begin(), vec.data.end(),
                                           query.begin(), 0.0f);

    // Convert to Cosine Distance (Smaller = Closer)
    float distance = 1.0f - dot_product;

    // Maintain the Top-K heap
    if (max_heap.size() < k) {
      max_heap.push({vec.id, distance});
    } else if (distance < max_heap.top().distance) {
      max_heap.pop();                     // Kick out the worst one
      max_heap.push({vec.id, distance});  // Insert the better one
    }
  }

  // Extract results from the heap
  std::vector<SearchResult> results;
  results.reserve(max_heap.size());
  while (!max_heap.empty()) {
    results.push_back(max_heap.top());
    max_heap.pop();
  }

  // Heaps pop largest first, so reverse to get closest first (ascending order)
  std::reverse(results.begin(), results.end());
  return results;
}

}  // namespace vecdb