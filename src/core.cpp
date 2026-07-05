#include "vecdb/core.hpp"

#include <immintrin.h>  //hardware intrinsics header

#include <algorithm>
#include <queue>
#include <stdexcept>

namespace vecdb {
VectorEngine::VectorEngine(size_t dimensions) : dimensions_(dimensions) {}

void VectorEngine::insert(VectorId id, const std::vector<float>& data) {
  if (data.size() != dimensions_) {
    throw std::invalid_argument(
        "Vector dimensions do not match databse dimensiosn.");
  }
  ids_.push_back(id);
  raw_data_.insert(raw_data_.end(), data.begin(), data.end());
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
}  // namespace vecdb