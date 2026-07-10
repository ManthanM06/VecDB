#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "vecdb/core.hpp"

std::vector<float> generate_random_vector(size_t dim, std::mt19937& gen) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> vec(dim);
  for (size_t i = 0; i < dim; ++i) {
    vec[i] = dist(gen);
  }
  return vec;
}

static void BM_HnswSearch(benchmark::State& state) {
  size_t num_vectors = static_cast<size_t>(state.range(0));
  size_t dim = 128;

  // Initialize engine with M=16
  vecdb::VectorEngine engine(dim, 16);
  std::mt19937 gen(42);

  // Setup: Populate the graph
  for (size_t i = 0; i < num_vectors; ++i) {
    engine.insert(static_cast<vecdb::VectorId>(i),
                  generate_random_vector(dim, gen));
  }

  std::vector<float> query = generate_random_vector(dim, gen);

  // The core measurement loop
  for (auto _ : state) {
    // Search for Top-10 closest neighbors
    auto results = engine.search(query, 10);
    benchmark::DoNotOptimize(results);
  }
}

// Test logarithmic scaling across massive jumps in data size
BENCHMARK(BM_HnswSearch)->Arg(10000)->Arg(50000)->Arg(100000);

BENCHMARK_MAIN();