#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "vecdb/core.hpp"

// Helper function to generate random floats
std::vector<float> generate_random_vector(size_t dim, std::mt19937& gen) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> vec(dim);
  for (size_t i = 0; i < dim; ++i) {
    vec[i] = dist(gen);
  }
  return vec;
}

// The benchmark fixture
static void BM_BruteForceSearch(benchmark::State& state) {
  size_t num_vectors = static_cast<size_t>(
      state.range(0));  // This takes the arguments (1000, 5000, 10000)
  size_t dim = 128;

  vecdb::VectorEngine engine(dim);
  std::mt19937 gen(42);  // Fixed seed so our benchmarks are consistent

  // Setup: Populate the database with random vectors
  for (size_t i = 0; i < num_vectors; ++i) {
    engine.insert(static_cast<vecdb::VectorId>(i),
                  generate_random_vector(dim, gen));
  }

  std::vector<float> query = generate_random_vector(dim, gen);

  // The core measurement loop
  for (auto _ : state) {
    auto results = engine.search(query, 10);

    // CRITICAL: If we don't use 'results', the C++ compiler is smart enough
    // to realize this loop does nothing and will completely delete the code!
    // DoNotOptimize forces the CPU to actually do the math.
    benchmark::DoNotOptimize(results);
  }
}

// Tell Google Benchmark to run this test at 3 different scale sizes
BENCHMARK(BM_BruteForceSearch)->Arg(100)->Arg(1000)->Arg(5000)->Arg(10000);

BENCHMARK_MAIN();