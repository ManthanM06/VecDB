# VecDB — Vector Database in C++

## Overview

VecDB is a from-scratch, high-performance vector database implemented in C++20. It is designed to store, index, and search high-dimensional floating-point vectors with millisecond latency.

## Architecture

The core engine (`src/core.cpp`) implements three major systems:

### 1. Flat Storage (Structure of Arrays)

Vectors are stored in a single contiguous flat array `raw_data_` of type `std::vector<float>`. Every inserted vector occupies exactly `dimensions` consecutive floats. This layout maximizes cache utilization during SIMD distance computations.

An `id_to_index_` hash map translates arbitrary external VectorIds to sequential positions in the flat array, allowing non-sequential insertion patterns without wasting memory.

### 2. HNSW Index (Hierarchical Navigable Small World Graphs)

VecDB uses HNSW — a state-of-the-art approximate nearest neighbor algorithm with logarithmic search complexity. Vectors are assigned to layers using an exponential probability distribution. Most vectors (about 93%) live only in layer 0. A small fraction appear in higher layers, forming a highway network for fast traversal.

During insertion, new nodes go through two phases:
- **Phase 1 (Greedy Drop):** Descend from the top layer down to the new node's level, following the closest neighbor at each layer. This finds the best entry point with zero wiring.
- **Phase 2 (Connect):** At each layer from the node's level down to 0, run a beam search with `ef_construction=128` candidates, then wire bidirectional edges to the best M neighbors.

Pruning keeps each node's neighbor list bounded to M connections. When a neighbor list overflows, the most distant (worst) connection is removed to preserve graph navigability.

### 3. SIMD Distance Computation

Distance is computed as `1 - dot_product`, using AVX2 FMA intrinsics (`_mm256_fmadd_ps`). Since all vectors are L2-normalized at insert time, dot product between unit vectors equals cosine similarity. This lets us use the fast inner product instead of the more expensive cosine formula.

## API Endpoints

The server (`src/server.cpp`) exposes a REST API via cpp-httplib on port 8080:

- `POST /insert` — Insert a vector: `{"id": 123, "vector": [...]}`
- `POST /search` — Find K nearest: `{"k": 10, "vector": [...]}`
- `POST /save`   — Flush database to disk

## Thread Safety

All public methods on `VectorEngine` are protected by a `std::shared_mutex`. Writers (`insert`, `load`) take an exclusive lock. Readers (`search`, `save`) take a shared lock, allowing multiple concurrent searches.

## Persistence Format

The binary `.vec` file format stores:
1. Magic number (4 bytes): `0x31434556` ("VEC1")
2. Vector count and dimensions (metadata header)
3. Raw float arrays (ids and vectors)
4. Graph state: `max_layer`, `ep_index`, `id_to_index` map, full HNSW adjacency list

## Performance

On a machine with AVX2 support, the release build achieves:
- Insert throughput: ~1,000–5,000 vectors/second (depends on M and ef_construction)
- Search latency: 1–5 ms for K=10 on 10K vectors with ef_search=200
- Recall@10 on random 128-dim vectors: approximately 92%

## Building

```bash
# Debug (with ASan/UBSan sanitizers)
cmake -B build-debug -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug

# Release (optimized)
cmake -B build-release -G Ninja -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release
cmake --build build-release
```

## Phase Roadmap

- Phase 1: Ground truth data generation and test harness
- Phase 2: Core structures and brute-force search
- Phase 3: Hardware optimization with AVX2/AVX-512 SIMD
- Phase 4: Binary persistence layer with full graph serialization
- Phase 5: HNSW index with correct two-phase insertion
- Phase 6: Python RAG pipeline integration (current phase)
- Phase 7: Metadata filtering, vector deletion, background snapshots
- Phase 8: Write-Ahead Logging and distributed architecture
