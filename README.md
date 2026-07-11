<div align="center">

# VecDB

**A high-performance vector database built from scratch in C++20.**

SIMD-accelerated · HNSW-indexed · Thread-safe · REST API · RAG-ready

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/CMake-3.24%2B-green.svg)](https://cmake.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## What is VecDB?

VecDB is a vector database written entirely in C++20 — no third-party DB engines, no ML framework wrappers. It implements the full stack from raw memory layout to a REST HTTP server, and connects directly to a Python RAG (Retrieval-Augmented Generation) pipeline.

**Core features:**

- **AVX2 SIMD** dot-product distance computation via FMA intrinsics
- **HNSW index** — approximate nearest-neighbor search with ~92% Recall@10
- **Structure-of-Arrays** flat memory layout for cache-optimal access
- **Full binary persistence** — serializes the entire graph topology to disk
- **Thread-safe** reader-writer locking (`std::shared_mutex`) for concurrent HTTP
- **REST API** via `cpp-httplib` — insert, search, save endpoints
- **Python RAG pipeline** — SDK, document ingestion, and a Streamlit chat UI

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Python Layer                         │
│  rag_chat.py (Streamlit UI)                             │
│  ingest_docs.py (chunk → embed → insert)               │
│  vecdb_client.py (HTTP SDK)                             │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP (localhost:8080)
┌────────────────────────▼────────────────────────────────┐
│               C++ Server (src/server.cpp)               │
│  POST /insert   POST /search   POST /save               │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│             VectorEngine (src/core.cpp)                 │
│                                                         │
│  ┌──────────────┐  ┌────────────────┐  ┌─────────────┐ │
│  │  Flat Store  │  │   HNSW Graph   │  │  Persistence│ │
│  │  raw_data_[] │  │   nodes_[]     │  │  save/load  │ │
│  │  id_to_index │  │  ef_const=128  │  │  .vec format│ │
│  └──────────────┘  └────────────────┘  └─────────────┘ │
│                                                         │
│  AVX2 SIMD simd_dot_product()                           │
│  std::shared_mutex  (readers concurrent, writers excl.) │
└─────────────────────────────────────────────────────────┘
```

### Memory layout

Vectors are stored in a **Structure-of-Arrays** flat layout:

```
raw_data_: [ v0[0] v0[1] ... v0[D] | v1[0] v1[1] ... v1[D] | ... ]
```

All vectors are **L2-normalized at insert time**, converting cosine similarity to a dot product — enabling maximum SIMD throughput without a division per query.

An `id_to_index_` hash map decouples external `VectorId` values from physical storage positions, so non-sequential IDs (e.g. `100`, `200`) work correctly.

### HNSW Index

The **Hierarchical Navigable Small World** index provides sub-linear approximate nearest-neighbor search:

| Parameter | Value | Notes |
|-----------|-------|-------|
| M | 16 | Max edges per node per layer |
| M_max0 | 32 | Max edges in layer 0 |
| ef_construction | 128 | Candidate pool during build |
| ef_search | 200 | Candidate pool during query |
| Recall@10 | ~92% | On 10K random 128-dim vectors |

Insertion uses the correct two-phase algorithm:
1. **Greedy Drop** — descend from `max_layer` to `level+1` updating entry point, no edge wiring
2. **Connect** — search + wire bidirectional edges for all layers ≤ node's level, prune by removing the *worst* (most distant) neighbor when limit exceeded

### Distance metric

```
distance = 1 - dot_product(normalize(query), normalize(stored_vector))
```

Smaller = more similar. Range: `[0, 2]` where `0` is identical.

---

## Project Structure

```
VecDB/
├── src/
│   ├── core.cpp          # VectorEngine: insert, search, save, load
│   ├── server.cpp        # HTTP REST server (cpp-httplib)
│   └── CMakeLists.txt
├── include/
│   └── vecdb/
│       └── core.hpp      # Public API header
├── tests/
│   ├── test_main.cpp     # GoogleTest suite (4 tests)
│   ├── test_data/        # ground_truth.json (generated by scripts/)
│   └── CMakeLists.txt
├── benchmarks/
│   ├── bench_core.cpp    # Google Benchmark: HNSW at 10K/50K/100K
│   └── CMakeLists.txt
├── scripts/
│   └── generate_truth.py # NumPy + scikit-learn ground truth generator
├── docs/                 # Sample documents for RAG demo
│   ├── vecdb_architecture.md
│   └── hnsw_algorithm.md
├── vecdb_client.py       # Python SDK (connection pooling, batch insert)
├── ingest_docs.py        # Document chunker + embedder + ingester
├── rag_chat.py           # Streamlit RAG chat UI
├── load_test.py          # Concurrent HTTP load tester (50 threads, 10K vecs)
├── requirements.txt      # Python dependencies
└── CMakeLists.txt        # Root build: fetches GoogleTest, Benchmark, httplib, nlohmann/json
```

---

## Building

### Prerequisites

- **CMake ≥ 3.24**
- **Ninja** (`sudo apt install ninja-build`)
- **clang++** or **g++** with C++20 support
- CPU with **AVX2** support (any Intel Haswell+ / AMD Ryzen+)
- Internet connection on first build (CMake fetches dependencies via FetchContent)

### Debug build (with AddressSanitizer + UBSanitizer)

```bash
cmake -B build-debug -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug
```

### Release build (optimized, AVX2 + LTO)

```bash
cmake -B build-release -G Ninja -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release
cmake --build build-release
```

---

## Running the Tests

```bash
./build-debug/tests/unit_tests
```

**Test suite (4 tests):**

| Test | What it verifies |
|------|-----------------|
| `GroundTruthTest.CanLoadAndVerifyJsonData` | JSON ground truth file structure |
| `EngineTest.ExactBruteForceMatchesGroundTruth` | HNSW Recall@10 ≥ 90% vs Python brute force |
| `PersistenceTest.CanSaveAndLoadBinaryDatabase` | Full binary round-trip save/load |
| `HnswTest.LayerProbabilityDistribution` | Exponential layer distribution |

Expected output:
```
[  PASSED  ] 4 tests.

[--- HNSW Recall@10 across 100 queries: 92.4% ---]
[--- HNSW LAYER DISTRIBUTION (10000 nodes) ---]
Layer 0: 9347 nodes (93.47%)
Layer 1:  604 nodes (6.04%)
Layer 2:   45 nodes (0.45%)
```

> **First run:** GoogleTest downloads and builds `ground_truth.json` (~80 MB JSON). This takes ~30s. Subsequent runs are instant.
>
> To regenerate ground truth: `python3 scripts/generate_truth.py`

---

## Running the Benchmarks

```bash
./build-release/benchmarks/core_bench
```

Sample output (HNSW search latency):

```
BM_HnswSearch/10000    ~3 ms
BM_HnswSearch/50000    ~5 ms
BM_HnswSearch/100000   ~7 ms
```

HNSW achieves logarithmic scaling — doubling the dataset barely increases latency.

---

## REST API Server

```bash
./build-release/vecdb_server
# [INFO] VecDB Server listening on http://localhost:8080
```

The server is configured for **384 dimensions** to match the `all-MiniLM-L6-v2` embedding model used in the RAG pipeline. Change `VectorEngine db(384, 16)` in `src/server.cpp` if you need a different dimension.

### Endpoints

#### `POST /insert`
```json
{ "id": 42, "vector": [0.12, -0.34, ..., 0.56] }
```
Response: `{"status": "success"}`

#### `POST /search`
```json
{ "k": 5, "vector": [0.12, -0.34, ..., 0.56] }
```
Response:
```json
[
  {"id": 7832, "distance": 0.142},
  {"id": 1205, "distance": 0.198},
  ...
]
```

#### `POST /save`
Flushes the in-memory database to `production_database.vec`. The server reloads this file automatically on next startup.

---

## Load Testing

```bash
python3 load_test.py
```

Fires 10,000 concurrent HTTP inserts using 50 threads, then performs a search and triggers a disk save.

```
Blasting 10000 vectors to VecDB Server (50 concurrent threads)...
Inserted 10000/10000 vectors in 9.34 seconds.

Testing Search Endpoint...
Search HTTP Latency: 2.92 ms
Top 5 Results: [{'distance': 0.67, 'id': 2442}, ...]

Triggering Disk Save...
Database saved successfully to production_database.vec
```

---

## RAG Pipeline (Phase 6)

Connect VecDB to a real embedding model and LLM for Retrieval-Augmented Generation.

### Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 1 — Start the C++ server

```bash
./build-release/vecdb_server
```

### Step 2 — Ingest your documents

Drop any `.txt`, `.md`, or `.pdf` files into `docs/` and run:

```bash
python3 ingest_docs.py docs/
```

This will:
1. Split each document into 300-word overlapping chunks
2. Embed every chunk using `all-MiniLM-L6-v2` (384-dim, runs locally, no GPU needed)
3. Stream all chunk vectors into VecDB via the Python SDK
4. Save a `chunks.json` sidecar mapping vector IDs to text

```
[✓] Connected to VecDB at http://localhost:8080
[✓] Model loaded  |  output dim = 384
[...] Found 2 document(s)
  hnsw_algorithm.md    →  7 chunks
  vecdb_architecture.md → 6 chunks
[✓] Total chunks: 13
  Done: 13/13 vectors inserted in 0.06s (233 vec/s)
[✓] Database saved successfully.
✅  Ingestion complete!
```

### Step 3 — Launch the chat UI

```bash
streamlit run rag_chat.py
```

Open `http://localhost:8501` in your browser.

> **LLM:** Install [Ollama](https://ollama.com) and run `ollama pull llama3.2` for local inference. Or set `USE_OLLAMA = False` in `rag_chat.py` to use the bundled `flan-t5-base` HuggingFace fallback (no GPU needed).

### Python SDK

```python
from vecdb_client import VecDBClient, InsertItem

client = VecDBClient("http://localhost:8080")

# Single insert
client.insert(42, [0.1, 0.2, ..., 0.9])  # 384 floats

# Batch insert (concurrent)
items = [InsertItem(id=i, vector=my_vectors[i]) for i in range(10000)]
client.batch_insert(items, workers=32)

# Search
results = client.search(query_vector, k=5)
for r in results:
    print(f"ID: {r.id}  Distance: {r.distance:.4f}")

# Persist
client.save()
```

---

## Binary File Format (`.vec`)

The persistence format stores the complete database state — including the HNSW graph topology — so the server restores in milliseconds without re-indexing:

```
[4 bytes]  Magic number: 0x31434556 ("VEC1")
[8 bytes]  Vector count
[8 bytes]  Dimensions
[N×4 bytes] VectorId array
[N×D×4 bytes] Float32 vector data (L2-normalized)
[4 bytes]  max_layer (HNSW state)
[4 bytes]  ep_index (entry point)
[variable] id_to_index map
[variable] Full HNSW adjacency list (layers × neighbors per node)
```

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Done | Ground truth generation & test harness |
| 2 | ✅ Done | Core structures + brute-force search |
| 3 | ✅ Done | AVX2 SIMD distance computation |
| 4 | ✅ Done | Binary persistence with full graph serialization |
| 5 | ✅ Done | HNSW index with correct two-phase insertion |
| 6 | ✅ Done | Python RAG pipeline (SDK, ingestion, Streamlit chat) |
| 7 | 🔲 Next | Metadata filtering, vector deletion (tombstoning), async snapshots |
| 8 | 🔲 Future | Write-Ahead Log (WAL), dynamic memory pooling, distributed sharding |

---

## Dependencies

All C++ dependencies are fetched automatically by CMake on first build:

| Library | Version | Use |
|---------|---------|-----|
| [GoogleTest](https://github.com/google/googletest) | v1.14.0 | Unit testing |
| [Google Benchmark](https://github.com/google/benchmark) | v1.8.3 | Performance benchmarks |
| [nlohmann/json](https://github.com/nlohmann/json) | v3.11.3 | JSON parsing |
| [cpp-httplib](https://github.com/yhirose/cpp-httplib) | v0.15.3 | HTTP server |

Python dependencies (install via `pip install -r requirements.txt`):

| Package | Use |
|---------|-----|
| `sentence-transformers` | Local embedding model (`all-MiniLM-L6-v2`) |
| `streamlit` | RAG chat UI |
| `ollama` | Local LLM inference |
| `PyPDF2` | PDF document ingestion |
| `requests` | HTTP client |

---

## License

MIT License — see [LICENSE](LICENSE).
