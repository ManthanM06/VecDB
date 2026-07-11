# VecDB Project Roadmap: The Next Frontier

## Project Status
**Core Engine Complete:** SIMD-accelerated, HNSW-backed, thread-safe vector database with binary disk persistence and a REST API.

---

## Phase 6: Real-World AI Integration (The RAG Pipeline)
**Objective:** Move beyond benchmarking with random floats and connect VecDB to a live Large Language Model ecosystem to build a functional AI application.

**Sub-phase 6.1: The Python Client SDK**
* Design a dedicated `VecDBClient` Python class to cleanly wrap the C++ REST API.
* Implement connection pooling, batch-insert logic, and robust error handling.
* **Deliverable:** `vecdb_client.py` library.

**Sub-phase 6.2: Embedding Generation & Ingestion**
* Integrate a real text-embedding model (e.g., `sentence-transformers` via HuggingFace or OpenAI's `text-embedding-3-small`).
* Write a document chunker to split local PDFs or Markdown files into semantic, overlapping paragraphs.
* **Deliverable:** `ingest_docs.py` script that reads a directory of documents, vectorizes them, and streams them into the VecDB server.

**Sub-phase 6.3: The End-to-End RAG Application**
* Build a local CLI or Streamlit chat interface.
* Implement the Retrieval-Augmented Generation workflow: Embed user query -> Query VecDB -> Retrieve Top-K semantic chunks -> Inject into LLM prompt -> Generate answer.
* **Deliverable:** `rag_chat.py` (a fully functional AI assistant grounded in your local data).

---

## Phase 7: Advanced Database Mechanics (CRUD & Filtering)
**Objective:** Upgrade VecDB from a pure mathematical index to a fully operational, production-ready database.

**Sub-phase 7.1: Metadata Storage & Pre-Filtering**
* Add the ability to store arbitrary JSON metadata alongside Vector IDs (e.g., `{"author": "parth", "year": 2026}`).
* Implement metadata filtering directly inside the HNSW search crawler to skip non-matching nodes without sacrificing search speed.
* **Deliverable:** Updated C++ `insert` and `search` endpoints supporting JSON payloads, backed by an internal key-value store.

**Sub-phase 7.2: Vector Deletion (Tombstoning)**
* Explicitly deleting nodes and severing HNSW edges destroys graph navigability. Implement "Soft Deletion" (Tombstoning).
* Add a bitset array to mark vectors as deleted, skipping them during `search()` and `insert()` routing.
* **Deliverable:** A new `POST /delete` endpoint and updated graph traversal logic.

**Sub-phase 7.3: Background Snapshotting (Non-Blocking IO)**
* Currently, `/save` holds a lock. Writing gigabytes to disk will block incoming HTTP requests.
* Implement double-buffering or copy-on-write mechanics so the DB can dump RAM to disk in a background thread while continuing to serve traffic.
* **Deliverable:** Refactored `src/server.cpp` with asynchronous disk I/O.

---

## Phase 8: Enterprise Reliability (The Distributed Horizon)
**Objective:** Make the database crash-proof, horizontally scalable, and ready for massive datasets.

**Sub-phase 8.1: Write-Ahead Logging (WAL)**
* If the server loses power between disk saves, all recent RAM inserts are permanently lost.
* Build an append-only WAL file that records every `/insert` operation instantly to the SSD before responding with HTTP 200.
* **Deliverable:** A fast WAL C++ module that automatically replays missed operations upon server startup.

**Sub-phase 8.2: Dynamic Memory Pooling**
* Relying on `std::vector::resize()` for massive arrays causes expensive memory reallocations and CPU stalling.
* Implement chunked memory pooling (segmented arrays) to dynamically handle 100M+ vectors without blocking execution threads.
* **Deliverable:** Highly optimized `raw_data_` and `nodes_` memory allocators in `src/core.cpp`.