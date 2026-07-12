"""
ingest_docs.py — Chunk, embed, and load documents into VecDB.

Usage:
    python3 ingest_docs.py [docs_directory]

This script:
  1. Recursively scans a directory for .txt, .md, and .pdf files.
  2. Splits each file into overlapping text chunks.
  3. Encodes every chunk with a local sentence-transformers model (384-dim).
  4. Streams all chunks into the running VecDB server.
  5. Saves a sidecar `chunks.json` so the RAG app can look up text by vector ID.

The VecDB server must be running at http://localhost:8080 with 384 dimensions.
Start it with:  ./build-release/vecdb_server
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer
from vecdb_client import InsertItem, VecDBClient

try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHUNK_SIZE = 300     # words per chunk
CHUNK_STRIDE = 100   # words of overlap between consecutive chunks
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, ~80MB
EMBED_BATCH = 64     # sentences per embedding batch
VECDB_URL = "http://localhost:8080"
CHUNKS_FILE = "chunks.json"   # sidecar: {id -> {"text": ..., "source": ...}}

# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    if not PDF_SUPPORT:
        print(f"  [warn] PyPDF2 not installed, skipping {path.name}")
        return ""
    text_parts: list[str] = []
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
    except Exception as exc:
        print(f"  [warn] Could not read PDF {path.name}: {exc}")
    return "\n".join(text_parts)


def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        return read_txt(path)
    if ext == ".pdf":
        return read_pdf(path)
    return ""


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    source: str,
    chunk_size: int = CHUNK_SIZE,
    stride: int = CHUNK_STRIDE,
) -> list[dict]:
    """
    Split text into overlapping word windows.
    Returns list of dicts: {"text": str, "source": str, "chunk_index": int}
    """
    words = text.split()
    chunks = []
    i = 0
    chunk_index = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunk_text_str = " ".join(chunk_words).strip()
        if chunk_text_str:
            chunks.append({
                "text": chunk_text_str,
                "source": source,
                "chunk_index": chunk_index,
            })
        i += stride
        chunk_index += 1
    return chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest documents into VecDB for RAG."
    )
    parser.add_argument(
        "docs_dir",
        nargs="?",
        default="docs",
        help="Directory containing .txt / .md / .pdf files (default: ./docs)",
    )
    parser.add_argument(
        "--url",
        default=VECDB_URL,
        help=f"VecDB server URL (default: {VECDB_URL})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear chunks.json before ingesting (start fresh)",
    )
    args = parser.parse_args()

    docs_path = Path(args.docs_dir)
    if not docs_path.is_dir():
        print(f"[error] Directory not found: {docs_path}")
        sys.exit(1)

    # ---- Connect to VecDB --------------------------------------------------
    client = VecDBClient(args.url)
    if not client.ping():
        print(f"[error] Cannot reach VecDB server at {args.url}")
        print("        Start it with:  ./build-release/vecdb_server")
        sys.exit(1)
    print(f"[✓] Connected to VecDB at {args.url}")

    # ---- Load embedding model ----------------------------------------------
    print(f"[...] Loading embedding model: {EMBED_MODEL}")
    print("      (first run downloads ~80 MB — subsequent runs use cache)")
    model = SentenceTransformer(EMBED_MODEL)
    print(f"[✓] Model loaded  |  output dim = {model.get_embedding_dimension()}")

    # ---- Scan documents ----------------------------------------------------
    extensions = {".txt", ".md", ".pdf"}
    doc_files = [p for p in docs_path.rglob("*") if p.suffix.lower() in extensions]
    if not doc_files:
        print(f"[warn] No .txt / .md / .pdf files found in {docs_path}")
        sys.exit(0)

    print(f"\n[...] Found {len(doc_files)} document(s):")
    for f in doc_files:
        print(f"       • {f.relative_to(docs_path)}")

    # ---- Chunk all documents -----------------------------------------------
    all_chunks: list[dict] = []
    for doc_file in doc_files:
        text = extract_text(doc_file)
        if not text.strip():
            print(f"  [warn] Empty or unreadable: {doc_file.name}")
            continue
        chunks = chunk_text(text, source=str(doc_file.relative_to(docs_path)))
        print(f"  {doc_file.name}  →  {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\n[✓] Total chunks: {len(all_chunks)}")

    # ---- Load existing sidecar or start fresh ------------------------------
    chunks_path = Path(CHUNKS_FILE)
    existing_chunks: dict[str, dict] = {}
    if chunks_path.exists() and not args.reset:
        with open(chunks_path) as f:
            existing_chunks = json.load(f)
        print(f"[✓] Loaded {len(existing_chunks)} existing chunks from {CHUNKS_FILE}")

    # Assign IDs starting after the last existing ID
    start_id = max((int(k) for k in existing_chunks), default=-1) + 1

    # ---- Embed all chunks --------------------------------------------------
    print(f"\n[...] Embedding {len(all_chunks)} chunks in batches of {EMBED_BATCH}...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,   # already unit-length → VecDB norm is a no-op
    )

    # ---- Build insert items + sidecar entries ------------------------------
    insert_items: list[InsertItem] = []
    new_chunks: dict[str, dict] = {}

    for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
        vec_id = start_id + i
        insert_items.append(InsertItem(id=vec_id, vector=embedding.tolist()))
        new_chunks[str(vec_id)] = {
            "text": chunk["text"],
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
        }

    # ---- Stream into VecDB -------------------------------------------------
    print(f"\n[...] Inserting {len(insert_items)} vectors into VecDB...")
    successful, total = client.batch_insert(insert_items, workers=32)

    if successful < total:
        print(f"[warn] Only {successful}/{total} inserts succeeded.")

    # ---- Save sidecar ------------------------------------------------------
    merged_chunks = {**existing_chunks, **new_chunks}
    with open(chunks_path, "w") as f:
        json.dump(merged_chunks, f, indent=2)
    print(f"[✓] Saved {len(merged_chunks)} chunks to {CHUNKS_FILE}")

    # ---- Persist to disk ---------------------------------------------------
    print("[...] Triggering VecDB disk save...")
    if client.save():
        print("[✓] Database saved successfully.")
    else:
        print("[warn] Disk save failed — data is in RAM only.")

    print(f"\n✅  Ingestion complete!  {successful} vectors ready for search.")
    print(f"    Run the chat app:  streamlit run rag_chat.py")


if __name__ == "__main__":
    main()
