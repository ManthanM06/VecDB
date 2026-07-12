"""
rag_chat.py — Streamlit RAG chat interface powered by VecDB.

Run with:
    streamlit run rag_chat.py

Requires:
  - VecDB server running:    ./build-release/vecdb_server
  - Docs already ingested:   python3 ingest_docs.py docs/
  - Ollama running locally:  ollama serve  (then: ollama pull llama3.2)
    OR set USE_OLLAMA=False to use the HuggingFace fallback (no GPU needed)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import streamlit as st
from sentence_transformers import SentenceTransformer

from vecdb_client import VecDBClient

# ---------------------------------------------------------------------------
# Config — tweak here
# ---------------------------------------------------------------------------
VECDB_URL   = os.getenv("VECDB_URL", "http://localhost:8080")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNKS_FILE = "chunks.json"
TOP_K       = 5                     # chunks to retrieve per query
OLLAMA_MODEL      = "llama3.2"      # or "mistral", "phi3", etc.
USE_OLLAMA        = True            # set False to use a small HF model instead
HF_FALLBACK_MODEL = "google/flan-t5-base"  # lightweight, no GPU needed

# ---------------------------------------------------------------------------
# Page config  (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VecDB RAG Chat",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark, premium feel
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: #0d0f14; }

    .chat-bubble-user {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3);
    }
    .chat-bubble-ai {
        background: #1e2130;
        color: #e2e8f0;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 85%;
        border: 1px solid #2d3748;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        white-space: pre-wrap;
    }
    .source-card {
        background: #161824;
        border: 1px solid #2d3748;
        border-left: 3px solid #4f46e5;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0;
        font-size: 0.82em;
        color: #94a3b8;
    }
    .stTextInput > div > div > input {
        background: #1e2130 !important;
        color: #e2e8f0 !important;
        border: 1px solid #2d3748 !important;
        border-radius: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached resources  (loaded once per session, survive reruns)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner=False)
def load_client() -> VecDBClient:
    return VecDBClient(VECDB_URL)


@st.cache_data(show_spinner=False)
def load_chunks() -> dict[str, dict]:
    p = Path(CHUNKS_FILE)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


@st.cache_resource(show_spinner="Loading HuggingFace LLM…")
def load_hf_llm():
    """Load HuggingFace fallback LLM (only when USE_OLLAMA=False)."""
    from transformers import pipeline  # type: ignore
    return pipeline(
        "text2text-generation",
        model=HF_FALLBACK_MODEL,
        max_new_tokens=256,
    )


# ---------------------------------------------------------------------------
# LLM answer generation
# ---------------------------------------------------------------------------

def generate_answer_ollama(context: str, question: str) -> str:
    """
    Call the locally running Ollama daemon.
    The ollama Python package (v0.2+) returns a ChatResponse *object*,
    not a dict — access fields with dot notation, not subscript.
    """
    try:
        import ollama  # noqa: PLC0415

        prompt = (
            "You are a precise assistant that answers questions using ONLY "
            "the provided context. If the answer is not in the context, say "
            "you don't know — do not make up information.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        # FIX: ollama ≥ 0.2 returns a ChatResponse object, not a dict.
        # Use attribute access instead of dict subscript.
        return response.message.content.strip()

    except Exception as exc:
        err = str(exc)
        # Surface a clean, actionable error instead of raw exception text
        if "connect" in err.lower() or "connection" in err.lower():
            return (
                "⚠️ **Ollama is not running.**\n\n"
                "Start it in a separate terminal:\n"
                "```\nollama serve\n```\n"
                "Then pull the model (first time only):\n"
                "```\nollama pull llama3.2\n```\n\n"
                "Or set `USE_OLLAMA = False` in `rag_chat.py` to use the "
                "built-in HuggingFace fallback instead."
            )
        return f"⚠️ Ollama error: {exc}"


def generate_answer_hf(context: str, question: str) -> str:
    """Use the cached HuggingFace pipeline as LLM."""
    llm = load_hf_llm()
    prompt = f"Context: {context}\n\nAnswer this question: {question}"
    try:
        out = llm(prompt)
        return out[0]["generated_text"].strip()
    except Exception as exc:
        return f"⚠️ HuggingFace error: {exc}"


def generate_answer(context: str, question: str) -> str:
    if USE_OLLAMA:
        return generate_answer_ollama(context, question)
    return generate_answer_hf(context, question)


# ---------------------------------------------------------------------------
# RAG pipeline core
# ---------------------------------------------------------------------------

def rag_query(question: str) -> dict:
    """
    Full RAG pipeline:
      embed → search VecDB → lookup chunk texts → generate LLM answer
    Returns dict: {answer, sources, latency_ms}
    """
    client   = load_client()
    embedder = load_embedder()
    chunks   = load_chunks()

    t0 = time.time()

    # 1. Embed the question
    query_vec = embedder.encode(
        question,
        normalize_embeddings=True,
    ).tolist()

    # 2. Retrieve top-K from VecDB
    results = client.search(query_vec, k=TOP_K)

    if not results:
        return {
            "answer": "⚠️ No results returned from VecDB. Is the server running and are documents ingested?",
            "sources": [],
            "latency_ms": (time.time() - t0) * 1000,
        }

    # 3. Resolve chunk text from sidecar
    retrieved: list[dict] = []
    for r in results:
        chunk_meta = chunks.get(str(r.id), {})
        retrieved.append({
            "id":          r.id,
            "distance":    r.distance,
            "text":        chunk_meta.get("text", "[chunk not found in chunks.json]"),
            "source":      chunk_meta.get("source", "unknown"),
            "chunk_index": chunk_meta.get("chunk_index", 0),
        })

    # 4. Build context string for LLM
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in retrieved
    )

    # 5. Generate answer
    answer = generate_answer(context, question)

    return {
        "answer":     answer,
        "sources":    retrieved,
        "latency_ms": (time.time() - t0) * 1000,
    }


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

if "avg_latency" not in st.session_state:
    st.session_state.avg_latency = 0.0

# ---------------------------------------------------------------------------
# Resolve server / chunk state ONCE at top level so both sidebar and the
# chat input can reference the same variables safely.
# ---------------------------------------------------------------------------
_client    = load_client()
_chunks    = load_chunks()
server_ok  = _client.ping()
chunks_ok  = bool(_chunks)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔍 VecDB RAG")
    st.markdown("*Retrieval-Augmented Generation*")
    st.divider()

    status_icon = "🟢" if server_ok else "🔴"
    st.markdown(f"**Server:** {status_icon} `{VECDB_URL}`")
    st.markdown(f"**Chunks indexed:** `{len(_chunks):,}`")
    st.markdown(f"**Embedding model:** `all-MiniLM-L6-v2`")
    llm_label = f"ollama/{OLLAMA_MODEL}" if USE_OLLAMA else HF_FALLBACK_MODEL
    st.markdown(f"**LLM:** `{llm_label}`")
    st.markdown(f"**Top-K retrieval:** `{TOP_K}`")

    st.divider()
    st.markdown("### 📊 Session Stats")
    col1, col2 = st.columns(2)
    col1.metric("Queries", st.session_state.total_queries)
    col2.metric("Avg Latency", f"{st.session_state.avg_latency:.0f} ms")

    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages      = []
        st.session_state.total_queries = 0
        st.session_state.avg_latency   = 0.0
        st.rerun()

    if not server_ok:
        st.error(
            "⚠️ VecDB server unreachable.\n\n"
            "Start it with:\n```\n./build-release/vecdb_server\n```"
        )

    if not chunks_ok:
        st.warning(
            "⚠️ No chunks found.\n\n"
            "Ingest documents first:\n"
            "```\npython3 ingest_docs.py docs/\n```"
        )

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.markdown("# 💬 VecDB RAG Chat")
st.markdown(
    "Ask any question about your ingested documents. "
    "VecDB retrieves the most semantically relevant passages and grounds the answer."
)
st.divider()

# Render chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="chat-bubble-user">👤 {msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="chat-bubble-ai">🤖 {msg["content"]}</div>',
            unsafe_allow_html=True,
        )
        if msg.get("sources"):
            with st.expander(
                f"📎 {len(msg['sources'])} retrieved chunks  ·  "
                f"⏱ {msg['latency_ms']:.0f} ms"
            ):
                for i, src in enumerate(msg["sources"], start=1):
                    # distance is 1 - cosine_similarity, so similarity = 1 - distance
                    similarity_pct = max(0.0, (1.0 - src["distance"])) * 100
                    st.markdown(
                        f'<div class="source-card">'
                        f'<strong>#{i}</strong> · <code>{src["source"]}</code> '
                        f'· chunk {src["chunk_index"]} '
                        f'· similarity {similarity_pct:.1f}%<br><br>'
                        f'{src["text"][:400]}{"…" if len(src["text"]) > 400 else ""}'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

# Chat input — disabled when server or chunks missing
user_input = st.chat_input(
    placeholder="Ask a question about your documents…",
    disabled=not server_ok or not chunks_ok,
)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("🔍 Retrieving and generating answer…"):
        result = rag_query(user_input)

    # Running average latency
    n        = st.session_state.total_queries
    prev_avg = st.session_state.avg_latency
    st.session_state.total_queries = n + 1
    st.session_state.avg_latency   = (prev_avg * n + result["latency_ms"]) / (n + 1)

    st.session_state.messages.append({
        "role":       "assistant",
        "content":    result["answer"],
        "sources":    result["sources"],
        "latency_ms": result["latency_ms"],
    })

    st.rerun()
