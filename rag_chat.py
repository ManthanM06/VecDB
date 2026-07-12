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
USE_OLLAMA        = False           # set False to use a small HF model instead
HF_FALLBACK_MODEL = "google/flan-t5-base"

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
# Custom CSS
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
        margin: 6px 0 2px 0;
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
    /* shrink action buttons beneath user bubbles */
    .action-row button {
        font-size: 0.75em !important;
        padding: 2px 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached resources
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
    """
    Load flan-t5-base via AutoModel (transformers v5 dropped the
    'text2text-generation' pipeline task, so we use the model directly).
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(HF_FALLBACK_MODEL)
    model     = AutoModelForSeq2SeqLM.from_pretrained(HF_FALLBACK_MODEL)
    return tokenizer, model


# ---------------------------------------------------------------------------
# LLM answer generation
# ---------------------------------------------------------------------------

def generate_answer_ollama(context: str, question: str) -> str:
    try:
        import ollama  # noqa: PLC0415

        prompt = (
            "You are a precise assistant. Answer the question using ONLY the "
            "provided context. If the answer is not in the context, say you "
            "don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.message.content.strip()

    except Exception as exc:
        err = str(exc)
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
    """
    Inference with flan-t5-base.

    Prompt strategy:
    - Use only the single most-relevant chunk (first in retrieved list) to
      stay well within flan-t5-base's 512-token limit.
    - Phrase as a direct extractive QA instruction — flan-t5 was trained on
      tasks like this and handles them better than open-ended generation.
    - Beam search (num_beams=4) gives more coherent answers than greedy.
    """
    tokenizer, model = load_hf_llm()

    # Use only the highest-similarity chunk to keep the prompt tight
    best_chunk = context.split("\n\n---\n\n")[0]

    prompt = (
        f"Based on the following text, answer this question: {question}\n\n"
        f"Text: {best_chunk}\n\n"
        f"Answer:"
    )

    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=4,           # beam search → more coherent output
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if not answer:
            answer = "I couldn't find a clear answer in the retrieved passages."
        return answer
    except Exception as exc:
        return f"⚠️ HuggingFace error: {exc}"


def generate_answer(context: str, question: str) -> str:
    if USE_OLLAMA:
        return generate_answer_ollama(context, question)
    return generate_answer_hf(context, question)


# ---------------------------------------------------------------------------
# RAG pipeline core
# ---------------------------------------------------------------------------

# (rag_query logic moved directly into _submit for dynamic status updates)


# ---------------------------------------------------------------------------
# Helper: submit a query and update session state
# ---------------------------------------------------------------------------

def _submit(question: str) -> None:
    """Run RAG query with dynamic status updates and append to history."""
    st.session_state.messages.append({"role": "user", "content": question})

    with st.status("Thinking...", expanded=True) as status:
        client   = load_client()
        embedder = load_embedder()
        chunks   = load_chunks()

        t0 = time.time()
        
        status.update(label="Embedding query...", state="running")
        query_vec = embedder.encode(question, normalize_embeddings=True).tolist()
        
        status.update(label="Retrieving context from VecDB...", state="running")
        results   = client.search(query_vec, k=TOP_K)

        if not results:
            answer = "⚠️ No results from VecDB. Is the server running and are documents ingested?"
            retrieved = []
        else:
            status.update(label="Formatting context...", state="running")
            retrieved: list[dict] = []
            for r in results:
                meta = chunks.get(str(r.id), {})
                retrieved.append({
                    "id":          r.id,
                    "distance":    r.distance,
                    "text":        meta.get("text", "[chunk not found]"),
                    "source":      meta.get("source", "unknown"),
                    "chunk_index": meta.get("chunk_index", 0),
                })
            
            context = "\n\n---\n\n".join(
                f"[Source: {c['source']}]\n{c['text']}" for c in retrieved
            )
            
            status.update(label="Generating response...", state="running")
            answer = generate_answer(context, question)
        
        latency_ms = (time.time() - t0) * 1000
        status.update(label=f"Done in {latency_ms:.0f} ms", state="complete")

    n        = st.session_state.total_queries
    prev_avg = st.session_state.avg_latency
    st.session_state.total_queries = n + 1
    st.session_state.avg_latency   = (prev_avg * n + latency_ms) / (n + 1)

    st.session_state.messages.append({
        "role":       "assistant",
        "content":    answer,
        "sources":    retrieved,
        "latency_ms": latency_ms,
    })


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

for _key, _val in [
    ("messages",      []),
    ("total_queries", 0),
    ("avg_latency",   0.0),
    ("pending_rerun", None),   # stores prompt text for re-run requests
]:
    if _key not in st.session_state:
        st.session_state[_key] = _val

# ---------------------------------------------------------------------------
# Top-level state: server + chunks (used by sidebar AND chat input)
# ---------------------------------------------------------------------------
_client   = load_client()
_chunks   = load_chunks()
server_ok = _client.ping()
chunks_ok = bool(_chunks)

# ---------------------------------------------------------------------------
# Handle re-run requests BEFORE rendering the UI so the new message
# appears immediately without a second click.
# ---------------------------------------------------------------------------
if st.session_state.pending_rerun is not None:
    prompt_to_rerun = st.session_state.pending_rerun
    st.session_state.pending_rerun = None
    _submit(prompt_to_rerun)
    st.rerun()

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

# Render conversation history
for idx, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        # ── User bubble ──────────────────────────────────────────────────
        st.markdown(
            f'<div class="chat-bubble-user">👤 {msg["content"]}</div>',
            unsafe_allow_html=True,
        )

        # Action row: Copy · Re-run
        # Push buttons to the right to align nicely under the user bubble
        col_spacer, col_copy, col_rerun = st.columns([8, 0.5, 0.5])

        with col_copy:
            if st.button("", icon=":material/content_copy:", key=f"copy_{idx}", help="Show prompt text to copy"):
                # Toggle a per-message "show copy" flag
                flag = f"show_copy_{idx}"
                st.session_state[flag] = not st.session_state.get(flag, False)

        with col_rerun:
            if st.button("", icon=":material/refresh:", key=f"rerun_{idx}", help="Submit this prompt again"):
                st.session_state.pending_rerun = msg["content"]
                st.rerun()

        # If copy is toggled on, show a code block (has a native copy button)
        if st.session_state.get(f"show_copy_{idx}", False):
            st.code(msg["content"], language=None)

    else:
        # ── Assistant bubble ─────────────────────────────────────────────
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
                    similarity_pct = max(0.0, 1.0 - src["distance"]) * 100
                    st.markdown(
                        f'<div class="source-card">'
                        f'<strong>#{i}</strong> · <code>{src["source"]}</code> '
                        f'· chunk {src["chunk_index"]} '
                        f'· similarity {similarity_pct:.1f}%<br><br>'
                        f'{src["text"][:400]}{"…" if len(src["text"]) > 400 else ""}'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
user_input = st.chat_input(
    placeholder="Ask a question about your documents…",
    disabled=not server_ok or not chunks_ok,
)

if user_input:
    _submit(user_input)
    st.rerun()
