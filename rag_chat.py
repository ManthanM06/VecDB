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
from threading import Thread

import streamlit as st
import streamlit.components.v1 as components
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300;0,14..32,400;0,14..32,500;0,14..32,600;0,14..32,700&display=swap');

    /* ── Global ─────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        -webkit-font-smoothing: antialiased;
    }

    /* App background */
    .stApp {
        background: #080a0f;
    }
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 6rem;
    }

    /* ── Sidebar ────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: rgba(12, 14, 20, 0.95);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stSidebar"] .stMarkdown h2 {
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        color: #a5b4fc;
        margin-bottom: 0;
    }

    /* ── Page header ────────────────────────────────────────── */
    .page-header {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 6px;
    }
    .page-header h1 {
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #a5b4fc 0%, #818cf8 50%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
    }
    .page-subtitle {
        font-size: 0.85rem;
        color: #64748b;
        margin-bottom: 1rem;
        font-weight: 400;
        letter-spacing: 0.01em;
    }

    /* ── Chat bubbles ────────────────────────────────────────── */
    .chat-bubble-user {
        background: linear-gradient(145deg, #312e81, #1e1b4b);
        color: #e0e7ff;
        padding: 11px 16px;
        border-radius: 20px 20px 5px 20px;
        margin: 4px 0 2px 0;
        max-width: 75%;
        margin-left: auto;
        width: fit-content;
        border: 1px solid rgba(99, 102, 241, 0.4);
        box-shadow: 0 4px 24px rgba(79, 70, 229, 0.25),
                    0 1px 0 rgba(255,255,255,0.05) inset;
        font-size: 0.925rem;
        line-height: 1.55;
        letter-spacing: 0.01em;
    }
    .chat-bubble-ai {
        background: rgba(30, 33, 48, 0.85);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        color: #cbd5e1;
        padding: 13px 17px;
        border-radius: 5px 20px 20px 20px;
        margin: 6px 0;
        max-width: 82%;
        width: fit-content;
        border: 1px solid rgba(255,255,255,0.07);
        box-shadow: 0 4px 24px rgba(0,0,0,0.35);
        white-space: pre-wrap;
        font-size: 0.925rem;
        line-height: 1.6;
        letter-spacing: 0.005em;
    }
    /* streaming cursor pulse */
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }
    .chat-bubble-ai .cursor { animation: blink 0.9s ease infinite; }

    /* ── Source cards ────────────────────────────────────────── */
    .source-card {
        background: rgba(15, 17, 26, 0.9);
        border: 1px solid rgba(255,255,255,0.06);
        border-left: 2px solid #6366f1;
        border-radius: 10px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.8rem;
        color: #64748b;
        transition: border-color 0.2s;
    }
    .source-card:hover { border-left-color: #a5b4fc; color: #94a3b8; }
    .source-card code {
        background: rgba(99,102,241,0.15);
        color: #a5b4fc;
        padding: 1px 6px;
        border-radius: 4px;
        font-size: 0.78rem;
    }

    /* ── Bottom input bar ────────────────────────────────────── */
    [data-testid="stBottomBlockContainer"] {
        background: rgba(8, 10, 15, 0.92);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-top: 1px solid rgba(255,255,255,0.06);
        padding: 12px 1rem 18px 1rem;
    }
    /* input field */
    [data-testid="stBottomBlockContainer"] [data-testid="stTextInput"] input {
        background: rgba(22, 25, 38, 0.9) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
        padding: 10px 14px !important;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    [data-testid="stBottomBlockContainer"] [data-testid="stTextInput"] input:focus {
        border-color: rgba(99, 102, 241, 0.7) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.12) !important;
    }

    /* ── Buttons ─────────────────────────────────────────────── */
    [data-testid="stButton"] > button {
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        transition: all 0.18s ease !important;
    }
    /* submit / action button in bottom bar */
    [data-testid="stBottomBlockContainer"] [data-testid="stButton"] > button {
        background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
        border: none !important;
        color: #fff !important;
        height: 42px !important;
        border-radius: 11px !important;
    }
    [data-testid="stBottomBlockContainer"] [data-testid="stButton"] > button:hover {
        background: linear-gradient(135deg, #6366f1, #818cf8) !important;
        box-shadow: 0 4px 16px rgba(99,102,241,0.4) !important;
        transform: translateY(-1px) !important;
    }
    /* stop button variant */
    [data-testid="stBottomBlockContainer"] [data-testid="stButton"] > button[kind="secondary"] {
        background: rgba(239, 68, 68, 0.15) !important;
        border: 1px solid rgba(239, 68, 68, 0.35) !important;
        color: #f87171 !important;
    }

    /* ── Divider ─────────────────────────────────────────────── */
    hr { border-color: rgba(255,255,255,0.06) !important; }

    /* ── Streamlit toolbar ──────────────────────────────── */
    header[data-testid="stHeader"] {
        background: rgba(8, 10, 15, 0.9) !important;
        backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }

    /* ── Placeholder + disabled input ─────────────────────── */
    [data-testid="stBottomBlockContainer"] [data-testid="stTextInput"] input::placeholder {
        color: rgba(148, 163, 184, 0.45) !important;
    }
    [data-testid="stBottomBlockContainer"] [data-testid="stTextInput"] input:disabled {
        opacity: 0.5 !important;
    }

    /* ── Bottom bar input height ────────────────────────── */
    [data-testid="stBottomBlockContainer"] [data-testid="stTextInput"] input {
        height: 44px !important;
    }

    /* ── Expander (sources) ────────────────────────────── */
    [data-testid="stExpander"] {
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 10px !important;
        background: rgba(15, 17, 26, 0.6) !important;
    }

    /* ── Scrollbar ─────────────────────────────────────── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.25); border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.45); }
    </style>
    """,
    unsafe_allow_html=True,
)
# ---------------------------------------------------------------------------
# Copy button HTML component generator
# ---------------------------------------------------------------------------
def render_copy_button(text_to_copy: str):
    escaped_text = text_to_copy.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$')
    html = f"""
    <style>
    body {{ margin: 0; padding: 0; background: transparent; display: flex; align-items: center; justify-content: center; height: 100vh; }}
    button {{
        background: transparent; border: 1px solid rgba(250, 250, 250, 0.2); 
        cursor: pointer; color: #a1a1aa; border-radius: 6px; 
        width: 100%; height: 36px;
        display: flex; align-items: center; justify-content: center;
        transition: all 0.2s; margin-top: 1px;
    }}
    button:hover {{ color: #fff; border-color: #4f46e5; background: rgba(79, 70, 229, 0.1); }}
    svg {{ width: 16px; height: 16px; }}
    </style>
    <button onclick="copyToClipboard()" title="Copy prompt">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
        </svg>
    </button>
    <script>
    function copyToClipboard() {{
        var temp = document.createElement("textarea");
        temp.value = `{escaped_text}`;
        document.body.appendChild(temp);
        temp.select();
        document.execCommand("copy");
        document.body.removeChild(temp);
        
        const btn = document.querySelector('button');
        const oldIcon = btn.innerHTML;
        btn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>';
        setTimeout(() => {{ btn.innerHTML = oldIcon; }}, 2000);
    }}
    </script>
    """
    components.html(html, height=38)

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
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

@st.cache_resource(show_spinner=False)
def load_hf_llm():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(HF_FALLBACK_MODEL)
    model     = AutoModelForSeq2SeqLM.from_pretrained(HF_FALLBACK_MODEL)
    return tokenizer, model


# ---------------------------------------------------------------------------
# LLM answer generation (Streaming)
# ---------------------------------------------------------------------------
def generate_answer_ollama_stream(context: str, question: str):
    try:
        import ollama
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
            stream=True
        )
        for chunk in response:
            yield chunk['message']['content']
    except Exception as exc:
        err = str(exc)
        if "connect" in err.lower() or "connection" in err.lower():
            yield (
                "⚠️ **Ollama is not running.**\n\n"
                "Start it in a separate terminal:\n"
                "```\nollama serve\n```\n"
                "Then pull the model (first time only):\n"
                "```\nollama pull llama3.2\n```\n\n"
                "Or set `USE_OLLAMA = False` in `rag_chat.py` to use the "
                "built-in HuggingFace fallback instead."
            )
        else:
            yield f"⚠️ Ollama error: {exc}"

def generate_answer_hf_stream(context: str, question: str):
    """
    Generate an answer using the HuggingFace fallback model.
    Uses the full retrieved context (all chunks) and a larger token budget.
    """
    tokenizer, model = load_hf_llm()

    # Use all retrieved chunks, not just the first one
    prompt = (
        f"Answer the question using only the information in the context below. "
        f"Be specific and concise.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    try:
        from transformers import TextIteratorStreamer
        # Allow up to 1024 input tokens to fit more context
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
        )
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=256,       # was 128 — allow longer, more complete answers
            num_beams=4,              # beam search for better quality
            length_penalty=1.2,       # encourage complete sentences
            no_repeat_ngram_size=3,
            streamer=streamer,
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

    except Exception as exc:
        yield f"⚠️ HuggingFace error: {exc}"

# ---------------------------------------------------------------------------
# Session state initialisation & Callbacks
# ---------------------------------------------------------------------------
for _key, _val in [
    ("messages",      []),
    ("total_queries", 0),
    ("avg_latency",   0.0),
    ("pending_rerun", None),   
    ("draft_input",   ""),
    ("is_generating", False),
    ("current_query", ""),
    ("interrupted",   False),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _val

def submit_callback():
    # Guard: ignore if already generating
    if st.session_state.is_generating:
        return
    val = st.session_state.draft_input.strip()
    if val:
        st.session_state.current_query = val
        st.session_state.draft_input = ""
        st.session_state.is_generating = True
        st.session_state.interrupted = False
        st.session_state.messages.append({"role": "user", "content": val})

def interrupt_callback():
    st.session_state.is_generating = False
    st.session_state.interrupted = True
    st.session_state.draft_input = st.session_state.current_query
    
    # Remove the user message since the generation was aborted
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        st.session_state.messages.pop()

# ---------------------------------------------------------------------------
# Top-level state: server + chunks
# ---------------------------------------------------------------------------
_client   = load_client()
_chunks   = load_chunks()
server_ok = _client.ping()
chunks_ok = bool(_chunks)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## VecDB RAG")
    st.markdown("*Retrieval-Augmented Generation*")
    st.divider()

    status_dot = '<span style="color:#22c55e;font-size:0.7rem">●</span>' if server_ok else '<span style="color:#ef4444;font-size:0.7rem">●</span>'
    st.markdown(f"**Server** {status_dot} `{VECDB_URL}`", unsafe_allow_html=True)
    st.markdown(f"**Chunks** &nbsp;`{len(_chunks):,}`", unsafe_allow_html=True)
    st.markdown(f"**Embedder** &nbsp;`all-MiniLM-L6-v2`", unsafe_allow_html=True)
    llm_label = f"ollama/{OLLAMA_MODEL}" if USE_OLLAMA else HF_FALLBACK_MODEL
    st.markdown(f"**LLM** &nbsp;`{llm_label}`", unsafe_allow_html=True)
    st.markdown(f"**Top-K** &nbsp;`{TOP_K}`", unsafe_allow_html=True)

    st.divider()
    st.markdown("**Session Stats**")
    col1, col2 = st.columns(2)
    col1.metric("Queries", st.session_state.total_queries)
    col2.metric("Avg Latency", f"{st.session_state.avg_latency:.0f} ms")

    st.divider()
    if st.button("", icon=":material/delete_sweep:", help="Clear chat history", use_container_width=False):
        st.session_state.messages      = []
        st.session_state.total_queries = 0
        st.session_state.avg_latency   = 0.0
        st.rerun()

    if not server_ok:
        st.error("VecDB server unreachable.\n\nStart it with:\n```\n./build-release/vecdb_server\n```")
    if not chunks_ok:
        st.warning("No chunks found.\n\nIngest documents first:\n```\npython3 ingest_docs.py docs/\n```")

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="page-header"><h1>VecDB RAG Chat</h1></div>'
    '<p class="page-subtitle">Ask anything about your ingested documents — '
    'VecDB retrieves the most semantically relevant passages and grounds the answer.</p>',
    unsafe_allow_html=True,
)
st.divider()

def render_user_msg(idx: int, content: str):
    st.markdown(f'<div class="chat-bubble-user">{content}</div>', unsafe_allow_html=True)
    cols = st.columns([8.6, 0.7, 0.7])
    with cols[1]:
        render_copy_button(content)
    with cols[2]:
        if st.button("", icon=":material/refresh:", key=f"rerun_{idx}", help="Submit this prompt again", use_container_width=True):
            st.session_state.pending_rerun = content
            st.rerun()

def render_ai_msg(content: str, sources: list = None, latency_ms: float = 0):
    st.markdown(f'<div class="chat-bubble-ai">{content}</div>', unsafe_allow_html=True)
    if sources:
        with st.expander(f"{len(sources)} sources  ·  {latency_ms:.0f} ms"):
            for i, src in enumerate(sources, start=1):
                similarity_pct = max(0.0, 1.0 - src["distance"]) * 100
                st.markdown(
                    f'<div class="source-card">'
                    f'<strong>#{i}</strong> &nbsp;<code>{src["source"]}</code> '
                    f'&nbsp;chunk&nbsp;{src["chunk_index"]} '
                    f'&nbsp;·&nbsp; {similarity_pct:.1f}% match<br><br>'
                    f'{src["text"][:400]}{"…" if len(src["text"]) > 400 else ""}'
                    f"</div>",
                    unsafe_allow_html=True,
                )

# Render conversation history
for idx, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        render_user_msg(idx, msg["content"])
    else:
        render_ai_msg(msg["content"], msg.get("sources"), msg.get("latency_ms", 0))

# ---------------------------------------------------------------------------
# Custom Chat Input (Pinned to bottom)
# ---------------------------------------------------------------------------
with st.bottom:
    cols = st.columns([9, 1])
    with cols[0]:
        st.text_input(
            "Ask a question...",
            key="draft_input",
            label_visibility="collapsed",
            placeholder="Ask a question about your documents…",
            on_change=submit_callback,
            disabled=st.session_state.is_generating or not server_ok or not chunks_ok
        )
    with cols[1]:
        if st.session_state.is_generating:
            st.button("", icon=":material/stop:", key="stop_btn", help="Stop generation",
                      on_click=interrupt_callback, use_container_width=True, type="secondary")
        else:
            st.button("", icon=":material/arrow_upward:", key="submit_btn", help="Send",
                      on_click=submit_callback, use_container_width=True,
                      disabled=not server_ok or not chunks_ok)

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
# If a rerun was requested, trigger submit
if st.session_state.pending_rerun:
    st.session_state.current_query = st.session_state.pending_rerun
    st.session_state.pending_rerun = None
    st.session_state.is_generating = True
    st.session_state.interrupted = False
    st.session_state.messages.append({"role": "user", "content": st.session_state.current_query})
    st.rerun()

if st.session_state.is_generating:
    with st.spinner("Generating response..."):
        client   = load_client()
        embedder = load_embedder()
        chunks   = load_chunks()

        t0 = time.time()
        query_vec = embedder.encode(st.session_state.current_query, normalize_embeddings=True).tolist()
        results   = client.search(query_vec, k=TOP_K)

        # Initialise so variables are always bound regardless of branch taken
        answer    = ""
        retrieved = []

        if not results:
            answer = "⚠️ No results from VecDB. Is the server running and are documents ingested?"
            retrieved = []
            st.markdown(f'<div class="chat-bubble-ai">{answer}</div>', unsafe_allow_html=True)
        else:
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
            
            # Streaming generation loop
            if USE_OLLAMA:
                stream = generate_answer_ollama_stream(context, st.session_state.current_query)
            else:
                stream = generate_answer_hf_stream(context, st.session_state.current_query)
            
            answer = ""
            placeholder = st.empty()
            
            for chunk in stream:
                answer += chunk
                placeholder.markdown(f'<div class="chat-bubble-ai">{answer}▌</div>', unsafe_allow_html=True)
                time.sleep(0.01) # Yield a tiny bit to event loop
                
            placeholder.empty()

        latency_ms = (time.time() - t0) * 1000

    # Output the answer and sources directly
    render_ai_msg(answer, retrieved, latency_ms)
    
    # Append assistant message to state
    st.session_state.messages.append({
        "role":       "assistant",
        "content":    answer,
        "sources":    retrieved,
        "latency_ms": latency_ms,
    })

    # Update stats
    n = st.session_state.total_queries
    prev_avg = st.session_state.avg_latency
    st.session_state.total_queries = n + 1
    st.session_state.avg_latency = (prev_avg * n + latency_ms) / (n + 1)
    
    # Finished generating, reset state and rerun to update UI
    st.session_state.is_generating = False
    st.rerun()
