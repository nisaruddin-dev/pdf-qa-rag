"""
RAG-based PDF Q&A App (Multi-Document).
Upload one or more PDFs, ask questions, get answers grounded in the
documents with page citations.

Stack: pdfplumber (extract) -> sentence-transformers (embed) -> FAISS (search) -> Groq (answer)
"""

import io

import faiss
import numpy as np
import pdfplumber
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# =============================================================================
# CONFIG
# =============================================================================

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"
CHUNK_SIZE = 800          # chars per chunk
CHUNK_OVERLAP = 100       # chars of overlap between consecutive chunks
TOP_K_PER_DOC = 3         # top-K chunks per document (so cross-doc queries work)
MAX_DOCS = 5              # safety cap on number of uploads

st.set_page_config(page_title="Multi-PDF Q&A (RAG)", page_icon="📄", layout="wide")


# =============================================================================
# CACHED RESOURCES
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_embedder() -> SentenceTransformer:
    """Load the embedding model once and cache it across reruns."""
    return SentenceTransformer(EMBED_MODEL_NAME)


def get_groq_client() -> Groq:
    """Groq client — key read from st.secrets, never hard-coded."""
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


# =============================================================================
# PDF EXTRACTION
# =============================================================================

def extract_pages(pdf_bytes: bytes) -> list[dict]:
    """
    Extract text from each page of a single PDF.
    Returns a list of {"page": <1-based number>, "text": <page text>}.
    """
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page": i, "text": text})
    return pages


# =============================================================================
# CHUNKING
# =============================================================================

def chunk_pages(pages: list[dict], doc_name: str) -> list[dict]:
    """
    Sliding window chunker over each page.
    Each chunk carries its source doc name + page number.
    """
    chunks = []
    for p in pages:
        text = p["text"]
        page_num = p["page"]
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page": page_num,
                    "doc": doc_name,
                })
            if end >= len(text):
                break
            start = end - CHUNK_OVERLAP
    return chunks


# =============================================================================
# EMBEDDINGS + FAISS  (one index per document)
# =============================================================================

def build_index_for_chunks(chunks: list[dict], embedder: SentenceTransformer):
    """Embed a list of chunks and build a FAISS index for that document."""
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def retrieve(
    query: str,
    doc_store: dict,           # {doc_name: {"chunks": [...], "index": <faiss>}}
    embedder: SentenceTransformer,
    k_per_doc: int = TOP_K_PER_DOC,
) -> list[dict]:
    """
    Retrieve top-k chunks from EACH document, then return all combined.
    This guarantees multi-document questions get context from every doc.
    """
    q_vec = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False).astype("float32")

    results = []
    for doc_name, entry in doc_store.items():
        chunks = entry["chunks"]
        index = entry["index"]
        if not chunks:
            continue
        distances, indices = index.search(q_vec, min(k_per_doc, len(chunks)))
        for idx in indices[0]:
            if 0 <= idx < len(chunks):
                results.append(chunks[idx])
    return results


# =============================================================================
# GROQ ANSWERING
# =============================================================================

def build_prompt(question: str, contexts: list[dict]) -> str:
    context_block = "\n\n".join(
        f"[{c['doc']} — Page {c['page']}]\n{c['text']}" for c in contexts
    )
    return f"""You are a document Q&A assistant. Answer the user's question using ONLY the context below.

Rules:
1. Use only the provided context. Do NOT use outside knowledge.
2. If the answer is not in the context, reply exactly: "I don't know based on the provided document."
3. Always cite the source document name and page number(s) you used, in the format: (Source: <doc name>, page X)
4. If the answer requires combining information from multiple documents, cite each one.
5. Be concise. Quote the document where helpful.

Context:
{context_block}

Question: {question}

Answer:"""


def answer_question(question: str, contexts: list[dict]) -> str:
    """Call Groq with the question + retrieved context. Returns the answer string."""
    client = get_groq_client()
    prompt = build_prompt(question, contexts)

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a careful document assistant that never invents information."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Groq error] {type(e).__name__}: {e}"


# =============================================================================
# PIPELINE  (stateful across reruns)
# =============================================================================

def process_uploads(uploaded_files, embedder: SentenceTransformer):
    """
    Extract, chunk, embed, and index each uploaded PDF.
    Stores a per-document store in session_state.
    """
    doc_store: dict = {}

    progress = st.progress(0, text="Processing documents...")
    total = len(uploaded_files)

    for i, uploaded in enumerate(uploaded_files):
        progress.progress(
            (i + 0.1) / total,
            text=f"Processing {uploaded.name} ({i+1}/{total}) — extracting text...",
        )

        try:
            pages = extract_pages(uploaded.read())
        except Exception as e:
            st.warning(f"Could not read `{uploaded.name}`: {e}")
            continue

        if not pages:
            st.warning(f"`{uploaded.name}` had no readable text — skipped.")
            continue

        chunks = chunk_pages(pages, uploaded.name)

        progress.progress(
            (i + 0.6) / total,
            text=f"Embedding {len(chunks)} chunks from {uploaded.name}...",
        )

        index = build_index_for_chunks(chunks, embedder)

        doc_store[uploaded.name] = {
            "chunks": chunks,
            "index": index,
            "pages": len(pages),
        }

        progress.progress((i + 1) / total, text=f"Indexed {uploaded.name}.")

    progress.empty()
    st.session_state["doc_store"] = doc_store
    st.session_state["embedder"] = embedder
    st.session_state["chat_history"] = []


# =============================================================================
# UI
# =============================================================================

st.title("📄 Multi-PDF Q&A (RAG)")
st.caption("Upload one or more PDFs, ask questions, get grounded answers with doc + page citations.")

with st.sidebar:
    st.header("1. Upload PDFs")

    uploaded_files = st.file_uploader(
        "Choose one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if len(uploaded_files) > MAX_DOCS:
            st.warning(f"Only the first {MAX_DOCS} files will be processed.")
            uploaded_files = uploaded_files[:MAX_DOCS]

        current_names = sorted([f.name for f in uploaded_files])
        indexed_names = sorted(st.session_state.get("doc_store", {}).keys())

        if current_names != indexed_names:
            embedder = load_embedder()
            process_uploads(uploaded_files, embedder)

    if "doc_store" in st.session_state and st.session_state["doc_store"]:
        st.success(f"Indexed **{len(st.session_state['doc_store'])}** document(s).")
        for name, entry in st.session_state["doc_store"].items():
            st.caption(f"• `{name}` — {entry['pages']} pages, {len(entry['chunks'])} chunks")

        if st.button("Clear all / New uploads"):
            for key in ("doc_store", "chat_history"):
                st.session_state.pop(key, None)
            st.rerun()


if "doc_store" not in st.session_state or not st.session_state["doc_store"]:
    st.info("👈 Upload one or more PDFs in the sidebar to begin.")
    st.stop()


# ---- Q&A ----
st.subheader("2. Ask a question")

question = st.text_input(
    "Your question",
    placeholder="e.g., What is the refund policy? Or: compare the leave policy across both documents.",
    key="question_input",
)

ask = st.button("🔎 Ask", type="primary")

if ask and question.strip():
    with st.spinner("Retrieving relevant passages from all documents..."):
        contexts = retrieve(
            question,
            st.session_state["doc_store"],
            st.session_state["embedder"],
        )

    with st.spinner("Asking Groq..."):
        answer = answer_question(question, contexts)

    st.session_state.setdefault("chat_history", []).append(
        {"q": question, "a": answer, "contexts": contexts}
    )


# ---- Chat history ----
if st.session_state.get("chat_history"):
    for turn in reversed(st.session_state["chat_history"]):
        with st.container(border=True):
            st.markdown(f"**Q:** {turn['q']}")
            st.markdown(f"**A:** {turn['a']}")
            with st.expander(f"📎 Retrieved context ({len(turn['contexts'])} chunks)"):
                for c in turn["contexts"]:
                    st.markdown(f"**{c['doc']} — Page {c['page']}**")
                    st.caption(c["text"][:400] + ("..." if len(c["text"]) > 400 else ""))