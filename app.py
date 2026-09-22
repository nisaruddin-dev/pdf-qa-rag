"""
RAG-based PDF Q&A App.
Upload a PDF, ask questions, get answers grounded in the document with page citations.

Stack: pdfplumber (extract) -> sentence-transformers (embed) -> FAISS (search) -> Cerebras (answer)
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
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 2

st.set_page_config(page_title="PDF Q&A (RAG)", page_icon="📄", layout="wide")


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
    Extract text from each page of the PDF.
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

def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Sliding window chunker over each page.
    Each chunk carries its source page number.
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
                chunks.append({"text": chunk_text, "page": page_num})
            if end >= len(text):
                break
            start = end - CHUNK_OVERLAP
    return chunks


# =============================================================================
# EMBEDDINGS + FAISS
# =============================================================================

def build_index(chunks: list[dict], embedder: SentenceTransformer):
    """
    Embed all chunks and build a FAISS index.
    Returns (index, embeddings_array).
    """
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings


def retrieve(query: str, chunks: list[dict], index, embedder: SentenceTransformer, k: int = TOP_K):
    """Return the top-k chunks most similar to the query."""
    q_vec = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False).astype("float32")
    distances, indices = index.search(q_vec, k)
    results = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])
    return results


# =============================================================================
# CEREBRAS ANSWERING
# =============================================================================

def build_prompt(question: str, contexts: list[dict]) -> str:
    context_block = "\n\n".join(
        f"[Page {c['page']}]\n{c['text']}" for c in contexts
    )
    return f"""You are a document Q&A assistant. Answer the user's question using ONLY the context below.

Rules:
1. Use only the provided context. Do NOT use outside knowledge.
2. If the answer is not in the context, reply exactly: "I don't know based on the provided document."
3. Always cite the page number(s) you used, in the format: (Source: page X)
4. Be concise. Quote the document where helpful.

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
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Groq error] {type(e).__name__}: {e}"


# =============================================================================
# PIPELINE (stateful across reruns)
# =============================================================================

def process_pdf(pdf_bytes: bytes, filename: str):
    """Extract, chunk, embed, and index the PDF. Stores everything in session_state."""
    with st.spinner("Extracting text from PDF..."):
        pages = extract_pages(pdf_bytes)

    if not pages:
        st.error("No readable text found in this PDF. It may be scanned images only.")
        return

    with st.spinner("Chunking text..."):
        chunks = chunk_pages(pages)

    with st.spinner(f"Loading embedding model ({EMBED_MODEL_NAME}) — first run may take 30–60s..."):
        embedder = load_embedder()

    with st.spinner(f"Embedding {len(chunks)} chunks and building FAISS index..."):
        index, _ = build_index(chunks, embedder)

    st.session_state["pages"] = pages
    st.session_state["chunks"] = chunks
    st.session_state["index"] = index
    st.session_state["embedder"] = embedder
    st.session_state["filename"] = filename
    st.session_state["chat_history"] = []


# =============================================================================
# UI
# =============================================================================

st.title("📄 PDF Q&A (RAG)")
st.caption("Upload a PDF, ask questions, get grounded answers with page citations.")

with st.sidebar:
    st.header("1. Upload")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded is not None:
        if st.session_state.get("filename") != uploaded.name:
            process_pdf(uploaded.read(), uploaded.name)

    if "chunks" in st.session_state:
        st.success(
            f"Indexed **{len(st.session_state['pages'])}** pages → "
            f"**{len(st.session_state['chunks'])}** chunks."
        )
        st.caption(f"File: `{st.session_state['filename']}`")

        if st.button("Clear / New PDF"):
            for key in ("pages", "chunks", "index", "embedder", "filename", "chat_history"):
                st.session_state.pop(key, None)
            st.rerun()


if "index" not in st.session_state:
    st.info("👈 Upload a PDF in the sidebar to begin.")
    st.stop()


# ---- Q&A ----
st.subheader("2. Ask a question")

question = st.text_input(
    "Your question",
    placeholder="e.g., How many weeks is this course?",
    key="question_input",
)

ask = st.button("🔎 Ask", type="primary")

if ask and question.strip():
    with st.spinner("Retrieving relevant passages..."):
        contexts = retrieve(
            question,
            st.session_state["chunks"],
            st.session_state["index"],
            st.session_state["embedder"],
        )

    with st.spinner("Asking Cerebras..."):
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
            with st.expander("📎 Retrieved context (top-3 chunks)"):
                for c in turn["contexts"]:
                    st.markdown(f"**Page {c['page']}**")
                    st.caption(c["text"][:400] + ("..." if len(c["text"]) > 400 else ""))