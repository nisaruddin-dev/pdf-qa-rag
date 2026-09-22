# 📄 Ask My PDF — RAG-powered Document Q&A

A Streamlit app that lets you upload a PDF and ask questions about it. Answers are grounded strictly in the document and include page citations. If the answer isn't in the PDF, it says so — no hallucinations.

Built as a clean **RAG (Retrieval-Augmented Generation)** pipeline: extract → chunk → embed → retrieve → answer.

---

## ✨ What It Does

1. Upload any text-based PDF
2. The app extracts the text, splits it into overlapping chunks, and builds a searchable vector index
3. Ask a question in plain English
4. The app retrieves the top-3 most relevant passages and sends them to an LLM
5. You get a concise answer with the source page number(s) cited

If the answer isn't in the retrieved context, the app responds: *"I don't know based on the provided document."*

---

## 🔄 How It Works

```
PDF upload
     │
     ▼
┌─────────────────────────────────┐
│ 1. EXTRACT                      │
│    pdfplumber → per-page text   │
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│ 2. CHUNK                        │
│    500-char windows,            │
│    50-char overlap              │
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│ 3. EMBED                        │
│    sentence-transformers        │
│    all-MiniLM-L6-v2 (~90 MB)    │
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│ 4. INDEX                        │
│    FAISS IndexFlatL2            │
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│ 5. RETRIEVE + ANSWER            │
│    top-3 chunks + question →    │
│    Groq (openai/gpt-oss-120b)   │
└─────────────────────────────────┘
     │
     ▼
Answer with page citations
```

Each retrieved chunk carries its source page number, so every answer can be traced back to a specific page in the PDF.

---

## 🏗️ Design Choices

| Decision | Reason |
|---|---|
| **pdfplumber** for extraction | Per-page text with layout awareness — needed for accurate page citations |
| **500-char chunks, 50-char overlap** | Small enough for precise retrieval, large enough to preserve context across sentence boundaries |
| **`all-MiniLM-L6-v2`** | Small (~90 MB), fast on CPU, well-suited for semantic search on short passages |
| **FAISS `IndexFlatL2`** | Exact nearest-neighbor search; trivial to set up and fast for single-document scale |
| **Top-3 chunks** | Balances context completeness against prompt size and API cost |
| **Strict "answer only from context" prompt** | Prevents hallucination on out-of-document questions |
| **Page numbers attached to chunks** | Enables verifiable citations in every answer |
| **`st.secrets` for the API key** | No hard-coded credentials anywhere in the code |
| **`try/except` around the LLM call** | API errors surface as readable messages instead of crashing the app |
| **`@st.cache_resource` on the embedder** | Model loads once per session, not on every rerun |

---

## 🚀 Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/pdf-qa-rag.git
cd pdf-qa-rag
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

⚠️ First install is slow — `sentence-transformers` pulls in `torch` (~800 MB) and `faiss-cpu` (~30 MB). Budget 5–15 minutes depending on your connection.

### 4. Add your Groq API key

Get a free key at [console.groq.com/keys](https://console.groq.com/keys).

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "gsk_your_actual_key_here"
```

### 5. Run

```bash
python -m streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 🌐 Live Demo

**Try it now:** [pdf-rag-app-io.streamlit.app](https://pdf-rag-app-io.streamlit.app/)

Upload a PDF, ask a question, get a grounded answer with page citations.

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (already done if you're reading this on GitHub).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click **New app**, select the repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and paste:
   ```toml
   GROQ_API_KEY = "gsk_your_actual_key_here"
   ```
5. Click **Deploy**.

First build takes 3–5 minutes because of the heavy dependencies. Subsequent rebuilds are cached and fast.

---

## 📁 Project Structure

```
pdf-qa-rag/
├── .streamlit/
│   └── secrets.toml          # Local only — NOT committed
├── .gitignore
├── app.py                    # UI + RAG pipeline
├── requirements.txt
└── README.md
```

---

## 📦 Requirements

```
streamlit
groq
pdfplumber
sentence-transformers
faiss-cpu
numpy
```

---

## 🧪 Testing It

Upload a small PDF (2–5 pages) with clear factual content — a syllabus, a policy document, a product manual.

**Test 1 — answer is in the document**
Ask a factual question whose answer you can verify. Expected: a correct answer ending with `(Source: page N)`.

**Test 2 — answer is NOT in the document**
Ask something the PDF doesn't cover. Expected: the exact reply `I don't know based on the provided document.`

If Test 2 ever hallucinates, tighten the prompt or lower `temperature` to `0.0` in `answer_question`.

---

## ⚡ Performance Notes

- **First query is slow.** The embedding model (~90 MB) downloads on first use. Subsequent runs load from cache in seconds.
- **The model is cached in the session.** `@st.cache_resource` means it loads once per user session, not per question.
- **Each question costs one Groq call.** Prompt size is roughly question + 3 chunks (~1,500 tokens). Well within Groq's free tier.
- **Streamlit Cloud cold start.** Free-tier apps sleep after inactivity. The first wake-up takes ~30 seconds; after that it's responsive.

---

## 🔒 Security

The Groq API key is **only** read from `st.secrets["GROQ_API_KEY"]`. There is no hard-coded fallback. If the secret is missing, the app surfaces a clear error on the first LLM call rather than silently failing.

`.streamlit/secrets.toml` is in `.gitignore` and is never committed. On Streamlit Cloud, the key is provided via the app's Secrets panel in the dashboard.

---

## 🗺️ Ideas for Extension

- Multi-PDF support (upload a folder, search across all documents)
- Highlight the exact passage in the source PDF when citing
- Streaming answers token-by-token for perceived speed
- Support for scanned PDFs (OCR via `pytesseract`)
- Chat history persistence across sessions
- Export Q&A transcript as Markdown

---

## 📜 License

MIT — free to use, modify, and share.
