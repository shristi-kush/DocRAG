# DocRAG

Private-document RAG over PDFs using LangChain, Chroma, and a local/OpenAI-compatible Llama endpoint. UI is a Detector.io-style Streamlit app; a FastAPI service exposes the same ingest/chat API for programmatic use.

## Stack

- **Python** — RAG core in `src/`
- **Streamlit** — primary UI (`streamlit_app.py`)
- **FastAPI** + **Uvicorn** — async HTTP API (`app.py`)
- **Chroma** (dense) + **BM25** (sparse) — hybrid retrieval via `EnsembleRetriever`
- **bge-small-en-v1.5** (`sentence-transformers`) — embeddings, with semantic chunking
- **Llama** via `LLAMA_BASE_URL` (OpenAI-compatible chat API)

## Setup

```bash
cd RAG_proj
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

Start Ollama first (required for answers):

```bash
ollama serve
# optional if the model is not pulled yet
ollama pull qwen2.5:3b
```

Defaults already point at Ollama (`http://localhost:11434/v1`, model `qwen2.5:3b`). Override if needed:

```bash
# Windows PowerShell
$env:LLAMA_BASE_URL = "http://localhost:11434/v1"
$env:LLAMA_MODEL = "qwen2.5:3b"

# macOS/Linux
export LLAMA_BASE_URL=http://localhost:11434/v1
export LLAMA_MODEL=qwen2.5:3b
```

Optional: `LLAMA_API_KEY` (default `not-needed`).

## Run Streamlit UI

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501 — upload a PDF, enter a question, click **Ask**.

## Run FastAPI API (optional)

```bash
uvicorn app:app --host 0.0.0.0 --port 5000
# or
python app.py
```

Interactive docs are served at http://localhost:5000/docs.

| Method | Path | Body |
|--------|------|------|
| GET | `/health` | — |
| POST | `/ingest` | multipart field `file` (PDF) |
| POST | `/chat` | JSON `{ "message": "..." }` |

## Offline ingest CLI

```bash
python ingest.py path/to/document.pdf
```

## Docker

Runs Streamlit, the FastAPI service, and Ollama together. Inside Compose, the LLM URL is `http://ollama:11434/v1` (not `localhost`).

```bash
docker compose up --build
```

First start pulls `qwen2.5:3b` via the `ollama-init` service (can take several minutes). Override the model if needed:

```bash
# Windows PowerShell
$env:LLAMA_MODEL = "qwen2.5:3b"
docker compose up --build

# macOS/Linux
LLAMA_MODEL=qwen2.5:3b docker compose up --build
```

| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| FastAPI API | http://localhost:5000 |
| Ollama | http://localhost:11434 |

Persisted volumes: `./data` (PDFs + Chroma), Docker volumes for Ollama models and Hugging Face embedding cache.

Stop:

```bash
docker compose down
```

## Project layout

```
app.py              FastAPI API
streamlit_app.py    DocRAG Streamlit UI
ingest.py           CLI ingest
Dockerfile
docker-compose.yml
src/
  config.py
  llm.py
  embeddings.py     shared bge embedding model
  ingest.py         semantic / recursive chunking
  retrieval.py      hybrid BM25 + dense ensemble
  rag.py
data/raw/           uploaded PDFs
data/chroma/        vector store
```

## Changelog

### Phase 1 — FastAPI migration, hybrid search, semantic chunking

- **API framework:** migrated the HTTP API from Flask to **FastAPI** (async, served by Uvicorn) with the same `/health`, `/ingest`, and `/chat` endpoints plus auto-generated OpenAPI docs at `/docs`.
- **Embeddings:** upgraded from `all-MiniLM-L6-v2` to **`BAAI/bge-small-en-v1.5`** (query-side instruction prefix applied). The Chroma collection was renamed (`docrag_v2`) since embedding dimensions changed.
- **Retrieval:** replaced pure dense similarity with **hybrid search** — a `EnsembleRetriever` combining `BM25Retriever` (sparse, keyword) and Chroma dense vectors (weights configurable via `BM25_WEIGHT` / `DENSE_WEIGHT`), rebuilt on every ingest.
- **Chunking:** added embedding-driven **semantic chunking** (`SemanticChunker`) with automatic fallback to `RecursiveCharacterTextSplitter` for short docs or on error (`CHUNKING_STRATEGY` env var).

**Informal before/after retrieval note.** Dense-only retrieval (old MiniLM) tended to miss queries that hinge on exact keywords, acronyms, or numbers (e.g. asking for a specific figure or a product code), because those tokens get smoothed away in a purely semantic match. Adding the BM25 arm to the ensemble surfaces those exact-term hits, while the bge dense arm still handles paraphrased/conceptual questions. Example queries where the hybrid setup helped over dense-only:

| Query type | Example | Dense-only | Hybrid (BM25 + bge) |
|------------|---------|------------|---------------------|
| Exact keyword / code | "What does clause 4.2 say?" | often missed the exact clause | retrieves the clause chunk |
| Acronym / rare term | "Define the ACME-42 metric" | drifted to related text | matches the literal term |
| Conceptual / paraphrase | "How is data kept private?" | good | still good |
