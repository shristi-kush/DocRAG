# DocRAG

Private-document RAG over PDFs using LangChain, Chroma, and a local/OpenAI-compatible Llama endpoint. UI is a Detector.io-style Streamlit app; Flask exposes the same ingest/chat API for programmatic use.

## Stack

- **Python** — RAG core in `src/`
- **Streamlit** — primary UI (`streamlit_app.py`)
- **Flask** — optional HTTP API (`app.py`)
- **Chroma** + **sentence-transformers** — embeddings / retrieval
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

## Run Flask API (optional)

```bash
flask --app app run
# or
python app.py
```

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

Runs Streamlit, Flask, and Ollama together. Inside Compose, the LLM URL is `http://ollama:11434/v1` (not `localhost`).

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
| Flask API | http://localhost:5000 |
| Ollama | http://localhost:11434 |

Persisted volumes: `./data` (PDFs + Chroma), Docker volumes for Ollama models and Hugging Face embedding cache.

Stop:

```bash
docker compose down
```

## Project layout

```
app.py              Flask API
streamlit_app.py    DocRAG Streamlit UI
ingest.py           CLI ingest
Dockerfile
docker-compose.yml
src/
  config.py
  llm.py
  ingest.py
  rag.py
data/raw/           uploaded PDFs
data/chroma/        vector store
```
