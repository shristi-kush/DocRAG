# DocRAG Architecture

DocRAG is a voice-enabled, agentic RAG assistant that runs on local models. This
document details each pipeline. All components run locally: the LLM via Ollama,
embeddings and reranking via sentence-transformers, and speech via
faster-whisper and Piper.

## Components

| Module | Responsibility |
|--------|----------------|
| `src/config.py` | Central, env-driven configuration |
| `src/embeddings.py` | Cached `bge-small-en-v1.5` embedding provider |
| `src/ingest.py` | PDF loading + semantic/recursive chunking |
| `src/retrieval.py` | Hybrid BM25 + dense ensemble index and retrieval |
| `src/reranker.py` | Cross-encoder reranking of candidates |
| `src/rag.py` | RAG chain + routing into the agent |
| `src/metadata.py` | SQLite store of ingested-document metadata |
| `src/agent/graph.py` | LangGraph tool-calling agent |
| `src/agent/tools/*` | search_documents, calculator, sql_metadata, web_search |
| `src/voice/stt.py` | faster-whisper speech-to-text |
| `src/voice/tts.py` | Piper text-to-speech |
| `src/evaluation.py` | RAGAS evaluation harness |
| `app.py` | FastAPI service |
| `streamlit_app.py` | Streamlit UI |

## 1. Ingestion pipeline

When a PDF is uploaded, it is loaded, chunked, embedded into Chroma, indexed for
BM25, and recorded in the metadata store.

```mermaid
flowchart LR
    PDF["PDF upload"] --> Load["PyPDFLoader"]
    Load --> Decide{"total chars >= threshold\nand strategy = semantic?"}
    Decide -->|yes| Semantic["SemanticChunker\n(embedding-based)"]
    Decide -->|no| Recursive["RecursiveCharacterTextSplitter"]
    Semantic --> Chunks["chunks"]
    Recursive --> Chunks
    Chunks --> Embed["bge-small-en-v1.5"]
    Embed --> Chroma["Chroma (dense, persisted)"]
    Chunks --> BM25["BM25 index (in-memory)"]
    Load --> Meta["SQLite metadata\n(filename, pages, chunks, date)"]
```

Notes:
- Re-ingesting replaces the vector collection (single active document), while
  the metadata store accumulates a row per document ever ingested.
- Semantic chunking falls back to recursive splitting on error or for very
  short documents (`SEMANTIC_MIN_CHARS`).

## 2. Retrieval + reranking pipeline

```mermaid
flowchart LR
    Q["query"] --> BM25R["BM25Retriever (top N)"]
    Q --> DenseR["Chroma dense retriever (top N)"]
    BM25R --> Ensemble["EnsembleRetriever\n(weighted RRF)"]
    DenseR --> Ensemble
    Ensemble --> Candidates["candidate passages"]
    Candidates --> Rerank["CrossEncoder\nms-marco-MiniLM-L-6-v2"]
    Rerank --> TopK["top K passages"]
```

- First stage casts a wide net (`RETRIEVE_TOP_N`) combining sparse keyword
  matching (BM25) and dense semantic similarity (bge) via reciprocal-rank
  fusion.
- The cross-encoder rescoring stage jointly scores each `(query, passage)` pair
  and keeps the best `RETRIEVER_K`, sharply improving precision and grounding.

## 3. Agent routing pipeline

The LangGraph agent is the router: the LLM decides, via tool-calling, how to
answer. A `ToolNode` executes requested tools and loops back until the model
emits a final answer.

```mermaid
flowchart TD
    Start(["START"]) --> Agent["agent node:\nLLM.bind_tools(...)"]
    Agent --> Cond{"tool_calls present?"}
    Cond -->|yes| Tools["ToolNode"]
    Tools --> Agent
    Cond -->|no| Done(["END: final answer"])

    Tools -.-> T1["search_documents\n(retrieval + rerank)"]
    Tools -.-> T2["calculator (numexpr)"]
    Tools -.-> T3["sql_metadata_query\n(read-only SELECT)"]
    Tools -.-> T4["web_search (Tavily, optional)"]
```

- The system prompt instructs a documents-first strategy: prefer
  `search_documents`, use `calculator` for math, `sql_metadata_query` for facts
  about the ingested corpus, and `web_search` only as a fallback.
- `web_search` is registered only when `TAVILY_API_KEY` is set.
- `sql_metadata_query` accepts a single read-only `SELECT` against the
  `documents` table; writes and multi-statements are rejected.
- Set `AGENT_ENABLED=false` to bypass the agent and use the plain RAG chain.

## 4. Voice pipeline

```mermaid
flowchart LR
    Audio["audio question"] --> STT["faster-whisper\ntranscribe"]
    STT --> Text["question text"]
    Text --> Pipe["chat pipeline\n(agent or RAG)"]
    Pipe --> Answer["answer text"]
    Answer --> TTS["Piper synthesize"]
    TTS --> Wav["answer.wav"]
```

- Exposed as `POST /voice-chat` (returns transcript, answer text, and base64
  WAV) and in the Streamlit UI via `st.audio_input` (browser mic) or file
  upload, with `st.audio` playback.
- Whisper model size (`STT_MODEL_SIZE`) and Piper voice (`TTS_VOICE`) are
  configurable; voices are downloaded on first use into `data/voice/`.

## 5. Evaluation pipeline

```mermaid
flowchart LR
    QA["qa_dataset.json"] --> Loop["for each question"]
    Docs["sample PDFs"] --> Index["build hybrid index"]
    Index --> Loop
    Loop --> Ans["answer_with_sources\n(answer + contexts)"]
    Ans --> DS["RAGAS EvaluationDataset"]
    DS --> Eval["ragas.evaluate\n(local judge LLM + embeddings)"]
    Eval --> Report["docs/eval_results.md"]
```

Metrics: faithfulness, answer relevancy, context precision, context recall. The
judge is a local Ollama model (`EVAL_LLM_MODEL`) wrapped via
`LangchainLLMWrapper`; embeddings via `LangchainEmbeddingsWrapper`. A small
compatibility shim (`src/_ragas_compat.py`) lets RAGAS import cleanly on
LangChain 1.x.

## Configuration reference

All configuration is environment-driven (see `.env.example`). Key variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLAMA_MODEL` | `qwen2.5:3b` | Chat/agent model (must support tool-calling) |
| `EVAL_LLM_MODEL` | = `LLAMA_MODEL` | RAGAS judge model |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embeddings |
| `CHUNKING_STRATEGY` | `semantic` | `semantic` or `recursive` |
| `RETRIEVER_K` | `4` | Passages passed to the LLM |
| `RETRIEVE_TOP_N` | `12` | Candidate pool before reranking |
| `BM25_WEIGHT` / `DENSE_WEIGHT` | `0.5` / `0.5` | Hybrid weights |
| `RERANK_ENABLED` | `true` | Toggle cross-encoder reranking |
| `AGENT_ENABLED` | `true` | Route through the LangGraph agent |
| `TAVILY_API_KEY` | (empty) | Enables the web_search tool |
| `STT_MODEL_SIZE` | `base` | Whisper size |
| `TTS_VOICE` | `en_US-lessac-medium` | Piper voice |
| `LANGCHAIN_TRACING_V2` | `false` | Enable LangSmith tracing |
```
