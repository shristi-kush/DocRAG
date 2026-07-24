import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "chroma"
EVAL_DIR = DATA_DIR / "eval"
VOICE_DIR = DATA_DIR / "voice"
METADATA_DB = DATA_DIR / "metadata.db"

# --- LLM (Ollama OpenAI-compatible API; `ollama serve` -> :11434) ------------
LLAMA_BASE_URL = os.getenv("LLAMA_BASE_URL", "http://localhost:11434/v1")
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY", "not-needed")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "qwen2.5:3b")
# Model used only for RAGAS evaluation (bigger = more reliable judging).
EVAL_LLM_MODEL = os.getenv("EVAL_LLM_MODEL", LLAMA_MODEL)

# --- Embeddings --------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
# BGE models expect a query-side instruction prefix (documents get none).
BGE_QUERY_INSTRUCTION = os.getenv(
    "BGE_QUERY_INSTRUCTION",
    "Represent this sentence for searching relevant passages: ",
)

# --- Chunking ----------------------------------------------------------------
# "semantic" (default) or "recursive"
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "semantic")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
# Below this many characters, semantic chunking is skipped (falls back).
SEMANTIC_MIN_CHARS = int(os.getenv("SEMANTIC_MIN_CHARS", "2000"))

# --- Retrieval ---------------------------------------------------------------
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "docrag_v2")
# Final number of chunks handed to the LLM.
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "4"))
# Candidate pool size fetched from each retriever before reranking.
RETRIEVE_TOP_N = int(os.getenv("RETRIEVE_TOP_N", "12"))
# Ensemble (hybrid search) weights: [BM25 sparse, dense vector].
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.5"))
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.5"))

# --- Reranking (Phase 2) -----------------------------------------------------
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() in ("1", "true", "yes")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Voice (Phase 3) ---------------------------------------------------------
STT_MODEL_SIZE = os.getenv("STT_MODEL_SIZE", "base")
STT_COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "int8")
STT_DEVICE = os.getenv("STT_DEVICE", "cpu")
TTS_VOICE = os.getenv("TTS_VOICE", "en_US-lessac-medium")
# Directory that holds the downloaded Piper .onnx voice files.
TTS_VOICE_DIR = Path(os.getenv("TTS_VOICE_DIR", str(VOICE_DIR / "piper")))

# --- Agent (Phase 4) ---------------------------------------------------------
# When enabled, /chat and the UI route through the LangGraph tool-calling agent.
AGENT_ENABLED = os.getenv("AGENT_ENABLED", "true").lower() in ("1", "true", "yes")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "3"))

# --- LangSmith tracing (Phase 2) ---------------------------------------------
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() in (
    "1",
    "true",
    "yes",
)
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "docrag")


def warn_if_tracing_misconfigured() -> None:
    """Emit a non-fatal warning if LangSmith tracing lacks an API key."""
    if LANGCHAIN_TRACING_V2 and not LANGCHAIN_API_KEY:
        import warnings

        warnings.warn(
            "LANGCHAIN_TRACING_V2 is enabled but LANGCHAIN_API_KEY is not set; "
            "LangSmith traces will not be recorded.",
            stacklevel=2,
        )


for _d in (DATA_RAW_DIR, CHROMA_DIR, EVAL_DIR, VOICE_DIR, TTS_VOICE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
