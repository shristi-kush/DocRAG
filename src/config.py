import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
CHROMA_DIR = BASE_DIR / "data" / "chroma"

# Defaults target Ollama's OpenAI-compatible API (ollama serve → :11434).
LLAMA_BASE_URL = os.getenv("LLAMA_BASE_URL", "http://localhost:11434/v1")
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY", "not-needed")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "qwen2.5:3b")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "4"))

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
