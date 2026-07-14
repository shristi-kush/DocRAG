from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.config import LLAMA_API_KEY, LLAMA_BASE_URL, LLAMA_MODEL

_llm: ChatOpenAI | None = None


def init_llm() -> ChatOpenAI:
    """Create and cache the private Llama chat client (OpenAI-compatible API)."""
    global _llm
    _llm = ChatOpenAI(
        base_url=LLAMA_BASE_URL,
        api_key=LLAMA_API_KEY,
        model=LLAMA_MODEL,
        temperature=0.1,
    )
    return _llm


def get_llm() -> ChatOpenAI:
    if _llm is None:
        raise RuntimeError("LLM not initialized. Call init_llm() first.")
    return _llm
