from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKING_STRATEGY,
    SEMANTIC_MIN_CHARS,
)

logger = logging.getLogger(__name__)


def _recursive_split(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def _semantic_split(docs: list) -> list:
    """Split using embedding-based semantic boundaries.

    Falls back to recursive splitting if the experimental splitter is
    unavailable or fails at runtime.
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker

        from src.embeddings import get_embeddings

        chunker = SemanticChunker(
            embeddings=get_embeddings(),
            breakpoint_threshold_type="percentile",
        )
        return chunker.split_documents(docs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Semantic chunking failed (%s); using recursive splitter.", exc)
        return _recursive_split(docs)


def load_and_split_pdf(path: str | Path) -> list:
    """Load a PDF and split it into chunks using the configured strategy."""
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported")

    docs = PyPDFLoader(str(pdf_path)).load()
    total_chars = sum(len(d.page_content) for d in docs)

    use_semantic = (
        CHUNKING_STRATEGY.lower() == "semantic" and total_chars >= SEMANTIC_MIN_CHARS
    )
    if use_semantic:
        return _semantic_split(docs)
    return _recursive_split(docs)
