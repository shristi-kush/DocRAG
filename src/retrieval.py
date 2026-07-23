"""Hybrid retrieval: BM25 (sparse) + Chroma dense vectors via an ensemble.

Owns the in-memory index state (vector store + BM25 retriever) so that both
the RAG chain and the agent's ``search_documents`` tool share one index.
"""

from __future__ import annotations

import logging

try:  # LangChain 1.x moved EnsembleRetriever into langchain_classic
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:  # LangChain 0.x
    from langchain.retrievers import EnsembleRetriever

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.config import (
    BM25_WEIGHT,
    CHROMA_DIR,
    COLLECTION_NAME,
    DENSE_WEIGHT,
    RETRIEVE_TOP_N,
    RETRIEVER_K,
)
from src.embeddings import get_embeddings

logger = logging.getLogger(__name__)

_vectorstore: Chroma | None = None
_bm25: BM25Retriever | None = None
_documents_loaded = False


def build_index(chunks: list[Document]) -> None:
    """(Re)build the dense + sparse indexes from a fresh set of chunks.

    Any previously indexed collection is dropped so re-uploads fully replace
    the searchable content.
    """
    global _vectorstore, _bm25, _documents_loaded

    if not chunks:
        raise ValueError("No chunks to index")

    embeddings = get_embeddings()

    if _vectorstore is not None:
        try:
            _vectorstore.delete_collection()
        except Exception:  # noqa: BLE001
            pass
        _vectorstore = None

    _vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )

    _bm25 = BM25Retriever.from_documents(chunks)
    _bm25.k = RETRIEVE_TOP_N
    _documents_loaded = True
    logger.info("Built hybrid index over %d chunks.", len(chunks))


def is_loaded() -> bool:
    return _documents_loaded


def get_hybrid_retriever() -> EnsembleRetriever:
    if _vectorstore is None or _bm25 is None:
        raise RuntimeError("Index not built. Call build_index() first.")
    dense = _vectorstore.as_retriever(search_kwargs={"k": RETRIEVE_TOP_N})
    return EnsembleRetriever(
        retrievers=[_bm25, dense],
        weights=[BM25_WEIGHT, DENSE_WEIGHT],
    )


def retrieve(query: str, top_k: int | None = None) -> list[Document]:
    """Return the most relevant chunks for a query via hybrid search.

    A cross-encoder reranking stage is applied when enabled (Phase 2).
    """
    k = top_k or RETRIEVER_K
    candidates = get_hybrid_retriever().invoke(query)

    try:
        from src.reranker import rerank  # local import: optional (Phase 2)

        return rerank(query, candidates, top_k=k)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Reranking unavailable (%s); returning hybrid order.", exc)
        return candidates[:k]
