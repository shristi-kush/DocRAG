"""Cross-encoder reranking stage.

Takes the candidate pool from hybrid retrieval and reorders it with a
cross-encoder that scores each (query, passage) pair jointly - far more precise
than the first-stage bi-encoder similarity, which improves grounding and
reduces hallucination.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document

from src.config import RERANK_ENABLED, RERANKER_MODEL, RETRIEVER_K

logger = logging.getLogger(__name__)

_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(RERANKER_MODEL)
    return _cross_encoder


def rerank(
    query: str, documents: list[Document], top_k: int | None = None
) -> list[Document]:
    """Return the ``top_k`` documents most relevant to ``query``.

    If reranking is disabled or there is nothing to score, the original order
    is preserved and simply truncated.
    """
    k = top_k or RETRIEVER_K
    if not RERANK_ENABLED or not documents:
        return documents[:k]

    model = _get_cross_encoder()
    pairs = [(query, doc.page_content) for doc in documents]
    scores = model.predict(pairs)

    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:k]]
