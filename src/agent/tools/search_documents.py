"""Document search tool: hybrid retrieval + cross-encoder reranking."""

from __future__ import annotations

from langchain_core.tools import tool

from src import retrieval


@tool
def search_documents(query: str) -> str:
    """Search the uploaded documents for passages relevant to the query.

    This is the primary tool: prefer it for any question that could be
    answered from the user's documents. Returns the most relevant passages;
    ground your answer in them and do not invent facts.
    """
    if not retrieval.is_loaded():
        return "No document has been ingested yet, so there is nothing to search."

    docs = retrieval.retrieve(query)
    if not docs:
        return "No relevant passages found in the uploaded documents."

    blocks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "document")
        blocks.append(f"[{i}] (source: {source})\n{doc.page_content}")
    return "\n\n".join(blocks)
