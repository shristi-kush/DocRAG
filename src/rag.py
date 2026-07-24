from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src import retrieval
from src.config import RETRIEVER_K
from src.ingest import load_and_split_pdf
from src.llm import get_llm

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions using only the "
            "provided document context. If the answer is not in the context, say "
            "you do not know based on the uploaded document.\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


class DocumentNotLoadedError(RuntimeError):
    """Raised when chat is requested before any document has been ingested."""


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def process_document(path: str | Path) -> dict:
    """Ingest a PDF into the hybrid index (replaces prior collection)."""
    chunks = load_and_split_pdf(path)
    if not chunks:
        raise ValueError("No text could be extracted from the PDF")

    retrieval.build_index(chunks)

    filename = Path(path).name
    try:
        from src.metadata import record_document

        record_document(path, chunk_count=len(chunks))
    except Exception:  # noqa: BLE001 - metadata store is optional (Phase 4)
        pass

    return {"ok": True, "chunks": len(chunks), "filename": filename}


def _handle_llm_error(exc: Exception) -> None:
    err = str(exc).lower()
    if "connection" in err or "connect" in err or "refused" in err:
        from src.config import LLAMA_BASE_URL, LLAMA_MODEL

        raise ConnectionError(
            f"Cannot reach LLM at {LLAMA_BASE_URL} (model={LLAMA_MODEL}). "
            "Start Ollama (`ollama serve`) and ensure the model is pulled "
            f"(e.g. `ollama pull {LLAMA_MODEL}`)."
        ) from exc
    raise exc


def get_context(message: str) -> list[Document]:
    """Retrieve the grounding chunks for a message (shared with the agent)."""
    return retrieval.retrieve(message.strip(), top_k=RETRIEVER_K)


def answer_with_sources(message: str) -> tuple[str, list[Document]]:
    """Generate a grounded answer and return it with the contexts used.

    Retrieving once and generating from the same contexts keeps the answer and
    its supporting passages consistent (important for faithful evaluation).
    """
    if not message or not message.strip():
        raise ValueError("Message must not be empty")
    if not retrieval.is_loaded():
        raise DocumentNotLoadedError(
            "No document loaded. Upload and ingest a PDF first."
        )

    docs = get_context(message)
    chain = _PROMPT | get_llm()
    try:
        result = chain.invoke(
            {"context": _format_docs(docs), "question": message.strip()}
        )
    except Exception as exc:  # noqa: BLE001
        _handle_llm_error(exc)
    return getattr(result, "content", str(result)), docs


def process_prompt(message: str) -> str:
    """Answer a question.

    Routes through the LangGraph agent (tool-calling over document search,
    calculator, SQL metadata, and optional web search) when the agent is
    enabled; otherwise falls back to the plain RAG chain.
    """
    if not message or not message.strip():
        raise ValueError("Message must not be empty")

    from src.config import AGENT_ENABLED

    if AGENT_ENABLED:
        from src.agent import graph

        try:
            return graph.answer(message)
        except Exception as exc:  # noqa: BLE001
            _handle_llm_error(exc)

    answer, _ = answer_with_sources(message)
    return answer


def is_document_loaded() -> bool:
    return retrieval.is_loaded()
