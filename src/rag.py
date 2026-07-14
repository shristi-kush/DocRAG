from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import CHROMA_DIR, RETRIEVER_K
from src.ingest import load_and_split_pdf
from src.llm import get_llm

_vectorstore: Chroma | None = None
_embeddings: HuggingFaceEmbeddings | None = None
_document_loaded = False

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


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings


def _format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def process_document(path: str | Path) -> dict:
    """Ingest a PDF into the Chroma vector store (replaces prior collection)."""
    global _vectorstore, _document_loaded

    chunks = load_and_split_pdf(path)
    if not chunks:
        raise ValueError("No text could be extracted from the PDF")

    embeddings = _get_embeddings()

    # Drop prior collection so re-uploads fully replace indexed content.
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
        collection_name="docrag",
    )
    _document_loaded = True
    return {"ok": True, "chunks": len(chunks), "filename": Path(path).name}


def process_prompt(message: str) -> str:
    """Retrieve relevant chunks and generate an answer with the private Llama."""
    if not message or not message.strip():
        raise ValueError("Message must not be empty")
    if not _document_loaded or _vectorstore is None:
        raise DocumentNotLoadedError(
            "No document loaded. Upload and ingest a PDF first."
        )

    retriever = _vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | _PROMPT
        | get_llm()
    )
    try:
        result = chain.invoke(message.strip())
    except Exception as exc:  # noqa: BLE001
        err = str(exc).lower()
        if "connection" in err or "connect" in err or "refused" in err:
            from src.config import LLAMA_BASE_URL, LLAMA_MODEL

            raise ConnectionError(
                f"Cannot reach LLM at {LLAMA_BASE_URL} (model={LLAMA_MODEL}). "
                "Start Ollama (`ollama serve`) and ensure the model is pulled "
                f"(e.g. `ollama pull {LLAMA_MODEL}`)."
            ) from exc
        raise
    return getattr(result, "content", str(result))


def is_document_loaded() -> bool:
    return _document_loaded
