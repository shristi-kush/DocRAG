from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from werkzeug.utils import secure_filename

from src.config import DATA_RAW_DIR
from src.llm import init_llm
from src.rag import DocumentNotLoadedError, process_document, process_prompt

app = FastAPI(title="DocRAG API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    init_llm()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str


class IngestResponse(BaseModel):
    ok: bool
    filename: str
    chunks: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    filename = secure_filename(file.filename)
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_RAW_DIR / filename
    dest.write_bytes(await file.read())

    try:
        result = process_document(dest)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}") from exc

    return IngestResponse(
        ok=True, filename=result["filename"], chunks=result["chunks"]
    )


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    if not payload.message or not payload.message.strip():
        raise HTTPException(
            status_code=400, detail="Body must include a non-empty 'message'"
        )

    try:
        answer = process_prompt(payload.message)
    except DocumentNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc

    return ChatResponse(answer=answer)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "5000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
