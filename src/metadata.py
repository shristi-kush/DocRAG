"""Lightweight SQLite store of ingested-document metadata.

Unlike the vector index (which holds only the most recently ingested document),
this store accumulates a row per document ever ingested, giving the agent's
``sql_metadata_query`` tool something factual to query (counts, page totals,
ingest dates).
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.config import METADATA_DB

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    filename     TEXT PRIMARY KEY,
    ingest_date  TEXT NOT NULL,
    page_count   INTEGER NOT NULL,
    chunk_count  INTEGER NOT NULL
);
"""

# Only single read-only SELECT statements against the documents table are
# allowed through the agent tool.
_SELECT_RE = re.compile(r"^\s*select\b", re.IGNORECASE)
_FORBIDDEN_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|pragma|replace|"
    r"vacuum|reindex)\b",
    re.IGNORECASE,
)


def _connect() -> sqlite3.Connection:
    METADATA_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(METADATA_DB))
    conn.execute(_SCHEMA)
    return conn


def _count_pages(path: str | Path) -> int:
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(path)).pages)
    except Exception:  # noqa: BLE001
        return 0


def record_document(path: str | Path, chunk_count: int) -> None:
    """Upsert a metadata row for an ingested document."""
    filename = Path(path).name
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO documents (filename, ingest_date, page_count, chunk_count) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(filename) DO UPDATE SET "
            "ingest_date=excluded.ingest_date, "
            "page_count=excluded.page_count, "
            "chunk_count=excluded.chunk_count",
            (
                filename,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                _count_pages(path),
                int(chunk_count),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def run_select(sql: str) -> list[dict]:
    """Execute a validated read-only SELECT and return rows as dicts."""
    statement = sql.strip().rstrip(";")
    if ";" in statement:
        raise ValueError("Only a single statement is allowed")
    if not _SELECT_RE.match(statement):
        raise ValueError("Only SELECT statements are allowed")
    if _FORBIDDEN_RE.search(statement):
        raise ValueError("Statement contains a forbidden keyword")

    conn = _connect()
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(statement).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def count_documents() -> int:
    rows = run_select("SELECT COUNT(*) AS n FROM documents")
    return int(rows[0]["n"]) if rows else 0


def schema_description() -> str:
    """Human/LLM-readable description of the queryable schema."""
    return (
        "Table 'documents' columns: "
        "filename (TEXT), ingest_date (TEXT ISO-8601), "
        "page_count (INTEGER), chunk_count (INTEGER)."
    )
