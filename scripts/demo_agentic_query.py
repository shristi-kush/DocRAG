#!/usr/bin/env python
"""Demonstrate multi-tool agent reasoning (SQL metadata + calculator).

Ingests the sample documents to populate the metadata store, then asks a
question that requires chaining two tools:

    "How many documents have been ingested, and what is 15% of that number?"

The agent must call ``sql_metadata_query`` (COUNT) and then ``calculator``.

Requires Ollama running with a tool-capable model (e.g. qwen2.5:3b).

    python scripts/demo_agentic_query.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EVAL_DIR  # noqa: E402
from src.llm import init_llm  # noqa: E402
from src.rag import process_document, process_prompt  # noqa: E402


def main() -> int:
    init_llm()

    pdfs = sorted((EVAL_DIR / "docs").glob("*.pdf"))
    if not pdfs:
        print("No sample PDFs. Run: python scripts/make_sample_docs.py")
        return 1
    for pdf in pdfs:
        result = process_document(pdf)
        print(f"Ingested {result['filename']} ({result['chunks']} chunks)")

    question = "How many documents have been ingested, and what is 15% of that number?"
    print(f"\nQ: {question}")
    answer = process_prompt(question)
    print(f"A: {answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
