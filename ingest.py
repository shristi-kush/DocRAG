#!/usr/bin/env python
"""Offline PDF ingest CLI: python ingest.py path/to/file.pdf"""

from __future__ import annotations

import sys
from pathlib import Path

from src.llm import init_llm
from src.rag import process_document


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python ingest.py <path-to-pdf>", file=sys.stderr)
        return 1

    pdf_path = Path(sys.argv[1])
    if not pdf_path.is_file():
        print(f"File not found: {pdf_path}", file=sys.stderr)
        return 1

    init_llm()
    result = process_document(pdf_path)
    print(f"Ingested {result['filename']} ({result['chunks']} chunks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
