#!/usr/bin/env python
"""Convenience entry point for the RAGAS evaluation harness.

Equivalent to ``python -m src.evaluation``. Requires Ollama to be running with
the chat/eval model pulled, plus the sample PDFs generated via
``python scripts/make_sample_docs.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
