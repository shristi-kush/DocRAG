"""RAGAS evaluation harness for the DocRAG retrieval + generation pipeline.

Runs a self-written Q&A set through the live pipeline and scores it with RAGAS
using a fully local judge (the Ollama model + the HuggingFace embeddings), so
no data leaves the machine. Results quantify faithfulness (hallucination),
context precision, and answer relevancy - the "measured hallucination
reduction" story from the project brief.

Usage:
    python -m src.evaluation          # or: python scripts/run_eval.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_openai import ChatOpenAI

from src import _ragas_compat
from src.config import (
    EVAL_DIR,
    EVAL_LLM_MODEL,
    LLAMA_API_KEY,
    LLAMA_BASE_URL,
)
from src.embeddings import get_embeddings
from src.ingest import load_and_split_pdf

logger = logging.getLogger(__name__)

DOCS_DIR = EVAL_DIR / "docs"
QA_PATH = EVAL_DIR / "qa_dataset.json"
RESULTS_MD = Path(__file__).resolve().parent.parent / "docs" / "eval_results.md"


def _load_qa() -> list[dict]:
    if not QA_PATH.is_file():
        raise FileNotFoundError(f"Q&A dataset not found: {QA_PATH}")
    return json.loads(QA_PATH.read_text(encoding="utf-8"))


def _build_eval_index() -> int:
    """Index every sample PDF together so questions can span documents."""
    from src import retrieval

    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No sample PDFs in {DOCS_DIR}. Run: python scripts/make_sample_docs.py"
        )
    all_chunks: list = []
    for pdf in pdfs:
        all_chunks.extend(load_and_split_pdf(pdf))
    retrieval.build_index(all_chunks)
    logger.info("Indexed %d chunks from %d sample PDFs.", len(all_chunks), len(pdfs))
    return len(pdfs)


def _eval_llm() -> ChatOpenAI:
    """LLM used as the RAGAS judge (configurable, defaults to the chat model)."""
    return ChatOpenAI(
        base_url=LLAMA_BASE_URL,
        api_key=LLAMA_API_KEY,
        model=EVAL_LLM_MODEL,
        temperature=0.0,
    )


def run(limit: int | None = None) -> dict:
    """Execute the evaluation and return the aggregated metric scores."""
    _ragas_compat.install()

    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    from src.llm import init_llm
    from src.rag import answer_with_sources

    init_llm()
    _build_eval_index()

    qa = _load_qa()
    if limit:
        qa = qa[:limit]

    samples = []
    for item in qa:
        question = item["question"]
        answer, docs = answer_with_sources(question)
        samples.append(
            {
                "user_input": question,
                "retrieved_contexts": [d.page_content for d in docs],
                "response": answer,
                "reference": item["ground_truth"],
            }
        )

    dataset = EvaluationDataset.from_list(samples)

    judge_llm = LangchainLLMWrapper(_eval_llm())
    judge_emb = LangchainEmbeddingsWrapper(get_embeddings())

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_emb,
        raise_exceptions=False,
        show_progress=True,
    )

    scores = _aggregate(result)
    _write_report(scores, n=len(samples))
    return scores


def _aggregate(result) -> dict:
    """Reduce RAGAS per-sample scores to mean values per metric."""
    df = result.to_pandas()
    metric_cols = [
        c
        for c in df.columns
        if c not in ("user_input", "retrieved_contexts", "response", "reference")
    ]
    scores = {}
    for col in metric_cols:
        series = df[col].dropna()
        if len(series):
            scores[col] = float(series.mean())
    return scores


def _write_report(scores: dict, n: int) -> None:
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RAGAS Evaluation Results",
        "",
        f"Evaluated **{n}** question/answer pairs from `data/eval/qa_dataset.json` "
        f"against the sample GreenLeaf corpus, judged locally with "
        f"`{EVAL_LLM_MODEL}`.",
        "",
        "| Metric | Score |",
        "|--------|-------|",
    ]
    for metric, value in scores.items():
        lines.append(f"| {metric} | {value:.3f} |")
    lines.append("")
    lines.append(
        "> Scores are produced by a local judge model; treat them as directional. "
        "Set `EVAL_LLM_MODEL` to a larger local model for more reliable judging."
    )
    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", RESULTS_MD)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    scores = run()
    print("\n=== RAGAS scores ===")
    for metric, value in scores.items():
        print(f"{metric:24s} {value:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
