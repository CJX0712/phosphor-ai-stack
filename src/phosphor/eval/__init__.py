"""Evaluation boundary."""

from .golden import CORPUS, GOLDEN, corpus_documents, golden_cases
from .metrics import block_recall, doc_hit, doc_mrr, grounding
from .runner import DEFAULT_THRESHOLDS, build_corpus_pipeline, run_case, run_evaluation

__all__ = [
    "CORPUS",
    "DEFAULT_THRESHOLDS",
    "GOLDEN",
    "block_recall",
    "build_corpus_pipeline",
    "corpus_documents",
    "doc_hit",
    "doc_mrr",
    "golden_cases",
    "grounding",
    "run_case",
    "run_evaluation",
]
