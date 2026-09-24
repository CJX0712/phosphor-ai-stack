"""Retrieval and grounding metrics."""

from __future__ import annotations

from ..core.mathx import mean
from ..core.text import cosine_sets, strip_citations, word_set
from ..core.types import CaseResult, Scored


def doc_hit(scored: list[Scored], expected: tuple[str, ...]) -> float:
    if not expected:
        return 0.0
    wanted = set(expected)
    got = {s.chunk.doc_id for s in scored}
    return 1.0 if got & wanted else 0.0


def doc_mrr(scored: list[Scored], expected: tuple[str, ...]) -> float:
    if not expected:
        return 0.0
    wanted = set(expected)
    for rank, item in enumerate(scored, start=1):
        if item.chunk.doc_id in wanted:
            return 1.0 / rank
    return 0.0


def block_recall(scored: list[Scored], substrings: tuple[str, ...], k: int) -> float:
    """Diagnostic only: fraction of expected substrings present in top-k text."""
    if not substrings:
        return 0.0
    haystack = "\n".join(s.chunk.text for s in scored[:k])
    hits = sum(1 for s in substrings if s.lower() in haystack.lower())
    return hits / len(substrings)


def grounding(answer: str, scored: list[Scored], threshold: float = 0.2) -> float:
    """Fraction of answer content supported by the cited evidence.

    Citation markers are stripped first: provenance is metadata, not an
    assertion, and leaving it in scores every grounded answer as unsupported.
    """
    clean = strip_citations(answer or "")
    if not clean:
        return 0.0
    answer_terms = word_set(clean)
    if not answer_terms:
        return 0.0
    evidence_terms: set[str] = set()
    for item in scored:
        evidence_terms |= word_set(strip_citations(item.chunk.text))
    if not evidence_terms:
        return 0.0
    support = cosine_sets(answer_terms, evidence_terms)
    if support >= threshold:
        return 1.0
    return round(support / threshold, 4)


def aggregate(cases: list[CaseResult]) -> dict[str, float]:
    retrieval = [c for c in cases if "no-retrieval" not in c.tags]
    if not retrieval:
        retrieval = cases
    return {
        "doc_hit_rate": round(mean([c.doc_hit for c in retrieval]), 4),
        "doc_mrr": round(mean([c.doc_mrr for c in retrieval]), 4),
        "recall_at_k": round(mean([c.recall_at_k for c in retrieval]), 4),
        "grounded_rate": round(mean([c.grounded for c in cases]), 4),
        "pass_rate": round(mean([1.0 if c.passed else 0.0 for c in cases]), 4),
        "retrieval_cases": float(len(retrieval)),
    }
