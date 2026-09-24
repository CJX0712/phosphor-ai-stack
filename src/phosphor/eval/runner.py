"""Evaluation runner.

The runner always builds a brand new pipeline before scoring. Reusing the
pipeline that already served runtime traffic duplicates documents, which makes
recall drop for reasons that have nothing to do with retrieval quality.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

from ..core.events import bus
from ..core.mathx import percentile
from ..core.types import CaseResult, EvalCase, EvalReport
from .golden import corpus_documents, golden_cases
from .metrics import block_recall, doc_hit, doc_mrr, grounding

DEFAULT_THRESHOLDS = {
    "doc_hit_rate": 0.85,
    "doc_mrr": 0.60,
    "recall_at_k": 0.55,
    "grounded_rate": 0.75,
    "pass_rate": 0.75,
}


class PipelineLike(Protocol):
    def ingest_text(self, title: str, text: str, doc_id: str | None = None):
        ...

    def search(self, query: str, k: int | None = None) -> list:
        ...

    def ask(self, query: str, top_k: int | None = None) -> object:
        ...


def build_corpus_pipeline(factory: Callable[[], PipelineLike]) -> PipelineLike:
    pipeline = factory()
    for title, text in corpus_documents():
        pipeline.ingest_text(title, text, doc_id=title)
    return pipeline


def run_case(
    pipeline: PipelineLike, case: EvalCase, k: int = 6, thresholds: dict[str, float] | None = None
) -> CaseResult:
    started = time.perf_counter()
    scored = [] if case.expect_no_retrieval else pipeline.search(case.query, k=k)
    answer_obj = pipeline.ask(case.query, top_k=k)
    answer = getattr(answer_obj, "answer", str(answer_obj))
    elapsed = int((time.perf_counter() - started) * 1000)
    bus().emit("eval.case", query=case.query, elapsed_ms=elapsed)

    evidence = getattr(answer_obj, "evidence", scored)
    if not evidence:
        evidence = scored

    hit = doc_hit(scored, case.expected_doc_ids) if scored else 0.0
    mrr = doc_mrr(scored, case.expected_doc_ids) if scored else 0.0
    recall = block_recall(scored, case.expected_substrings, k) if scored else 0.0
    ground = grounding(answer, evidence)

    if case.expect_no_retrieval:
        passed = all(s.lower() in answer.lower() for s in case.expected_substrings)
        tags = tuple(case.tags) + ("no-retrieval",)
    else:
        hit_ok = hit >= 1.0 or not case.expected_doc_ids
        answer_ok = any(s.lower() in answer.lower() for s in case.expected_substrings)
        passed = bool(hit_ok and answer_ok and ground >= 0.5)
        tags = tuple(case.tags)

    return CaseResult(
        query=case.query,
        passed=passed,
        doc_hit=hit,
        doc_mrr=mrr,
        recall_at_k=recall,
        grounded=ground,
        answer=answer,
        retrieved=len(scored),
        tags=tags,
    )


def run_evaluation(
    factory: Callable[[], PipelineLike],
    cases: list[EvalCase] | None = None,
    k: int = 6,
    thresholds: dict[str, float] | None = None,
) -> EvalReport:
    """Evaluate on a freshly built pipeline; never on the runtime singleton."""
    thresholds = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    cases = cases if cases is not None else golden_cases()
    pipeline = build_corpus_pipeline(factory)

    results: list[CaseResult] = []
    latencies: list[float] = []
    for case in cases:
        started = time.perf_counter()
        result = run_case(pipeline, case, k=k, thresholds=thresholds)
        latencies.append((time.perf_counter() - started) * 1000)
        results.append(result)

    retrieval = [r for r in results if "no-retrieval" not in r.tags]
    if not retrieval:
        retrieval = results
    # Grounding is only meaningful when evidence exists: tool routing cases are
    # designed to retrieve nothing and would otherwise drag the metric down.
    grounded_pool = [r for r in results if r.retrieved > 0] or results
    report = EvalReport(
        cases=results,
        doc_hit_rate=round(sum(r.doc_hit for r in retrieval) / len(retrieval), 4),
        doc_mrr=round(sum(r.doc_mrr for r in retrieval) / len(retrieval), 4),
        recall_at_k=round(sum(r.recall_at_k for r in retrieval) / len(retrieval), 4),
        grounded_rate=round(sum(r.grounded for r in grounded_pool) / len(grounded_pool), 4),
        pass_rate=round(sum(1.0 if r.passed else 0.0 for r in results) / len(results), 4),
        retrieval_cases=len(retrieval),
        latency_ms_p50=round(percentile(latencies, 0.5), 2),
        thresholds=thresholds,
    )

    actual = {
        "doc_hit_rate": report.doc_hit_rate,
        "doc_mrr": report.doc_mrr,
        "recall_at_k": report.recall_at_k,
        "grounded_rate": report.grounded_rate,
        "pass_rate": report.pass_rate,
    }
    report.violations = [
        f"{name}: {actual[name]} < threshold {limit}"
        for name, limit in thresholds.items()
        if actual.get(name, 0.0) < limit
    ]
    return report
