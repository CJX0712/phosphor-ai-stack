"""Heuristic planner and lexical critic.

Both are real implementations, not placeholders: the planner splits compound
questions on discourse markers and the critic measures whether each sub answer
is actually supported by the evidence it claims to use. A model backed
planner/critic can be injected through the same Protocol.
"""

from __future__ import annotations

import re

from ..core.ids import stable_id
from ..core.text import cosine_sets, strip_citations, word_set
from .protocol import Critic, Critique, Plan, Planner, SubTask, TaskResult

_SPLIT_RE = re.compile(r"(?:以及|并且|同时|另外|还有|然后|和|与|，|；|;|,| and | then | also )", re.I)
_COMPOUND_HINT = re.compile(r"(?:以及|并且|同时|另外|还有|和|与|，|；|;|,)", re.I)
_MIN_PART = 6


class HeuristicPlanner:
    """Split a compound question into independently answerable sub questions."""

    name = "heuristic"

    def __init__(self, max_tasks: int = 4) -> None:
        self.max_tasks = max_tasks

    def plan(self, query: str) -> Plan:
        stripped = query.strip()
        parts = [p.strip(" ?？。.") for p in _SPLIT_RE.split(stripped)]
        parts = [p for p in parts if len(p) >= _MIN_PART]
        if len(parts) <= 1 or not _COMPOUND_HINT.search(stripped):
            return Plan(query=stripped, tasks=[SubTask(id="t0", question=stripped)], strategy="single")
        tasks: list[SubTask] = []
        for i, part in enumerate(parts[: self.max_tasks]):
            tasks.append(
                SubTask(
                    id=stable_id(stripped, str(i), prefix="t"),
                    question=part,
                    depends_on=(),
                    kind="retrieval",
                )
            )
        return Plan(query=stripped, tasks=tasks, strategy="decompose")


class LexicalCritic:
    """Score sub answers by evidence support and question coverage."""

    name = "lexical"

    def __init__(self, min_support: float = 0.18, min_score: float = 0.5) -> None:
        self.min_support = min_support
        self.min_score = min_score

    def review(self, query: str, results: list[TaskResult]) -> Critique:
        if not results:
            return Critique(score=0.0, issues=["no results"], retry=True, accepted=False)
        issues: list[str] = []
        scores: list[float] = []
        for result in results:
            if not result.ok:
                issues.append(f"{result.task.id}: {result.issue}")
                scores.append(0.0)
                continue
            answer_terms = word_set(strip_citations(result.answer))
            if not answer_terms:
                issues.append(f"{result.task.id}: empty answer")
                scores.append(0.0)
                continue
            question_terms = word_set(result.task.question)
            coverage = cosine_sets(question_terms, answer_terms)
            evidence_terms: set[str] = set()
            for item in result.evidence:
                text = getattr(getattr(item, "chunk", None), "text", str(item))
                evidence_terms |= word_set(strip_citations(text))
            support = cosine_sets(answer_terms, evidence_terms) if evidence_terms else 0.0
            scores.append(min(1.0, coverage * 0.5 + support * 0.5))
            if support < self.min_support:
                issues.append(f"{result.task.id}: unsupported by evidence")
        score = sum(scores) / len(scores) if scores else 0.0
        accepted = score >= self.min_score
        return Critique(
            score=round(score, 4),
            issues=issues,
            retry=not accepted,
            accepted=accepted,
        )
