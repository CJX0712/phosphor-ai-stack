"""Plan -> execute -> critique -> synthesise.

The orchestrator never talks to a model directly: it composes the agent (which
owns the ReAct loop) and the retriever, which keeps both independently testable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..agent.react import ReActAgent
from ..core.config import OrchestrationConfig
from ..core.errors import OrchestrationError
from ..core.events import bus
from ..core.ids import random_id
from ..core.types import AgentAnswer
from .graph import HeuristicPlanner, LexicalCritic
from .protocol import (
    Critic,
    Critique,
    Plan,
    Planner,
    TaskResult,
)


@dataclass(slots=True)
class OrchestratedResult:
    query: str
    answer: str
    plan: Plan
    results: list[TaskResult] = field(default_factory=list)
    critique: Critique | None = None
    rounds: int = 0
    elapsed_ms: int = 0
    trace_id: str = ""
    meta: dict[str, object] = field(default_factory=dict)


class Orchestrator:
    def __init__(
        self,
        agent: ReActAgent,
        cfg: OrchestrationConfig | None = None,
        planner: Planner | None = None,
        critic: Critic | None = None,
    ) -> None:
        self.agent = agent
        self.cfg = cfg or OrchestrationConfig()
        self.planner = planner or HeuristicPlanner()
        self.critic = critic or LexicalCritic()

    def run(self, query: str, trace_id: str = "") -> OrchestratedResult:
        started = time.perf_counter()
        trace_id = trace_id or random_id("tr_")
        plan = self.planner.plan(query)
        if not plan.tasks:
            raise OrchestrationError("planner produced no tasks")

        results: list[TaskResult] = []
        critique: Critique | None = None
        rounds = 0
        for round_index in range(max(1, self.cfg.max_rounds)):
            rounds = round_index + 1
            results = [self._execute(task, trace_id) for task in plan.tasks]
            critique = self.critic.review(query, results) if self.cfg.critic_enabled else None
            bus().emit(
                "orch.round",
                trace_id=trace_id,
                round=rounds,
                tasks=len(results),
                score=critique.score if critique else None,
            )
            if critique is None or critique.accepted:
                break

        answer = self._synthesize(query, results, critique)
        return OrchestratedResult(
            query=query,
            answer=answer,
            plan=plan,
            results=results,
            critique=critique,
            rounds=rounds,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            trace_id=trace_id,
            meta={
                "planner": self.planner.name,
                "critic": self.critic.name if self.cfg.critic_enabled else None,
                "strategy": plan.strategy,
            },
        )

    def _execute(self, task: TaskResult | Plan | object, trace_id: str) -> TaskResult:
        if not isinstance(task, TaskResult) and not hasattr(task, "question"):
            raise OrchestrationError("invalid task")
        subtask = task  # type: ignore[assignment]
        answer: AgentAnswer = self.agent.run(subtask.question, trace_id=trace_id)
        ok = bool(answer.answer) and "cannot answer" not in answer.answer.lower()
        return TaskResult(
            task=subtask,
            answer=answer.answer,
            evidence=list(answer.evidence),
            ok=ok,
            issue="" if ok else "no supported answer",
        )

    def _synthesize(
        self, query: str, results: list[TaskResult], critique: Critique | None
    ) -> str:
        if len(results) == 1:
            return results[0].answer
        parts: list[str] = []
        for result in results:
            if result.answer:
                parts.append(f"- {result.answer}")
        if not parts:
            return f"cannot answer from available evidence: {query}"
        header = f"Answer ({len(results)} sub questions):"
        if critique is not None and critique.issues:
            header += f" [critique score {critique.score}]"
        return "\n".join([header, *parts])
