"""Orchestration contracts: planning and critique are injectable."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(slots=True)
class SubTask:
    id: str
    question: str
    depends_on: tuple[str, ...] = ()
    kind: str = "retrieval"  # retrieval | tool | synthesis


@dataclass(slots=True)
class Plan:
    query: str
    tasks: list[SubTask] = field(default_factory=list)
    strategy: str = "decompose"

    @property
    def size(self) -> int:
        return len(self.tasks)


@dataclass(slots=True)
class TaskResult:
    task: SubTask
    answer: str = ""
    evidence: list[Any] = field(default_factory=list)
    ok: bool = True
    issue: str = ""


@dataclass(slots=True)
class Critique:
    score: float = 1.0
    issues: list[str] = field(default_factory=list)
    retry: bool = False
    accepted: bool = True


@runtime_checkable
class Planner(Protocol):
    name: str

    def plan(self, query: str) -> Plan:
        ...


@runtime_checkable
class Critic(Protocol):
    name: str

    def review(self, query: str, results: list[TaskResult]) -> Critique:
        ...
