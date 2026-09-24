"""Core value types.

Plain dataclasses on purpose: the domain model must be importable and
testable with zero third party packages. Pydantic models live only in the API
layer, where serialization validation is actually needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class Document:
    id: str
    title: str
    text: str
    source: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        from .ids import content_hash

        return content_hash(self.text)


@dataclass(slots=True)
class Chunk:
    """A slice of a document that carries its heading ancestry."""

    id: str
    doc_id: str
    text: str
    order: int = 0
    start: int = 0
    end: int = 0
    heading_path: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def render(self, max_chars: int = 1200) -> str:
        """Single line rendering.

        Multi line evidence blocks get parsed line by line downstream, which
        silently drops everything but the first line of every chunk.
        """
        body = " ".join(self.text.split())
        if len(body) > max_chars:
            body = body[: max_chars - 1].rstrip() + "…"
        path = " / ".join(self.heading_path)
        return f"[id#{self.id}]" + (f" ({path}) " if path else " ") + body


@dataclass(slots=True)
class Scored:
    """A chunk plus the score that produced it and where the score came from."""

    chunk: Chunk
    score: float = 0.0
    source: str = ""
    detail: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class Message:
    role: Role
    content: str
    name: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""
    raw: str = ""


@dataclass(slots=True)
class ToolResult:
    call: ToolCall
    output: str = ""
    ok: bool = True
    error: str = ""
    elapsed_ms: int = 0

    def observation(self) -> str:
        if self.ok:
            return self.output
        return f"tool_error: {self.error}"


@dataclass(slots=True)
class AgentStep:
    index: int
    thought: str = ""
    action: str = ""
    observation: str = ""
    tool_result: ToolResult | None = None


@dataclass(slots=True)
class AgentAnswer:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    evidence: list[Scored] = field(default_factory=list)
    tool_calls: int = 0
    iterations: int = 0
    converged: bool = True
    elapsed_ms: int = 0
    trace_id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def citations(self) -> list[str]:
        seen: list[str] = []
        for s in self.evidence:
            cid = s.chunk.id
            if cid not in seen:
                seen.append(cid)
        return seen


@dataclass(slots=True)
class IngestReport:
    doc_id: str
    title: str
    chunks: int = 0
    skipped: bool = False
    reason: str = ""
    elapsed_ms: int = 0


@dataclass(slots=True)
class EvalCase:
    query: str
    expected_substrings: tuple[str, ...] = ()
    expected_doc_ids: tuple[str, ...] = ()
    expect_no_retrieval: bool = False
    tags: tuple[str, ...] = ()


@dataclass(slots=True)
class CaseResult:
    query: str
    passed: bool
    doc_hit: float = 0.0
    doc_mrr: float = 0.0
    recall_at_k: float = 0.0
    grounded: float = 0.0
    answer: str = ""
    retrieved: int = 0
    failure: str = ""
    tags: tuple[str, ...] = ()


@dataclass(slots=True)
class EvalReport:
    cases: list[CaseResult] = field(default_factory=list)
    doc_hit_rate: float = 0.0
    doc_mrr: float = 0.0
    recall_at_k: float = 0.0
    grounded_rate: float = 0.0
    pass_rate: float = 0.0
    retrieval_cases: int = 0
    latency_ms_p50: float = 0.0
    thresholds: dict[str, float] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations
