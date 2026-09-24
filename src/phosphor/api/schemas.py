"""HTTP request/response models.

Pydantic lives here and only here: the domain model stays dependency free.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    text: str = ""
    title: str = "inline"
    doc_id: str | None = None
    path: str | None = None
    directory: str | None = None


class IngestResponse(BaseModel):
    ok: bool = True
    reports: list[dict[str, Any]] = Field(default_factory=list)
    skipped: int = 0


class SearchRequest(BaseModel):
    query: str
    k: int = 6


class SearchHit(BaseModel):
    chunk_id: str
    doc_id: str
    score: float
    source: str
    heading: str = ""
    text: str


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit] = Field(default_factory=list)
    elapsed_ms: int = 0


class AskRequest(BaseModel):
    query: str
    top_k: int = 6
    orchestrate: bool = False
    session_id: str = ""


class AskResponse(BaseModel):
    query: str
    answer: str
    citations: list[str] = Field(default_factory=list)
    evidence: list[SearchHit] = Field(default_factory=list)
    tool_calls: int = 0
    iterations: int = 0
    converged: bool = True
    elapsed_ms: int = 0
    trace_id: str = ""
    strategy: str = "react"


class ToolInvokeRequest(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolInvokeResponse(BaseModel):
    ok: bool
    output: str = ""
    error: str = ""
    elapsed_ms: int = 0


class EvaluateRequest(BaseModel):
    k: int = 6


class EvaluateResponse(BaseModel):
    doc_hit_rate: float
    doc_mrr: float
    recall_at_k: float
    grounded_rate: float
    pass_rate: float
    latency_ms_p50: float
    retrieval_cases: int
    violations: list[str] = Field(default_factory=list)
    cases: list[dict[str, Any]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = ""
    env: str = ""
    components: dict[str, Any] = Field(default_factory=dict)
