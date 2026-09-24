"""Assembly root.

The only place that knows every concrete implementation. Every other module
depends on a Protocol, which is what makes the stack testable with fakes and
reconfigurable by environment without touching business logic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .agent.react import ReActAgent
from .chunk import split_document
from .core.config import Config
from .core.errors import IngestError
from .core.events import bus
from .core.ids import random_id
from .core.types import (
    AgentAnswer,
    Chunk,
    Document,
    EvalReport,
    IngestReport,
    Scored,
)
from .embed import build_embedder
from .eval import run_evaluation
from .eval.golden import golden_cases
from .io import load_dir, load_path, load_text
from .llm import build_llm
from .memory import build_memory
from .observe import metrics
from .orch import OrchestratedResult, Orchestrator
from .retrieve import HybridRetriever, build_retriever, build_reranker
from .store import build_store
from .tools import ToolRegistry, register_builtins, route_deterministic


@dataclass
class Pipeline:
    cfg: Config
    embedder: Any
    store: Any
    retriever: HybridRetriever
    llm: Any
    tools: ToolRegistry
    agent: ReActAgent
    orchestrator: Orchestrator
    memory: Any = None
    documents: dict[str, Document] = field(default_factory=dict)

    # -- ingestion --------------------------------------------------------
    def ingest(self, doc: Document) -> IngestReport:
        started = time.perf_counter()
        if not doc.text.strip():
            return IngestReport(doc_id=doc.id, title=doc.title, skipped=True, reason="empty")
        # drop first: upsert is not replace, and a shorter re-ingest would
        # otherwise leave the tail chunks of the previous revision orphaned.
        self.retriever.drop_document(doc.id)
        chunks = split_document(doc, self.cfg.chunk)
        if not chunks:
            return IngestReport(doc_id=doc.id, title=doc.title, skipped=True, reason="no chunks")
        vectors = self.embedder.embed([c.text for c in chunks])
        self.retriever.index(chunks, vectors)
        self.documents[doc.id] = doc
        report = IngestReport(
            doc_id=doc.id,
            title=doc.title,
            chunks=len(chunks),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
        metrics().inc("ingest.documents")
        metrics().inc("ingest.chunks", len(chunks))
        bus().emit("pipeline.ingest", doc_id=doc.id, chunks=len(chunks))
        return report

    def ingest_text(self, title: str, text: str, doc_id: str | None = None) -> IngestReport:
        return self.ingest(load_text(title, text, doc_id=doc_id))

    def ingest_path(self, path: str | Path, doc_id: str | None = None) -> IngestReport:
        return self.ingest(load_path(path, doc_id=doc_id))

    def ingest_dir(self, path: str | Path) -> list[IngestReport]:
        return [self.ingest(doc) for doc in load_dir(path)]

    def drop(self, doc_id: str) -> int:
        self.documents.pop(doc_id, None)
        return self.retriever.drop_document(doc_id)

    # -- retrieval --------------------------------------------------------
    def search(self, query: str, k: int | None = None, trace_id: str = "") -> list[Scored]:
        with metrics().timer("retrieval.latency"):
            hits = self.retriever.search(query, k=k, trace_id=trace_id)
        metrics().inc("retrieval.queries")
        return hits

    # -- reasoning --------------------------------------------------------
    def ask(self, query: str, top_k: int | None = None, trace_id: str = "") -> AgentAnswer:
        trace_id = trace_id or random_id("tr_")
        with metrics().timer("agent.latency"):
            answer = self.agent.run(query, top_k=top_k, trace_id=trace_id)
        metrics().inc("agent.queries")
        if self.memory is not None:
            from .core.types import Message

            self.memory.append(trace_id, Message(role="user", content=query))
            self.memory.append(trace_id, Message(role="assistant", content=answer.answer))
        return answer

    def orchestrate(self, query: str, trace_id: str = "") -> OrchestratedResult:
        with metrics().timer("orch.latency"):
            result = self.orchestrator.run(query, trace_id=trace_id)
        metrics().inc("orch.queries")
        return result

    def stream(self, query: str, top_k: int | None = None) -> Iterator[str]:
        hits = self.search(query, k=top_k)
        from .agent.prompts import build_messages

        evidence = "\n".join(s.chunk.render() for s in hits)
        messages = build_messages(query, evidence)
        for piece in self.llm.stream(messages):
            yield piece

    # -- introspection ----------------------------------------------------
    def evaluate(self, k: int = 6) -> EvalReport:
        """Evaluate on a fresh pipeline built from the same configuration."""
        return run_evaluation(lambda: build_pipeline(self.cfg), cases=golden_cases(), k=k)

    def chunks(self) -> list[Chunk]:
        return self.retriever.chunks()

    def stats(self) -> dict:
        return {
            "env": self.cfg.env,
            "documents": len(self.documents),
            "retriever": self.retriever.stats(),
            "llm": getattr(self.llm, "name", "unknown"),
            "llm_model": getattr(self.llm, "model", ""),
            "embedder": getattr(self.embedder, "name", "unknown"),
            "tools": [s.name for s in self.tools.list_specs()],
            "memory": self.memory.stats() if self.memory else None,
        }


def build_pipeline(cfg: Config | None = None, allow_network: bool = False) -> Pipeline:
    cfg = cfg or Config.from_env()
    embedder = build_embedder(cfg.embed)
    store = build_store(cfg.store, dim=cfg.embed.dim)
    reranker = build_reranker(
        cfg.retrieve.rerank, cfg.retrieve.rerank_model, cfg.retrieve.rerank_provider
    )
    retriever = build_retriever(cfg.retrieve, embedder, store, reranker)
    llm = build_llm(cfg.llm)
    tools = register_builtins(
        ToolRegistry(allow_network=allow_network), retriever=retriever, allow_network=allow_network
    )
    agent = ReActAgent(
        llm=llm,
        retriever=retriever,
        tools=tools,
        cfg=cfg.agent,
        deterministic_router=route_deterministic if cfg.agent.deterministic_routing else None,
    )
    orchestrator = Orchestrator(agent=agent, cfg=cfg.orch)
    memory = build_memory(path=str(Path(cfg.data_dir) / "memory.json") if cfg.data_dir else "")
    return Pipeline(
        cfg=cfg,
        embedder=embedder,
        store=store,
        retriever=retriever,
        llm=llm,
        tools=tools,
        agent=agent,
        orchestrator=orchestrator,
        memory=memory,
    )
