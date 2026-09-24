"""FastAPI application.

Routes are registered inside the factory: a factory that leaves registration to
the caller produces an app where every route 404s while the module level app
keeps working, which is invisible until the HTTP layer is exercised.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Iterator

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from ..core.config import Config
from ..core.errors import PhosphorError, normalize_exception
from ..core.types import Message
from ..observe import metrics, tracer
from ..pipeline import Pipeline, build_pipeline
from .schemas import (
    AskRequest,
    AskResponse,
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
    ToolInvokeRequest,
    ToolInvokeResponse,
)

STATE: dict[str, Any] = {}


def get_pipeline() -> Pipeline:
    pipeline = STATE.get("pipeline")
    if pipeline is None:
        pipeline = build_pipeline(STATE.get("config"))
        STATE["pipeline"] = pipeline
    return pipeline


def _hit(item) -> SearchHit:
    return SearchHit(
        chunk_id=item.chunk.id,
        doc_id=item.chunk.doc_id,
        score=round(float(item.score), 6),
        source=item.source,
        heading=" / ".join(item.chunk.heading_path),
        text=item.chunk.text,
    )


def create_app(config: Config | None = None, pipeline: Pipeline | None = None) -> FastAPI:
    cfg = config or Config.from_env()
    STATE["config"] = cfg
    if pipeline is not None:
        STATE["pipeline"] = pipeline

    app = FastAPI(
        title="Phosphor AI Stack",
        version="1.0.0",
        description="End to end retrieval augmented agent platform.",
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.api.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    async def auth(x_api_token: str | None = Header(default=None)) -> None:
        if cfg.api.api_token and x_api_token != cfg.api.api_token:
            raise HTTPException(status_code=401, detail="invalid or missing API token")

    @app.get("/health", response_model=HealthResponse)
    def health() -> Any:
        pipe = get_pipeline()
        return HealthResponse(
            version="1.0.0",
            env=cfg.env,
            components={
                "documents": len(pipe.documents),
                "chunks": len(pipe.chunks()),
                "embedder": getattr(pipe.embedder, "name", "unknown"),
                "llm": getattr(pipe.llm, "name", "unknown"),
                "store": pipe.retriever.stats().get("dense", {}).get("provider", "memory"),
                "reranker": pipe.retriever.reranker.name if pipe.retriever.reranker else None,
            },
        )

    @app.get("/api/v1/stats", dependencies=[Depends(auth)])
    def stats() -> Any:
        return get_pipeline().stats()

    @app.get("/api/v1/metrics", dependencies=[Depends(auth)])
    def get_metrics() -> Any:
        return metrics().snapshot()

    @app.get("/api/v1/traces/{trace_id}", dependencies=[Depends(auth)])
    def get_trace(trace_id: str) -> Any:
        return {"trace_id": trace_id, "spans": tracer().trace(trace_id)}

    @app.get("/api/v1/documents", dependencies=[Depends(auth)])
    def documents() -> Any:
        pipe = get_pipeline()
        return {
            "documents": [
                {"doc_id": d.id, "title": d.title, "source": d.source}
                for d in pipe.documents.values()
            ]
        }

    @app.post("/api/v1/ingest", response_model=IngestResponse, dependencies=[Depends(auth)])
    def ingest(req: IngestRequest) -> Any:
        pipe = get_pipeline()
        reports: list[dict[str, Any]] = []
        skipped = 0
        try:
            if req.directory:
                for report in pipe.ingest_dir(req.directory):
                    reports.append(asdict(report))
                    skipped += int(report.skipped)
            else:
                if req.path:
                    report = pipe.ingest_path(req.path, doc_id=req.doc_id)
                else:
                    report = pipe.ingest_text(req.title, req.text, doc_id=req.doc_id)
                reports.append(asdict(report))
                skipped += int(report.skipped)
        except PhosphorError as exc:
            raise HTTPException(status_code=400, detail=exc.as_dict()) from exc
        return IngestResponse(ok=True, reports=reports, skipped=skipped)

    @app.delete("/api/v1/documents/{doc_id}", dependencies=[Depends(auth)])
    def drop(doc_id: str) -> Any:
        removed = get_pipeline().drop(doc_id)
        return {"doc_id": doc_id, "removed_chunks": removed}

    @app.post("/api/v1/search", response_model=SearchResponse, dependencies=[Depends(auth)])
    def search(req: SearchRequest) -> Any:
        started = time.perf_counter()
        hits = get_pipeline().search(req.query, k=req.k)
        return SearchResponse(
            query=req.query,
            hits=[_hit(h) for h in hits],
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    @app.post("/api/v1/ask", response_model=AskResponse, dependencies=[Depends(auth)])
    def ask(req: AskRequest) -> Any:
        pipe = get_pipeline()
        history: list[Message] = []
        if req.session_id and pipe.memory is not None:
            history = pipe.memory.recent(req.session_id, limit=6)
        try:
            if req.orchestrate:
                result = pipe.orchestrate(req.query)
                return AskResponse(
                    query=req.query,
                    answer=result.answer,
                    citations=[],
                    evidence=[],
                    elapsed_ms=result.elapsed_ms,
                    trace_id=result.trace_id,
                    strategy="orchestrated",
                )
            answer = pipe.ask(req.query, top_k=req.top_k)
        except PhosphorError as exc:
            detail = exc.as_dict()
            raise HTTPException(status_code=400, detail=detail) from exc
        except Exception as exc:  # noqa: BLE001
            err = normalize_exception(exc)
            raise HTTPException(status_code=500, detail=err.as_dict()) from exc
        if req.session_id and pipe.memory is not None:
            pipe.memory.append(req.session_id, Message(role="user", content=req.query))
            pipe.memory.append(req.session_id, Message(role="assistant", content=answer.answer))
        return AskResponse(
            query=req.query,
            answer=answer.answer,
            citations=answer.citations(),
            evidence=[_hit(e) for e in answer.evidence],
            tool_calls=answer.tool_calls,
            iterations=answer.iterations,
            converged=answer.converged,
            elapsed_ms=answer.elapsed_ms,
            trace_id=answer.trace_id,
            strategy="react",
        )

    @app.post("/api/v1/stream", dependencies=[Depends(auth)])
    def stream(req: AskRequest) -> Any:
        pipe = get_pipeline()

        def gen() -> Iterator[bytes]:
            for piece in pipe.stream(req.query, top_k=req.top_k):
                yield f"data: {piece}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/api/v1/tools", dependencies=[Depends(auth)])
    def list_tools() -> Any:
        return {"tools": [s.as_dict() for s in get_pipeline().tools.list_specs()]}

    @app.post("/api/v1/tools/invoke", dependencies=[Depends(auth)])
    def invoke_tool(req: ToolInvokeRequest) -> Any:
        result = get_pipeline().tools.invoke(req.name, req.arguments)
        return ToolInvokeResponse(
            ok=result.ok, output=result.output, error=result.error, elapsed_ms=result.elapsed_ms
        )

    @app.post("/api/v1/evaluate", response_model=EvaluateResponse, dependencies=[Depends(auth)])
    def evaluate(req: EvaluateRequest) -> Any:
        report = get_pipeline().evaluate(k=req.k)
        return EvaluateResponse(
            doc_hit_rate=report.doc_hit_rate,
            doc_mrr=report.doc_mrr,
            recall_at_k=report.recall_at_k,
            grounded_rate=report.grounded_rate,
            pass_rate=report.pass_rate,
            latency_ms_p50=report.latency_ms_p50,
            retrieval_cases=report.retrieval_cases,
            violations=report.violations,
            cases=[asdict(c) for c in report.cases],
        )

    @app.get("/api/v1/config", dependencies=[Depends(auth)])
    def config_view() -> Any:
        safe = cfg.to_dict()
        safe["llm"]["api_key"] = "***" if cfg.llm.api_key else ""
        safe["api"]["api_token"] = "***" if cfg.api.api_token else ""
        return safe

    return app


def default_app() -> FastAPI:
    return create_app()
