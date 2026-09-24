"""Command line interface.

argparse only: the CLI must work in the same bare environment that
scripts/verify.py targets.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any

from ..core.config import Config
from ..pipeline import build_pipeline


def _pipeline(args: argparse.Namespace) -> Any:
    cfg = Config.from_env()
    if getattr(args, "top_k", None):
        cfg.retrieve.top_k = args.top_k
    return build_pipeline(cfg)


def cmd_ingest(args: argparse.Namespace) -> int:
    pipeline = _pipeline(args)
    reports = []
    if args.directory:
        reports = [asdict(r) for r in pipeline.ingest_dir(args.directory)]
    elif args.path:
        reports = [asdict(pipeline.ingest_path(args.path, doc_id=args.doc_id))]
    elif args.text:
        reports = [asdict(pipeline.ingest_text(args.title, args.text, doc_id=args.doc_id))]
    else:
        print("--path, --directory or --text is required", file=sys.stderr)
        return 2
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    pipeline = _pipeline(args)
    for title, text in _demo_corpus():
        pipeline.ingest_text(title, text, doc_id=title)
    hits = pipeline.search(args.query, k=args.top_k or 6)
    print(json.dumps(
        [{"doc_id": h.chunk.doc_id, "chunk_id": h.chunk.id, "score": round(h.score, 4),
          "heading": " / ".join(h.chunk.heading_path), "text": h.chunk.text[:200]}
         for h in hits],
        ensure_ascii=False, indent=2))
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    pipeline = _pipeline(args)
    for title, text in _demo_corpus():
        pipeline.ingest_text(title, text, doc_id=title)
    if args.orchestrate:
        result = pipeline.orchestrate(args.query)
        payload = {
            "answer": result.answer,
            "rounds": result.rounds,
            "tasks": len(result.results),
            "critique": result.critique.score if result.critique else None,
            "elapsed_ms": result.elapsed_ms,
            "trace_id": result.trace_id,
        }
    else:
        answer = pipeline.ask(args.query, top_k=args.top_k)
        payload = {
            "answer": answer.answer,
            "citations": answer.citations(),
            "evidence": [e.chunk.id for e in answer.evidence],
            "tool_calls": answer.tool_calls,
            "iterations": answer.iterations,
            "elapsed_ms": answer.elapsed_ms,
            "trace_id": answer.trace_id,
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    pipeline = _pipeline(args)
    report = pipeline.evaluate(k=args.top_k or 6)
    payload = {
        "doc_hit_rate": report.doc_hit_rate,
        "doc_mrr": report.doc_mrr,
        "recall_at_k": report.recall_at_k,
        "grounded_rate": report.grounded_rate,
        "pass_rate": report.pass_rate,
        "latency_ms_p50": report.latency_ms_p50,
        "violations": report.violations,
        "cases": [
            {"query": c.query, "passed": c.passed, "doc_hit": c.doc_hit,
             "grounded": c.grounded, "answer": c.answer[:120]}
            for c in report.cases
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if report.violations else 0


def cmd_tools(args: argparse.Namespace) -> int:
    pipeline = _pipeline(args)
    specs = [s.as_dict() for s in pipeline.tools.list_specs()]
    print(json.dumps(specs, ensure_ascii=False, indent=2))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    pipeline = _pipeline(args)
    print(json.dumps(pipeline.stats(), ensure_ascii=False, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    cfg = Config.from_env()
    if args.host:
        cfg.api.host = args.host
    if args.port:
        cfg.api.port = args.port
    from ..api.app import create_app

    app = create_app(cfg)
    uvicorn.run(app, host=cfg.api.host, port=cfg.api.port, log_level="info")
    return 0


def _demo_corpus() -> list[tuple[str, str]]:
    from ..eval.golden import corpus_documents

    return corpus_documents()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phosphor", description="Phosphor AI Stack CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="ingest a file, directory or inline text")
    p_ingest.add_argument("--path")
    p_ingest.add_argument("--directory")
    p_ingest.add_argument("--text")
    p_ingest.add_argument("--title", default="inline")
    p_ingest.add_argument("--doc-id", dest="doc_id", default=None)
    p_ingest.add_argument("--top-k", dest="top_k", type=int, default=None)
    p_ingest.set_defaults(func=cmd_ingest)

    p_search = sub.add_parser("search", help="hybrid search over the built-in corpus")
    p_search.add_argument("query")
    p_search.add_argument("--top-k", dest="top_k", type=int, default=6)
    p_search.set_defaults(func=cmd_search)

    p_ask = sub.add_parser("ask", help="ask a question (ReAct agent)")
    p_ask.add_argument("query")
    p_ask.add_argument("--top-k", dest="top_k", type=int, default=6)
    p_ask.add_argument("--orchestrate", action="store_true")
    p_ask.set_defaults(func=cmd_ask)

    p_eval = sub.add_parser("eval", help="run the golden evaluation")
    p_eval.add_argument("--top-k", dest="top_k", type=int, default=6)
    p_eval.set_defaults(func=cmd_eval)

    p_tools = sub.add_parser("tools", help="list tools")
    p_tools.add_argument("--top-k", dest="top_k", type=int, default=None)
    p_tools.set_defaults(func=cmd_tools)

    p_stats = sub.add_parser("stats", help="pipeline statistics")
    p_stats.add_argument("--top-k", dest="top_k", type=int, default=None)
    p_stats.set_defaults(func=cmd_stats)

    p_serve = sub.add_parser("serve", help="start the HTTP API")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
