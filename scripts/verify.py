"""One command verification.

    python scripts/verify.py

Eight stages, each of which can fail independently; the run short-circuits on
the first failure. The point is that "git clone && python scripts/verify.py"
is a real claim about the system running, not just about its units compiling.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

ARTIFACTS = ROOT / ".artifacts"

MODULES = [
    "phosphor",
    "phosphor.core",
    "phosphor.io",
    "phosphor.chunk",
    "phosphor.embed",
    "phosphor.store",
    "phosphor.lexical",
    "phosphor.retrieve",
    "phosphor.tools",
    "phosphor.llm",
    "phosphor.agent",
    "phosphor.orch",
    "phosphor.memory",
    "phosphor.observe",
    "phosphor.eval",
    "phosphor.pipeline",
    "phosphor.api",
    "phosphor.cli",
]


class Stage:
    def __init__(self, name: str) -> None:
        self.name = name
        self.ok = True
        self.detail: dict = {}
        self.started = time.perf_counter()

    @property
    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.started) * 1000)

    def fail(self, message: str) -> None:
        self.ok = False
        self.detail["error"] = message


def stage_guard() -> Stage:
    stage = Stage("p0_guard")
    sys.path.insert(0, str(ROOT / "scripts"))
    from scan_guard import scan  # noqa: PLC0415

    findings = scan(ROOT)
    stage.detail["findings"] = len(findings)
    if findings:
        stage.fail("; ".join(findings[:5]))
    return stage


def stage_imports() -> Stage:
    import importlib

    stage = Stage("imports")
    missing = []
    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{name}: {exc}")
    stage.detail["modules"] = len(MODULES)
    if missing:
        stage.fail("; ".join(missing))
    return stage


def stage_pytest() -> Stage:
    stage = Stage("pytest")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    summary = next(
        (line for line in reversed(lines)
         if "passed" in line or "failed" in line or "error" in line),
        "",
    )
    stage.detail["summary"] = summary.strip()
    stage.detail["exit"] = proc.returncode
    if proc.returncode != 0:
        stage.fail(summary.strip() or f"exit={proc.returncode}")
    return stage


def stage_http_e2e() -> Stage:
    stage = Stage("http_e2e")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "e2e_http.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    tail = [line for line in proc.stdout.splitlines() if line.strip()]
    stage.detail["summary"] = tail[-1].strip() if tail else ""
    stage.detail["exit"] = proc.returncode
    if proc.returncode != 0:
        stage.fail((tail[-1] if tail else "") + " :: " + proc.stderr[-500:])
    return stage


def stage_eval() -> Stage:
    stage = Stage("evaluation")
    from phosphor.core.config import Config  # noqa: PLC0415
    from phosphor.pipeline import build_pipeline  # noqa: PLC0415

    cfg = Config.from_env()
    pipeline = build_pipeline(cfg)
    report = pipeline.evaluate(k=6)
    stage.detail.update(
        {
            "doc_hit_rate": report.doc_hit_rate,
            "doc_mrr": report.doc_mrr,
            "recall_at_k": report.recall_at_k,
            "grounded_rate": report.grounded_rate,
            "pass_rate": report.pass_rate,
            "latency_ms_p50": report.latency_ms_p50,
            "violations": report.violations,
        }
    )
    if report.violations:
        stage.fail("; ".join(report.violations))
    return stage


def stage_invariants() -> Stage:
    """Runtime invariants that unit tests cannot cover end to end."""
    stage = Stage("runtime_invariants")
    from phosphor.core.config import Config  # noqa: PLC0415
    from phosphor.pipeline import build_pipeline  # noqa: PLC0415
    from phosphor.tools.arith import detect_arithmetic, safe_eval  # noqa: PLC0415

    cfg = Config.from_env()
    pipeline = build_pipeline(cfg)
    checks: dict[str, bool] = {}

    long_text = ("Phosphor 的混合检索由稠密与稀疏两路组成。" * 40)
    report = pipeline.ingest_text("long", long_text, doc_id="inv-long")
    checks["long_document_chunks_gt_1"] = report.chunks > 1

    # Re-ingesting a shorter document must not leave orphaned chunks behind.
    short = pipeline.ingest_text("long", "短文档。", doc_id="inv-long")
    checks["reingest_shorter_replaces_chunks"] = short.chunks == 1
    checks["chunk_count_after_reingest"] = (
        len([c for c in pipeline.chunks() if c.doc_id == "inv-long"]) == 1
    )

    heading_doc = "# 标题一\n正文内容一段。\n\n## 小节\n小节正文。"
    pipeline.ingest_text("heading", heading_doc, doc_id="inv-head")
    heading_chunks = [c for c in pipeline.chunks() if c.doc_id == "inv-head"]
    checks["heading_inherited"] = any(c.heading_path for c in heading_chunks)
    checks["heading_body_not_dropped"] = any(
        "正文内容" in c.text for c in heading_chunks
    )

    pipeline.ingest_text("e2e", "混合检索使用 RRF 融合。", doc_id="inv-evidence")
    hits = pipeline.search("RRF 融合", k=3)
    checks["evidence_single_line"] = all("\n" not in h.chunk.render() for h in hits)

    checks["fullwidth_arithmetic"] = safe_eval("１２×（３＋４）") == 84.0
    checks["arith_detect"] = detect_arithmetic("请计算 12*(3+4) 的结果") is not None

    answer = pipeline.ask("12*(3+4) 等于多少？")
    checks["deterministic_route_answer"] = "84" in answer.answer

    stage.detail["checks"] = checks
    failed = [k for k, v in checks.items() if not v]
    if failed:
        stage.fail("failed invariants: " + ", ".join(failed))
    return stage


def stage_determinism() -> Stage:
    stage = Stage("determinism")
    from phosphor.core.config import Config  # noqa: PLC0415
    from phosphor.pipeline import build_pipeline  # noqa: PLC0415

    def fingerprint() -> list[tuple[str, float]]:
        pipeline = build_pipeline(Config.from_env())
        for title, text in _corpus():
            pipeline.ingest_text(title, text, doc_id=title)
        return [
            (h.chunk.id, round(float(h.score), 9))
            for h in pipeline.search("混合检索用什么融合策略？", k=5)
        ]

    from phosphor.eval.golden import corpus_documents as _corpus  # noqa: PLC0415

    first = fingerprint()
    second = fingerprint()
    stage.detail["identical"] = first == second
    stage.detail["size"] = len(first)
    if first != second:
        stage.fail("retrieval is not deterministic across pipeline instances")
    return stage


def stage_openapi() -> Stage:
    stage = Stage("openapi_export")
    try:
        import yaml  # type: ignore
    except Exception:  # noqa: BLE001
        stage.detail["skipped"] = "pyyaml not installed"
        return stage
    from phosphor.api.app import create_app  # noqa: PLC0415

    app = create_app()
    spec = app.openapi()
    out = ROOT / "docs" / "openapi.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(spec, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    stage.detail["paths"] = len(spec.get("paths", {}))
    if not spec.get("paths"):
        stage.fail("openapi has no paths")
    return stage


STAGES = [
    stage_guard,
    stage_imports,
    stage_pytest,
    stage_http_e2e,
    stage_eval,
    stage_invariants,
    stage_determinism,
    stage_openapi,
]


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    results = []
    print("Phosphor verify")
    print("-" * 60)
    for runner in STAGES:
        stage = runner()
        results.append(
            {"stage": stage.name, "ok": stage.ok, "elapsed_ms": stage.elapsed_ms, **stage.detail}
        )
        status = "OK  " if stage.ok else "FAIL"
        print(f"{status} {stage.name:<20} {stage.elapsed_ms:>6} ms")
        if not stage.ok:
            print(f"     {stage.detail.get('error', '')[:400]}")
            break

    report_path = ARTIFACTS / "verify-report.json"
    report_path.write_text(
        json.dumps({"results": results, "ok": all(r["ok"] for r in results)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    ok = all(r["ok"] for r in results) and len(results) == len(STAGES)
    print("-" * 60)
    print(f"report: {report_path}")
    print("VERIFY PASSED" if ok else "VERIFY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
