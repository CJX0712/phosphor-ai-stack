"""Self contained HTTP end to end check.

curl, docker compose and jest are unreliable in the sandbox, so the smoke test
is a single python script that boots the real ASGI app in-process and drives it
over HTTP with httpx, asserting both the success path and every error path.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import httpx  # noqa: E402
import uvicorn  # noqa: E402

from phosphor.api.app import create_app  # noqa: E402
from phosphor.core.config import Config  # noqa: E402


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Server:
    def __init__(self, app, port: int) -> None:
        self.config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        self.server = uvicorn.Server(self.config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self, timeout: float = 30.0) -> None:
        self.thread.start()
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.server.started:
                return
            time.sleep(0.1)
        raise RuntimeError("server did not start; check port availability")

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


def main() -> int:
    cfg = Config.from_env()
    cfg.api.api_token = ""
    app = create_app(cfg)
    port = free_port()
    server = Server(app, port)
    server.start()
    base = f"http://127.0.0.1:{port}"
    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  PASS  {name}")
        else:
            failed += 1
            print(f"  FAIL  {name} {detail}")

    try:
        with httpx.Client(base_url=base, timeout=60.0, trust_env=False) as client:
            resp = client.get("/health")
            check("GET /health -> 200", resp.status_code == 200, str(resp.status_code))
            body = resp.json()
            check("health status ok", body.get("status") == "ok", str(body))

            resp = client.post(
                "/api/v1/ingest",
                json={"title": "e2e", "doc_id": "e2e-doc",
                      "text": "Phosphor merges dense and sparse retrieval with reciprocal rank fusion. "
                              "The reranker is lexical by default and switches to a cross encoder "
                              "when fastembed is installed."},
            )
            check("POST /api/v1/ingest -> 200", resp.status_code == 200, resp.text[:200])
            check("ingest produced chunks",
                  bool(resp.json().get("reports")) and resp.json()["reports"][0]["chunks"] >= 1,
                  resp.text[:200])

            resp = client.get("/api/v1/documents")
            check("GET /api/v1/documents lists doc",
                  any(d["doc_id"] == "e2e-doc" for d in resp.json().get("documents", [])),
                  resp.text[:200])

            resp = client.post("/api/v1/search", json={"query": "fusion strategy", "k": 3})
            check("POST /api/v1/search -> 200", resp.status_code == 200, resp.text[:200])
            check("search returned hits", len(resp.json().get("hits", [])) >= 1, resp.text[:200])

            resp = client.post("/api/v1/ask", json={"query": "What fusion strategy does Phosphor use?"})
            check("POST /api/v1/ask -> 200", resp.status_code == 200, resp.text[:200])
            ask_body = resp.json()
            check("ask returned answer", bool(ask_body.get("answer")), str(ask_body)[:200])
            check("ask has trace id", bool(ask_body.get("trace_id")), str(ask_body)[:200])

            resp = client.post("/api/v1/ask", json={"query": "12*(3+4) 等于多少？"})
            check("deterministic routing answers 84",
                  "84" in resp.json().get("answer", ""), resp.text[:200])

            resp = client.post("/api/v1/ask", json={"query": "复合问题?", "orchestrate": True})
            check("orchestrated ask -> 200", resp.status_code == 200, resp.text[:200])

            resp = client.get("/api/v1/tools")
            check("GET /api/v1/tools has calculator",
                  any(t["name"] == "calculator" for t in resp.json().get("tools", [])),
                  resp.text[:200])

            resp = client.post("/api/v1/tools/invoke",
                               json={"name": "calculator", "arguments": {"expression": "7*6"}})
            check("tool invoke calculator -> 42", "42" in resp.text, resp.text[:200])

            resp = client.post("/api/v1/tools/invoke", json={"name": "nope", "arguments": {}})
            check("unknown tool is reported not raised",
                  resp.status_code == 200 and resp.json().get("ok") is False, resp.text[:200])

            resp = client.get("/api/v1/metrics")
            check("GET /api/v1/metrics -> 200", resp.status_code == 200, resp.text[:120])

            resp = client.get("/api/v1/stats")
            check("GET /api/v1/stats -> 200", resp.status_code == 200, resp.text[:120])

            resp = client.post("/api/v1/ingest", json={"text": "   "})
            check("empty ingest -> 400", resp.status_code == 400, str(resp.status_code))

            resp = client.delete("/api/v1/documents/e2e-doc")
            check("DELETE document -> 200", resp.status_code == 200, resp.text[:120])

            resp = client.post("/api/v1/evaluate", json={"k": 6})
            check("POST /api/v1/evaluate -> 200", resp.status_code == 200, resp.text[:160])
            eval_body = resp.json()
            check("evaluate has no violations", not eval_body.get("violations"),
                  str(eval_body.get("violations"))[:200])

            resp = client.get("/openapi.json")
            check("GET /openapi.json -> 200", resp.status_code == 200, str(resp.status_code))
    finally:
        server.stop()

    print(f"HTTP E2E: passed {passed}, failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
