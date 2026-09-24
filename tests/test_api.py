import pytest

from phosphor.api.app import STATE, create_app
from phosphor.core.config import Config
from phosphor.pipeline import build_pipeline

fastapi = pytest.importorskip("fastapi")
TestClient = pytest.importorskip("starlette.testclient").TestClient


@pytest.fixture()
def client():
    cfg = Config.from_env()
    cfg.api.api_token = ""
    pipeline = build_pipeline(cfg)
    for title, text in corpus():
        pipeline.ingest_text(title, text, doc_id=title)
    STATE.clear()
    app = create_app(cfg, pipeline=pipeline)
    with TestClient(app) as c:
        yield c
    STATE.clear()


def corpus():
    from phosphor.eval.golden import corpus_documents

    return corpus_documents()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ingest_and_search(client):
    resp = client.post("/api/v1/ingest", json={"title": "api", "doc_id": "api-doc",
                                               "text": "Phosphor 使用 RRF 融合两路召回。"})
    assert resp.status_code == 200
    resp = client.post("/api/v1/search", json={"query": "RRF 融合", "k": 3})
    assert resp.status_code == 200
    assert resp.json()["hits"]


def test_ask(client):
    resp = client.post("/api/v1/ask", json={"query": "混合检索用什么融合策略？"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert body["trace_id"]


def test_ask_orchestrated(client):
    resp = client.post("/api/v1/ask", json={"query": "融合策略是什么，以及推荐哪种索引？",
                                            "orchestrate": True})
    assert resp.status_code == 200
    assert resp.json()["strategy"] == "orchestrated"


def test_tools_and_invoke(client):
    resp = client.get("/api/v1/tools")
    assert any(t["name"] == "calculator" for t in resp.json()["tools"])
    resp = client.post("/api/v1/tools/invoke", json={"name": "calculator",
                                                     "arguments": {"expression": "6*7"}})
    assert resp.json()["ok"] and "42" in resp.json()["output"]


def test_empty_ingest_returns_400(client):
    resp = client.post("/api/v1/ingest", json={"text": "   "})
    assert resp.status_code == 400


def test_delete_document(client):
    client.post("/api/v1/ingest", json={"doc_id": "tmp", "text": "一些内容"})
    resp = client.delete("/api/v1/documents/tmp")
    assert resp.status_code == 200


def test_metrics_endpoint(client):
    resp = client.get("/api/v1/metrics")
    assert resp.status_code == 200


def test_openapi_contains_paths(client):
    spec = client.get("/openapi.json").json()
    assert "/api/v1/ask" in spec["paths"]


def test_auth_is_enforced_when_token_set():
    cfg = Config.from_env()
    cfg.api.api_token = "secret"
    STATE.clear()
    app = create_app(cfg, pipeline=build_pipeline(cfg))
    with TestClient(app) as c:
        assert c.get("/api/v1/stats").status_code == 401
        assert c.get("/api/v1/stats", headers={"X-API-Token": "secret"}).status_code == 200
    STATE.clear()
