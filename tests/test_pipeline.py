from phosphor.core.config import Config
from phosphor.eval.golden import corpus_documents
from phosphor.pipeline import build_pipeline


def fresh() -> object:
    return build_pipeline(Config.from_env())


def test_ingest_and_search():
    pipeline = fresh()
    report = pipeline.ingest_text("t", "Phosphor 使用 reciprocal rank fusion 融合两路召回。", doc_id="t")
    assert report.chunks >= 1
    hits = pipeline.search("reciprocal rank fusion", k=3)
    assert hits and hits[0].chunk.doc_id == "t"


def test_reingest_shorter_document_replaces_chunks():
    pipeline = fresh()
    pipeline.ingest_text("d", ("很长的文档内容。" * 200), doc_id="d")
    before = len([c for c in pipeline.chunks() if c.doc_id == "d"])
    pipeline.ingest_text("d", "短文档。", doc_id="d")
    after = len([c for c in pipeline.chunks() if c.doc_id == "d"])
    assert before > 1
    assert after == 1


def test_ask_returns_grounded_answer():
    pipeline = fresh()
    for title, text in corpus_documents():
        pipeline.ingest_text(title, text, doc_id=title)
    answer = pipeline.ask("Phosphor 的混合检索用什么融合策略？")
    assert "RRF" in answer.answer or "reciprocal rank fusion" in answer.answer
    assert answer.evidence


def test_orchestrate_runs():
    pipeline = fresh()
    for title, text in corpus_documents():
        pipeline.ingest_text(title, text, doc_id=title)
    result = pipeline.orchestrate("融合策略是什么，以及 CPU 场景推荐哪种索引？")
    assert result.answer
    assert len(result.results) >= 2


def test_stats_exposes_components():
    pipeline = fresh()
    stats = pipeline.stats()
    assert stats["embedder"] and stats["llm"]
    assert "calculator" in stats["tools"]


def test_stream_yields_pieces():
    pipeline = fresh()
    pipeline.ingest_text("s", "Phosphor 使用 RRF 融合。", doc_id="s")
    pieces = list(pipeline.stream("融合策略"))
    assert pieces


def test_deterministic_arithmetic_answer():
    pipeline = fresh()
    answer = pipeline.ask("计算 12*(3+4)")
    assert "84" in answer.answer
