from phosphor.core.types import Chunk, Scored
from phosphor.eval import block_recall, doc_hit, doc_mrr, grounding, run_evaluation
from phosphor.eval.golden import corpus_documents, golden_cases
from phosphor.pipeline import build_pipeline


def scored(doc_id: str, text: str) -> Scored:
    return Scored(chunk=Chunk(id=f"c-{doc_id}", doc_id=doc_id, text=text), score=1.0)


def test_doc_hit_and_mrr():
    hits = [scored("wrong", "x"), scored("right", "y")]
    assert doc_hit(hits, ("right",)) == 1.0
    assert doc_mrr(hits, ("right",)) == 0.5
    assert doc_hit(hits, ("absent",)) == 0.0


def test_grounding_strips_citations():
    hits = [scored("d", "Phosphor 使用 reciprocal rank fusion 融合两路召回")]
    assert grounding("使用 reciprocal rank fusion 融合 (source: d)", hits) >= 0.5


def test_grounding_rejects_unsupported_answer():
    hits = [scored("d", "向量索引 HNSW 适合 CPU 部署")]
    assert grounding("完全无关的答案内容", hits) == 0.0


def test_block_recall_is_fraction_of_expected_terms():
    hits = [scored("d", "reciprocal rank fusion RRF")]
    assert block_recall(hits, ("RRF", "missing"), k=3) == 0.5


def test_golden_corpus_is_non_uniform_and_long():
    docs = corpus_documents()
    assert len(docs) >= 3
    lengths = {len(text) for _t, text in docs}
    assert len(lengths) == len(docs), "corpus lengths must differ"


def test_golden_evaluation_passes_thresholds():
    report = run_evaluation(build_pipeline, cases=golden_cases(), k=6)
    assert report.doc_hit_rate >= 0.85, report.violations
    assert report.doc_mrr >= 0.60, report.violations
    assert report.grounded_rate >= 0.75, report.violations
    assert not report.violations


def test_evaluation_uses_a_fresh_pipeline_each_time():
    first = run_evaluation(build_pipeline, cases=golden_cases()[:3])
    second = run_evaluation(build_pipeline, cases=golden_cases()[:3])
    assert first.doc_hit_rate == second.doc_hit_rate
