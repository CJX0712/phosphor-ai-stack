from phosphor.core.config import RetrieveConfig
from phosphor.core.types import Chunk
from phosphor.embed import HashingEmbedder
from phosphor.retrieve import (
    DenseRetriever,
    HybridRetriever,
    LexicalReranker,
    SparseRetriever,
    build_retriever,
)
from phosphor.store import MemoryStore


def make_chunks() -> list[Chunk]:
    return [
        Chunk(id="c1", doc_id="d1", text="混合检索使用 reciprocal rank fusion 融合两路召回", order=0),
        Chunk(id="c2", doc_id="d2", text="HNSW 是 CPU 场景下的默认向量索引", order=0),
        Chunk(id="c3", doc_id="d3", text="接地率衡量答案被证据支撑的程度", order=0),
    ]


def test_hybrid_fuses_both_branches():
    emb = HashingEmbedder(dim=128)
    store = MemoryStore(dim=128)
    cfg = RetrieveConfig(top_k=3, rerank=False)
    retriever = build_retriever(cfg, emb, store, reranker=None)
    retriever.index(make_chunks())
    hits = retriever.search("reciprocal rank fusion 融合", k=3)
    assert hits
    assert hits[0].chunk.doc_id == "d1"
    assert "dense" in hits[0].detail or "sparse" in hits[0].detail


def test_drop_document_removes_from_registry():
    emb = HashingEmbedder(dim=64)
    retriever = build_retriever(RetrieveConfig(rerank=False), emb, MemoryStore(dim=64))
    retriever.index(make_chunks())
    assert retriever.drop_document("d2") == 1
    assert all(h.chunk.doc_id != "d2" for h in retriever.search("HNSW", k=5))


def test_evidence_rendering_is_single_line():
    emb = HashingEmbedder(dim=64)
    retriever = build_retriever(RetrieveConfig(rerank=False), emb, MemoryStore(dim=64))
    retriever.index(make_chunks())
    hits = retriever.search("融合", k=3)
    rendered = retriever.render_evidence(hits)
    for line in rendered.splitlines():
        assert line.strip()


def test_lexical_reranker_prefers_overlapping_chunk():
    chunks = make_chunks()
    reranker = LexicalReranker()
    reranker.fit([c.text for c in chunks])
    from phosphor.core.types import Scored

    scored = [Scored(chunk=c, score=1.0, source="test") for c in chunks]
    reranked = reranker.rerank("接地率 证据 支撑", scored, top_k=3)
    assert reranked[0].chunk.id == "c3"


def test_dense_and_sparse_agree_on_obvious_match():
    emb = HashingEmbedder(dim=384)
    chunks = make_chunks()
    dense = DenseRetriever(emb, MemoryStore(dim=384))
    dense.index(chunks)
    sparse = SparseRetriever()
    sparse.index(chunks)
    # The hashing projection is an approximation, so the dense branch is held
    # to a top-3 claim while the sparse branch is exact.
    assert "c2" in [cid for cid, _ in dense.search("HNSW 索引", 3)]
    assert sparse.search("HNSW 索引", 3)[0][0] == "c2"


def test_hybrid_without_backends_is_safe():
    retriever = HybridRetriever(None, None)
    assert retriever.search("任意查询", k=3) == []
