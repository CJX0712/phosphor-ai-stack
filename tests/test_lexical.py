from phosphor.lexical import BM25, expand_query, expand_terms


def test_idf_is_never_negative():
    bm = BM25()
    bm.index(["a", "b", "c"], ["检索 融合", "融合 索引", "部署 延迟"])
    for term in ["检索", "融合", "部署", "不存在的词"]:
        assert bm.idf(term) >= 0.0


def test_target_document_ranks_first_on_three_documents():
    """Two document corpora make IDF degenerate; three is the minimum."""
    bm = BM25()
    bm.index(
        ["doc-a", "doc-b", "doc-c"],
        [
            "混合检索使用 reciprocal rank fusion 融合两路召回",
            "向量索引 HNSW 适合 CPU 部署",
            "评估指标包含文档级命中率与 MRR",
        ],
    )
    hits = bm.search("reciprocal rank fusion 融合", k=3)
    assert hits, "bm25 returned nothing"
    assert hits[0][0] == "doc-a"


def test_query_without_overlap_returns_nothing():
    bm = BM25()
    bm.index(["a", "b", "c"], ["检索 融合", "索引 部署", "评估 指标"])
    assert bm.search("完全不相关的查询词", k=3) == []


def test_cross_lingual_expansion():
    expanded = expand_query("how does reranking work")
    assert "重排" in expanded or "重排序" in expanded


def test_expansion_is_empty_for_unknown_terms():
    assert expand_terms("今天天气不错") == []
