from phosphor.core.mathx import cosine
from phosphor.embed import HashingEmbedder, build_embedder
from phosphor.core.config import EmbedConfig


def test_hashing_dimension_and_normalisation():
    emb = HashingEmbedder(dim=64)
    vec = emb.embed_one("向量检索测试")
    assert len(vec) == 64
    assert abs(sum(x * x for x in vec) - 1.0) < 1e-6 or vec == [0.0] * 64


def test_hashing_is_deterministic():
    emb = HashingEmbedder(dim=32)
    assert emb.embed_one("同样的输入") == emb.embed_one("同样的输入")


def test_similar_text_is_closer_than_unrelated():
    emb = HashingEmbedder(dim=256)
    a = emb.embed_one("混合检索使用 reciprocal rank fusion 融合两路召回")
    b = emb.embed_one("混合检索使用 RRF 融合策略召回结果")
    c = emb.embed_one("今天中午吃什么完全无关的话题")
    assert cosine(a, b) > cosine(a, c)


def test_batch_matches_single():
    emb = HashingEmbedder(dim=48)
    texts = ["第一条", "第二条内容", "第三条不同的内容"]
    batch = emb.embed(texts)
    assert batch == [emb.embed_one(t) for t in texts]


def test_factory_default_is_hashing():
    emb = build_embedder(EmbedConfig())
    assert isinstance(emb, HashingEmbedder)
