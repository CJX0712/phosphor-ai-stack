from phosphor.core.mathx import cosine, spearman
from phosphor.core.text import (
    cosine_sets,
    normalize,
    split_sentences,
    strip_citations,
    tokenize,
    word_set,
)


def test_tokenize_bilingual():
    tokens = tokenize("检索增强 generation 与 FAISS 索引")
    assert "检索" in tokens
    assert "generation" in tokens
    assert "faiss" in tokens


def test_tokenize_cjk_bigram():
    tokens = tokenize("向量数据库")
    assert "向量" in tokens and "量数" in tokens


def test_sentence_split_keeps_version_numbers():
    sentences = split_sentences("Python 3.10 发布了。这是第二句。")
    assert any("3.10" in s for s in sentences)
    assert len(sentences) == 2


def test_strip_citations_removes_provenance_only():
    raw = "答案是 42 (source: doc#3) [id#c_12]"
    cleaned = strip_citations(raw)
    assert "42" in cleaned
    assert "doc#3" not in cleaned


def test_cosine_sets_length_insensitive():
    short = word_set("RRF 融合")
    long_q = word_set("请问 Phosphor 在混合检索里到底使用了哪一种融合策略呢")
    hit = word_set("Phosphor 使用 RRF 融合策略")
    assert cosine_sets(long_q, hit) > 0
    assert cosine_sets(short, hit) > 0


def test_cosine_math():
    assert abs(cosine([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9
    assert abs(cosine([1.0, 0.0], [0.0, 1.0])) < 1e-9
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_spearman_constant_input_is_zero():
    assert spearman([1.0, 1.0, 1.0], [3.0, 1.0, 2.0]) == 0.0
    assert spearman([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_normalize_collapses_whitespace():
    assert normalize("a    b\n\n\nc") == "a b\n\nc"
