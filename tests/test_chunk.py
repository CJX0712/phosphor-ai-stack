from phosphor.chunk import split_text
from phosphor.core.config import ChunkConfig


def test_heading_is_inherited_by_body():
    doc = "# 第一章\n这是第一章的内容。" + "补充内容。" * 30 + "\n\n## 小节\n小节内容。"
    chunks = split_text(doc, "d1", ChunkConfig(size=120, overlap=20))
    assert chunks, "no chunks produced"
    body = [c for c in chunks if "第一章的内容" in c.text]
    assert body, "body text was dropped"
    assert any("第一章" in h for h in body[0].heading_path)


def test_heading_line_does_not_swallow_body():
    doc = "## 标题\n紧随其后的正文。"
    chunks = split_text(doc, "d2", ChunkConfig(size=200, overlap=0))
    assert any("紧随其后的正文" in c.text for c in chunks)


def test_long_text_produces_multiple_chunks():
    text = "这是用于分块的中文段落。" * 100
    chunks = split_text(text, "d3", ChunkConfig(size=200, overlap=30))
    assert len(chunks) > 3


def test_non_uniform_paragraphs_vary_chunk_boundaries():
    """A periodic corpus makes boundaries always land on the same heading."""
    paragraphs = []
    for i in range(9):
        length = 40 + (i % 4) * 90
        paragraphs.append(f"## 章节{i}\n" + ("内容" * length))
    doc = "\n\n".join(paragraphs)
    chunks = split_text(doc, "d4", ChunkConfig(size=300, overlap=40))
    assert len(chunks) > len(paragraphs)
    assert len({tuple(c.heading_path) for c in chunks}) > 1


def test_chunk_ids_are_stable():
    text = "稳定的分块标识符测试。" * 20
    first = [c.id for c in split_text(text, "d5", ChunkConfig(size=120, overlap=10))]
    second = [c.id for c in split_text(text, "d5", ChunkConfig(size=120, overlap=10))]
    assert first == second


def test_empty_text_yields_no_chunks():
    assert split_text("", "d6") == []
