"""Structure aware chunking.

Two failure modes this module exists to prevent:

1. Heading loss - a chunk must know which section it came from, otherwise
   retrieved evidence has no context and the model answers from a fragment.
2. Heading/body fusion - "## Title" immediately followed by body text is one
   paragraph under a naive blank line split, and the body gets skipped as if
   it were the heading itself (observed chunk count: 0).
"""

from __future__ import annotations

import re

from ..core.config import ChunkConfig
from ..core.ids import stable_id
from ..core.types import Chunk, Document

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_NUM_HEADING_RE = re.compile(r"^(\d+(?:\.\d+){0,4})[\s、]+(\S.{0,80})$")
_CHAPTER_RE = re.compile(r"^(第[一二三四五六七八九十百千零\d]+[章节篇部])\s*(\S.{0,80})?$")

_SENT_END = "。！？；!?;"
_ELLIPSIS = "…"


def _classify(line: str) -> tuple[int, str] | None:
    """Return (level, title) when the line is a heading, else None."""
    text = line.strip()
    if not text:
        return None
    m = _MD_HEADING_RE.match(text)
    if m:
        return len(m.group(1)), m.group(2).strip()
    m = _CHAPTER_RE.match(text)
    if m:
        title = m.group(1)
        if m.group(2):
            title = f"{title} {m.group(2).strip()}"
        return 2, title
    m = _NUM_HEADING_RE.match(text)
    if m and len(text) <= 90:
        depth = m.group(1).count(".") + 1
        return min(6, depth + 1), f"{m.group(1)} {m.group(2).strip()}"
    if len(text) <= 40 and text.endswith(("：", ":")) and not text.endswith("::"):
        return 6, text.rstrip("：:")
    return None


def _sliding_windows(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    windows: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + size)
        if end < n:
            # Prefer to cut on a sentence boundary inside the last third.
            search_from = start + int(size * 0.6)
            cut = -1
            for i in range(end, search_from, -1):
                if text[i - 1] in _SENT_END:
                    cut = i
                    break
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            windows.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return windows


def split_text(
    text: str,
    doc_id: str,
    cfg: ChunkConfig | None = None,
    heading_path: tuple[str, ...] = (),
) -> list[Chunk]:
    cfg = cfg or ChunkConfig()
    chunks: list[Chunk] = []
    stack: list[tuple[int, str]] = list(
        (i + 1, h) for i, h in enumerate(heading_path)
    )
    para_lines: list[str] = []
    para_path: tuple[str, ...] = tuple(h for _, h in stack)
    pos = 0

    def heading_tuple() -> tuple[str, ...]:
        return tuple(h for _, h in stack)

    def flush() -> None:
        nonlocal para_lines, para_path
        if not para_lines:
            return
        body = "\n".join(para_lines).strip()
        para_lines = []
        if not body:
            return
        for piece in _sliding_windows(body, cfg.size, cfg.overlap):
            if len(piece) < cfg.min_size and chunks:
                # Merge undersized tail into the previous chunk when adjacent.
                prev = chunks[-1]
                if prev.heading_path == para_path and len(prev.text) + len(piece) <= cfg.size:
                    prev.text = f"{prev.text} {piece}".strip()
                    continue
            chunks.append(
                Chunk(
                    id=stable_id(doc_id, str(len(chunks)), piece[:96], prefix="c_"),
                    doc_id=doc_id,
                    text=piece,
                    order=len(chunks),
                    start=pos,
                    end=pos + len(piece),
                    heading_path=para_path,
                )
            )
        para_path = heading_tuple()

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        start_pos = pos
        pos += len(raw_line) + 1
        if not line.strip():
            flush()
            continue
        heading = _classify(line) if cfg.respect_headings else None
        if heading is not None:
            flush()
            level, title = heading
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            para_path = heading_tuple()
            continue
        if not para_lines:
            para_path = heading_tuple()
        para_lines.append(line)
        del start_pos
    flush()

    if not chunks:
        body = text.strip()
        if body:
            chunks.append(
                Chunk(
                    id=stable_id(doc_id, "0", body[:96], prefix="c_"),
                    doc_id=doc_id,
                    text=body,
                    order=0,
                    start=0,
                    end=len(body),
                    heading_path=tuple(heading_path),
                )
            )
    return chunks


def split_document(doc: Document, cfg: ChunkConfig | None = None) -> list[Chunk]:
    return split_text(doc.text, doc.id, cfg)
