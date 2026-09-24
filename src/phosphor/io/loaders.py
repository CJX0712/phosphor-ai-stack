"""Document loaders: bytes on disk become core.Document.

Every loader is dependency free except the PDF one, which degrades to a clear
error instead of crashing the ingest pipeline when pypdf is absent.
"""

from __future__ import annotations

import io as _io
import re
from pathlib import Path
from typing import Callable

from ..core.errors import IngestError
from ..core.ids import stable_id
from ..core.text import normalize
from ..core.types import Document

Loader = Callable[[Path], str]

_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".json", ".html", ".htm", ".pdf"}

_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
_HTML_RE = re.compile(r"<[^>]+>")


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise IngestError(f"cannot decode {path}")


def read_markdown(path: Path) -> str:
    return read_text(path)


def read_html(path: Path) -> str:
    text = read_text(path)
    text = _TAG_RE.sub(" ", text)
    text = _HTML_RE.sub("\n", text)
    return normalize(text)


def read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise IngestError(
            "pypdf is required for PDF ingest; install requirements-ai.txt"
        ) from exc
    try:
        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:
        raise IngestError(f"pdf parse failed: {path.name}") from exc
    return "\n\n".join(p for p in pages if p.strip())


LOADERS: dict[str, Loader] = {
    ".txt": read_text,
    ".md": read_markdown,
    ".markdown": read_markdown,
    ".rst": read_text,
    ".log": read_text,
    ".csv": read_text,
    ".json": read_text,
    ".html": read_html,
    ".htm": read_html,
    ".pdf": read_pdf,
}


def load_path(path: str | Path, doc_id: str | None = None, source: str | None = None) -> Document:
    p = Path(path)
    if not p.exists():
        raise IngestError(f"path not found: {p}")
    suffix = p.suffix.lower()
    loader = LOADERS.get(suffix)
    if loader is None:
        if suffix not in _SUFFIXES:
            loader = read_text
        else:
            raise IngestError(f"unsupported suffix: {suffix}")
    text = normalize(loader(p))
    if not text:
        raise IngestError(f"empty document: {p.name}")
    did = doc_id or stable_id(p.name, text[:512], prefix="doc_")
    return Document(
        id=did,
        title=p.stem,
        text=text,
        source=source or str(p),
        meta={"suffix": suffix, "bytes": str(p.stat().st_size)},
    )


def load_text(title: str, text: str, doc_id: str | None = None, source: str = "") -> Document:
    clean = normalize(text)
    if not clean:
        raise IngestError("empty document text")
    did = doc_id or stable_id(title, clean[:512], prefix="doc_")
    return Document(id=did, title=title, text=clean, source=source or "inline", meta={"inline": True})


def load_bytes(name: str, data: bytes, doc_id: str | None = None) -> Document:
    return load_text(name, data.decode("utf-8", errors="replace"), doc_id=doc_id, source=name)


def load_dir(path: str | Path, recursive: bool = True) -> list[Document]:
    p = Path(path)
    if not p.is_dir():
        raise IngestError(f"not a directory: {p}")
    it = p.rglob("*") if recursive else p.glob("*")
    docs: list[Document] = []
    for child in sorted(it):
        if child.is_file() and child.suffix.lower() in _SUFFIXES:
            try:
                docs.append(load_path(child))
            except IngestError:
                continue
    return docs
