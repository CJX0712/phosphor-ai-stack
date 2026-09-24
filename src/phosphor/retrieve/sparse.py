"""Sparse retrieval: BM25 over the bilingual tokenizer plus glossary expansion."""

from __future__ import annotations

from ..core.types import Chunk, Scored
from ..lexical.bm25 import BM25
from ..lexical.expand import expand_query


class SparseRetriever:
    name = "sparse"

    def __init__(self, expand: bool = True) -> None:
        self.bm25 = BM25()
        self.expand = expand
        self._ids: list[str] = []
        self._texts: list[str] = []

    def index(self, chunks: list[Chunk], vectors: list[list[float]] | None = None) -> None:
        if not chunks:
            return
        self._ids = [c.id for c in chunks]
        self._texts = [f"{' '.join(c.heading_path)} {c.text}" for c in chunks]
        self.bm25.index(self._ids, self._texts)

    def drop_document(self, doc_id: str) -> int:
        return 0

    def search(self, query: str, k: int = 6) -> list[tuple[str, float]]:
        q = expand_query(query) if self.expand else query
        return self.bm25.search(q, k)

    def search_scored(self, query: str, k: int, chunk_of: dict[str, Chunk]) -> list[Scored]:
        out: list[Scored] = []
        for cid, score in self.search(query, k):
            chunk = chunk_of.get(cid)
            if chunk is None:
                continue
            out.append(Scored(chunk=chunk, score=score, source=self.name, detail={"bm25": score}))
        return out

    def stats(self) -> dict:
        base = self.bm25.stats()
        base["provider"] = self.name
        return base
