"""Hybrid retrieval with reciprocal rank fusion.

RRF is used instead of score normalisation because dense cosine and BM25 live
on incomparable scales; rank fusion is scale free and behaves the same whether
the dense backend is a hashing projection or a real sentence transformer.
"""

from __future__ import annotations

import time

from ..core.config import RetrieveConfig
from ..core.events import bus
from ..core.types import Chunk, Scored
from .dense import DenseRetriever
from .protocol import Reranker, Retriever
from .sparse import SparseRetriever


class HybridRetriever:
    name = "hybrid"

    def __init__(
        self,
        dense: DenseRetriever | None,
        sparse: SparseRetriever | None,
        cfg: RetrieveConfig | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.cfg = cfg or RetrieveConfig()
        self.reranker = reranker
        self._chunk_of: dict[str, Chunk] = {}

    # -- indexing ---------------------------------------------------------
    def index(self, chunks: list[Chunk], vectors: list[list[float]] | None = None) -> None:
        if not chunks:
            return
        for chunk in chunks:
            self._chunk_of[chunk.id] = chunk
        if self.dense is not None:
            self.dense.index(chunks, vectors)
        if self.sparse is not None:
            self.sparse.index(list(self._chunk_of.values()))

    def drop_document(self, doc_id: str) -> int:
        """Remove a document everywhere, including the local chunk registry."""
        removed = 0
        for cid in [c.id for c in self._chunk_of.values() if c.doc_id == doc_id]:
            self._chunk_of.pop(cid, None)
            removed += 1
        if self.dense is not None:
            self.dense.drop_document(doc_id)
        if self.sparse is not None:
            self.sparse.index(list(self._chunk_of.values()))
        return removed

    # -- search -----------------------------------------------------------
    def search(self, query: str, k: int | None = None, trace_id: str = "") -> list[Scored]:
        k = k or self.cfg.top_k
        pool = max(k * self.cfg.candidate_multiplier, k + 4)
        started = time.perf_counter()

        dense_hits: list[tuple[str, float]] = []
        sparse_hits: list[tuple[str, float]] = []
        if self.dense is not None:
            dense_hits = self.dense.search(query, pool)
        if self.sparse is not None:
            sparse_hits = self.sparse.search(query, pool)

        fused = self._fuse(dense_hits, sparse_hits, pool)
        if self.reranker is not None and fused:
            fused = self.reranker.rerank(query, fused, top_k=max(k, len(fused) // 2 + 1))
        fused = fused[:k]

        elapsed = int((time.perf_counter() - started) * 1000)
        bus().emit(
            "retrieval.search",
            trace_id=trace_id,
            query=query,
            dense=len(dense_hits),
            sparse=len(sparse_hits),
            fused=len(fused),
            elapsed_ms=elapsed,
        )
        return fused

    def _fuse(
        self,
        dense_hits: list[tuple[str, float]],
        sparse_hits: list[tuple[str, float]],
        pool: int,
    ) -> list[Scored]:
        rrf_k = max(1, self.cfg.rrf_k)
        scores: dict[str, float] = {}
        detail: dict[str, dict[str, float]] = {}
        for rank, (cid, score) in enumerate(dense_hits, start=1):
            scores[cid] = scores.get(cid, 0.0) + self.cfg.dense_weight / (rrf_k + rank)
            detail.setdefault(cid, {})["dense"] = float(score)
            detail[cid]["dense_rank"] = float(rank)
        for rank, (cid, score) in enumerate(sparse_hits, start=1):
            scores[cid] = scores.get(cid, 0.0) + self.cfg.sparse_weight / (rrf_k + rank)
            detail.setdefault(cid, {})["sparse"] = float(score)
            detail[cid]["sparse_rank"] = float(rank)

        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:pool]
        out: list[Scored] = []
        for cid, score in ordered:
            chunk = self._chunk_of.get(cid)
            if chunk is None:
                continue
            out.append(
                Scored(chunk=chunk, score=score, source="rrf", detail=dict(detail.get(cid, {})))
            )
        return out

    # -- introspection ----------------------------------------------------
    def stats(self) -> dict:
        return {
            "provider": self.name,
            "chunks": len(self._chunk_of),
            "dense": self.dense.stats() if self.dense else None,
            "sparse": self.sparse.stats() if self.sparse else None,
            "reranker": getattr(self.reranker, "name", None),
        }

    def chunks(self) -> list[Chunk]:
        return list(self._chunk_of.values())

    def render_evidence(self, scored: list[Scored], max_chars: int | None = None) -> str:
        """Render evidence one chunk per line.

        Multi line rendering is silently truncated by line based consumers:
        this is why the model used to see only the first line of every chunk.
        """
        limit = max_chars or self.cfg.max_evidence_chars
        lines: list[str] = []
        used = 0
        for item in scored:
            line = item.chunk.render()
            if used + len(line) > limit and lines:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)


def _self_check() -> None:
    assert isinstance(HybridRetriever(None, None), Retriever)


_self_check()
