"""Exact cosine index in pure python.

Exact search is the reference implementation: every approximate backend
(FAISS IVF/HNSW) is validated against it in tests, which is what makes the
"recall did not regress" claim meaningful instead of self reported.
"""

from __future__ import annotations

import threading

from ..core.mathx import cosine, l2_normalize
from .protocol import VectorStore


class MemoryStore:
    name = "memory"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self._vectors: dict[str, list[float]] = {}
        self._meta: dict[str, dict] = {}
        self._lock = threading.RLock()

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metas: list[dict] | None = None,
    ) -> None:
        if len(ids) != len(vectors):
            raise ValueError("ids/vectors length mismatch")
        metas = metas or [{} for _ in ids]
        with self._lock:
            for cid, vec, meta in zip(ids, vectors, metas):
                if self.dim == 0:
                    self.dim = len(vec)
                if len(vec) != self.dim:
                    raise ValueError(f"dimension mismatch: {len(vec)} != {self.dim}")
                self._vectors[cid] = l2_normalize(list(vec))
                self._meta[cid] = dict(meta)

    def search(self, query: list[float], k: int = 10) -> list[tuple[str, float]]:
        if not self._vectors:
            return []
        q = l2_normalize(list(query))
        scored = [(cid, cosine(q, vec)) for cid, vec in self._vectors.items()]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[: max(0, k)]

    def delete(self, ids: list[str]) -> int:
        removed = 0
        with self._lock:
            for cid in ids:
                if cid in self._vectors:
                    del self._vectors[cid]
                    self._meta.pop(cid, None)
                    removed += 1
        return removed

    def drop_document(self, doc_id: str) -> int:
        with self._lock:
            targets = [
                cid
                for cid, meta in self._meta.items()
                if meta.get("doc_id") == doc_id
            ]
        return self.delete(targets)

    def get(self, cid: str) -> list[float] | None:
        return self._vectors.get(cid)

    def stats(self) -> dict:
        return {
            "provider": self.name,
            "dim": self.dim,
            "vectors": len(self._vectors),
            "documents": len({m.get("doc_id") for m in self._meta.values()}),
        }
