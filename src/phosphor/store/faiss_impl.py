"""FAISS backed store (optional).

Uses IndexFlatIP with pre-normalised vectors, so the inner product is exactly
cosine similarity and scores stay comparable with MemoryStore.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from ..core.errors import StorageError
from ..core.mathx import l2_normalize


class FaissStore:
    name = "faiss"

    def __init__(self, dim: int = 384, path: str = "") -> None:
        try:
            import faiss  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise StorageError("faiss-cpu not installed") from exc
        self._faiss: Any = faiss
        self.dim = dim
        self.path = path
        self._index: Any = faiss.IndexFlatIP(dim)
        self._ids: list[str] = []
        self._meta: dict[str, dict] = {}
        self._lock = threading.RLock()

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metas: list[dict] | None = None,
    ) -> None:
        if not ids:
            return
        metas = metas or [{} for _ in ids]
        with self._lock:
            matrix = [l2_normalize(list(v)) for v in vectors]
            known = set(self._ids)
            new_pairs = [
                (cid, vec, meta)
                for cid, vec, meta in zip(ids, matrix, metas, strict=False)
                if cid not in known
            ]
            if not new_pairs:
                for cid, meta in zip(ids, metas, strict=False):
                    self._meta[cid] = dict(meta)
                return
            import numpy as np  # type: ignore

            arr = np.asarray([p[1] for p in new_pairs], dtype="float32")
            self._index.add(arr)
            for cid, _vec, meta in new_pairs:
                self._ids.append(cid)
                self._meta[cid] = dict(meta)

    def search(self, query: list[float], k: int = 10) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        import numpy as np  # type: ignore

        vec = np.asarray([l2_normalize(list(query))], dtype="float32")
        scores, idxs = self._index.search(vec, min(max(1, k), len(self._ids)))
        out: list[tuple[str, float]] = []
        for score, i in zip(scores[0], idxs[0], strict=False):
            if i < 0 or i >= len(self._ids):
                continue
            out.append((self._ids[int(i)], float(score)))
        return out

    def delete(self, ids: list[str]) -> int:
        """FAISS has no cheap delete; rebuild without the removed ids."""
        with self._lock:
            keep = [cid for cid in self._ids if cid not in set(ids)]
            if len(keep) == len(self._ids):
                return 0
            removed = len(self._ids) - len(keep)
            import numpy as np  # type: ignore

            matrix = np.asarray(
                [self._index.reconstruct(i) for i in range(len(self._ids))
                 if self._ids[i] in keep],
                dtype="float32",
            ) if keep else np.zeros((0, self.dim), dtype="float32")
            self._index = self._faiss.IndexFlatIP(self.dim)
            if len(matrix):
                self._index.add(matrix)
            for cid in ids:
                self._meta.pop(cid, None)
            self._ids = keep
            return removed

    def drop_document(self, doc_id: str) -> int:
        targets = [cid for cid, m in self._meta.items() if m.get("doc_id") == doc_id]
        return self.delete(targets)

    def save(self) -> None:
        if not self.path:
            return
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(p))

    def stats(self) -> dict:
        return {
            "provider": self.name,
            "dim": self.dim,
            "vectors": len(self._ids),
            "documents": len({m.get("doc_id") for m in self._meta.values()}),
        }
