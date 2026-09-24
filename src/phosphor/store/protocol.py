"""Vector store contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class VectorStore(Protocol):
    dim: int

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metas: list[dict] | None = None,
    ) -> None:
        ...

    def search(self, query: list[float], k: int = 10) -> list[tuple[str, float]]:
        """Return (id, score) descending; score is higher-is-better cosine."""
        ...

    def delete(self, ids: list[str]) -> int:
        ...

    def drop_document(self, doc_id: str) -> int:
        """Remove every vector belonging to a document, return removed count."""
        ...

    def stats(self) -> dict:
        ...
