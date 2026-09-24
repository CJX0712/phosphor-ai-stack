"""Retrieval contract shared by dense, sparse and hybrid backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..core.types import Chunk, Scored


@runtime_checkable
class Retriever(Protocol):
    def index(self, chunks: list[Chunk], vectors: list[list[float]] | None = None) -> None:
        ...

    def drop_document(self, doc_id: str) -> int:
        ...

    def search(self, query: str, k: int = 6) -> list[Scored]:
        ...

    def stats(self) -> dict:
        ...


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(self, query: str, scored: list[Scored], top_k: int = 6) -> list[Scored]:
        ...
