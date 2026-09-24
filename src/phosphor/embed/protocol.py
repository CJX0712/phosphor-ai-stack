"""Embedding contract.

Modules depend on this Protocol only; the concrete backend is injected at
assembly time in phosphor.pipeline.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch; output order must match input order."""
        ...

    def embed_one(self, text: str) -> list[float]:
        ...
