"""LLM contract.

Every backend exposes the same two methods so the agent, the orchestrator and
the API layer can be tested against a deterministic backend.
"""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

from ..core.types import Message


@runtime_checkable
class LLM(Protocol):
    name: str
    model: str

    def complete(self, messages: list[Message], **kwargs: object) -> str:
        ...

    def stream(self, messages: list[Message], **kwargs: object) -> Iterator[str]:
        ...
