"""Conversation memory contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..core.types import Message


@runtime_checkable
class Memory(Protocol):
    name: str

    def append(self, session_id: str, message: Message) -> None:
        ...

    def recent(self, session_id: str, limit: int = 8) -> list[Message]:
        ...

    def clear(self, session_id: str) -> int:
        ...

    def sessions(self) -> list[str]:
        ...
