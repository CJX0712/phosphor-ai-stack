"""Memory assembly."""

from __future__ import annotations

from .inproc import InProcMemory
from .protocol import Memory


def build_memory(path: str = "", max_per_session: int = 64) -> Memory:
    return InProcMemory(path=path, max_per_session=max_per_session)
