"""In-process memory with optional JSON persistence.

Turn history is capped per session so the agent prompt stays bounded; the
window is the only thing that matters for grounding and the rest is archived.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from ..core.types import Message


class InProcMemory:
    name = "inproc"

    def __init__(self, path: str = "", max_per_session: int = 64) -> None:
        self.path = path
        self.max_per_session = max_per_session
        self._store: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.RLock()
        if path:
            self._load()

    def append(self, session_id: str, message: Message) -> None:
        with self._lock:
            bucket = self._store.setdefault(session_id, [])
            bucket.append({"role": message.role, "content": message.content, "name": message.name})
            if len(bucket) > self.max_per_session:
                del bucket[: len(bucket) - self.max_per_session]
            if self.path:
                self._save()

    def recent(self, session_id: str, limit: int = 8) -> list[Message]:
        with self._lock:
            bucket = self._store.get(session_id, [])
        return [
            Message(role=item["role"], content=item["content"], name=item.get("name", ""))
            for item in bucket[-max(0, limit) :]
        ]

    def clear(self, session_id: str) -> int:
        with self._lock:
            count = len(self._store.pop(session_id, []))
            if self.path:
                self._save()
            return count

    def sessions(self) -> list[str]:
        with self._lock:
            return sorted(self._store)

    def stats(self) -> dict:
        with self._lock:
            return {
                "provider": self.name,
                "sessions": len(self._store),
                "messages": sum(len(v) for v in self._store.values()),
            }

    # -- persistence ------------------------------------------------------
    def _save(self) -> None:
        if not self.path:
            return
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self._store, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load(self) -> None:
        p = Path(self.path)
        if not p.exists():
            return
        try:
            self._store = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._store = {}
