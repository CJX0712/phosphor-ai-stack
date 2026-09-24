"""In process event bus.

The only coupling between modules that are not allowed to know about each
other: observability, metrics and tracing subscribe, everything else emits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

Listener = Callable[["Event"], None]


@dataclass(slots=True)
class Event:
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    trace_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ts": self.ts,
            "trace_id": self.trace_id,
            "payload": self.payload,
        }


class EventBus:
    """Synchronous fan-out bus with wildcard subscriptions."""

    def __init__(self) -> None:
        self._exact: dict[str, list[Listener]] = {}
        self._wild: list[Listener] = []
        self._history: list[Event] = []
        self._history_limit = 512

    def subscribe(self, name: str, fn: Listener) -> Callable[[], None]:
        self._exact.setdefault(name, []).append(fn)

        def _unsub() -> None:
            self._exact[name].remove(fn)

        return _unsub

    def subscribe_all(self, fn: Listener) -> Callable[[], None]:
        self._wild.append(fn)

        def _unsub() -> None:
            self._wild.remove(fn)

        return _unsub

    def emit(self, name: str, **payload: Any) -> Event:
        trace_id = str(payload.get("trace_id", "") or "")
        event = Event(name=name, payload=payload, trace_id=trace_id)
        self._history.append(event)
        if len(self._history) > self._history_limit:
            del self._history[: len(self._history) - self._history_limit]
        for fn in self._exact.get(name, ()):
            fn(event)
        for fn in tuple(self._wild):
            fn(event)
        return event

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        return [e.as_dict() for e in self._history[-limit:]]

    def clear(self) -> None:
        self._history.clear()


_BUS = EventBus()


def bus() -> EventBus:
    return _BUS
