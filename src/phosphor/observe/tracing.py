"""Tracing: a span tree derived from the event bus.

Events carry a trace_id; the collector builds per-trace timelines so a single
answer can be replayed step by step (retrieval -> tool -> agent -> orchestrator)
without an external tracing backend.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import RLock

from ..core.events import Event, bus


@dataclass(slots=True)
class Span:
    name: str
    trace_id: str
    started: float
    ended: float | None = None
    attributes: dict[str, object] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        end = self.ended or time.time()
        return round((end - self.started) * 1000, 2)


class Tracer:
    def __init__(self, max_traces: int = 256) -> None:
        self._traces: dict[str, list[Span]] = {}
        self._started: dict[str, float] = {}
        self._lock = RLock()
        self.max_traces = max_traces
        bus().subscribe_all(self._on_event)

    def _on_event(self, event: Event) -> None:
        if not event.trace_id:
            return
        name = event.name
        with self._lock:
            if name.endswith(".start"):
                self._started[(event.trace_id, name)] = event.ts
                return
            started = self._started.pop((event.trace_id, name[:-6]), event.ts)
            spans = self._traces.setdefault(event.trace_id, [])
            spans.append(
                Span(
                    name=name,
                    trace_id=event.trace_id,
                    started=started,
                    ended=event.ts,
                    attributes={k: v for k, v in event.payload.items() if k != "trace_id"},
                )
            )
            if len(self._traces) > self.max_traces:
                for key in list(self._traces)[: len(self._traces) - self.max_traces]:
                    self._traces.pop(key, None)

    def note(self, trace_id: str, name: str, **attributes: object) -> None:
        with self._lock:
            self._traces.setdefault(trace_id, []).append(
                Span(name=name, trace_id=trace_id, started=time.time(), ended=time.time(),
                     attributes=dict(attributes))
            )

    def trace(self, trace_id: str) -> list[dict]:
        with self._lock:
            spans = self._traces.get(trace_id, [])
        return [
            {
                "name": s.name,
                "duration_ms": s.duration_ms,
                "attributes": s.attributes,
            }
            for s in spans
        ]

    def traces(self) -> list[str]:
        with self._lock:
            return sorted(self._traces)


TRACER = Tracer()


def tracer() -> Tracer:
    return TRACER
