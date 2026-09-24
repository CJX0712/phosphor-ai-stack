"""Counters, gauges and latency histograms with no external dependency."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager

from ..core.mathx import percentile


class Metrics:
    def __init__(self) -> None:
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.RLock()
        self._started = time.time()

    def inc(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += value

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            bucket = self._samples[name]
            bucket.append(value)
            if len(bucket) > 1000:
                del bucket[: len(bucket) - 1000]

    @contextmanager
    def timer(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, (time.perf_counter() - started) * 1000)

    def snapshot(self) -> dict:
        with self._lock:
            latencies = {}
            for name, values in self._samples.items():
                latencies[name] = {
                    "count": len(values),
                    "p50_ms": round(percentile(values, 0.5), 2),
                    "p95_ms": round(percentile(values, 0.95), 2),
                    "max_ms": round(max(values), 2),
                }
            return {
                "uptime_s": round(time.time() - self._started, 1),
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "latency": latencies,
            }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._samples.clear()


METRICS = Metrics()


def metrics() -> Metrics:
    return METRICS
