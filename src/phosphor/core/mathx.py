"""Numeric helpers with a pure python fallback.

numpy is used when present (fast path) but is never required: the stack must
boot and pass its own verification on a bare CPython install.
"""

from __future__ import annotations

import math

try:  # pragma: no cover - depends on environment
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

HAS_NUMPY = _np is not None


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} != {len(b)}")
    if HAS_NUMPY:
        va = _np.asarray(a, dtype=_np.float32)
        vb = _np.asarray(b, dtype=_np.float32)
        na = float(_np.linalg.norm(va))
        nb = float(_np.linalg.norm(vb))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(_np.dot(va, vb) / (na * nb))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return list(vec)
    return [x / norm for x in vec]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], pct: float) -> float:
    """Deterministic percentile; no interpolation, no numpy dependency."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = int(round((len(ordered) - 1) * pct))
    return ordered[max(0, min(len(ordered) - 1, idx))]


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation that is 0.0 for constant input.

    Returning a tie-broken rank for a constant vector yields a meaningless
    +1.0 correlation and makes a useless component look perfectly calibrated.
    """
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    if len(set(xs)) <= 1 or len(set(ys)) <= 1:
        return 0.0
    rx = _ranks(xs)
    ry = _ranks(ys)
    n = len(xs)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry, strict=False))
    return 1.0 - (6.0 * d2) / (n * (n * n - 1))


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks
