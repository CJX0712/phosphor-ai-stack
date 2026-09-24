"""BM25 with Robertson IDF.

The textbook IDF ln(N/n) can be negative and vanishes entirely when a term
appears in exactly half the corpus; the epsilon floor used by common
implementations then distorts the ranking (measured: correct document ranked
last on a two document corpus). Robertson's smoothed form

    idf(t) = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))

is non-negative for every n(t) <= N, which is what makes small corpora behave.
"""

from __future__ import annotations

import math
from collections import Counter

from ..core.text import tokenize


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._ids: list[str] = []
        self._tf: list[Counter] = []
        self._len: list[int] = []
        self._df: Counter = Counter()
        self._avg_len = 0.0
        self._n = 0

    # -- indexing ---------------------------------------------------------
    def index(self, ids: list[str], texts: list[str]) -> None:
        self._ids = list(ids)
        self._tf = [Counter(tokenize(t)) for t in texts]
        self._len = [sum(c.values()) for c in self._tf]
        self._n = len(self._ids)
        self._df = Counter()
        for counter in self._tf:
            self._df.update(counter.keys())
        total = sum(self._len)
        self._avg_len = total / self._n if self._n else 0.0

    def clear(self) -> None:
        self._ids = []
        self._tf = []
        self._len = []
        self._df = Counter()
        self._n = 0
        self._avg_len = 0.0

    @property
    def size(self) -> int:
        return self._n

    # -- scoring ----------------------------------------------------------
    def idf(self, term: str) -> float:
        if self._n == 0:
            return 0.0
        n_t = self._df.get(term, 0)
        return math.log(1.0 + (self._n - n_t + 0.5) / (n_t + 0.5))

    def score_terms(self, terms: list[str], doc_idx: int) -> float:
        tf = self._tf[doc_idx]
        dl = self._len[doc_idx] or 1
        score = 0.0
        for term in terms:
            f = tf.get(term, 0)
            if not f:
                continue
            denom = f + self.k1 * (1.0 - self.b + self.b * dl / (self._avg_len or 1.0))
            score += self.idf(term) * (f * (self.k1 + 1.0)) / denom
        return score

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        terms = tokenize(query)
        if not terms or self._n == 0:
            return []
        scored: list[tuple[str, float]] = []
        for i, cid in enumerate(self._ids):
            score = self.score_terms(terms, i)
            if score > 0:
                scored.append((cid, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[: max(0, k)]

    def stats(self) -> dict:
        return {
            "documents": self._n,
            "terms": len(self._df),
            "avg_len": round(self._avg_len, 2),
        }
