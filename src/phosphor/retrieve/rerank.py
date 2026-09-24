"""Reranking.

Default backend is a lexical reranker (IDF weighted term overlap with a cosine
style length penalty) - a real reranker, not a passthrough, so the rerank stage
is exercised by default and can be measured. The cross-encoder backend is used
when fastembed is available.
"""

from __future__ import annotations

import math

from ..core.text import tokenize
from ..core.types import Scored
from .protocol import Reranker


class LexicalReranker:
    """Score = IDF weighted overlap / sqrt(len) with a phrase bonus."""

    name = "lexical"

    def __init__(self, phrase_bonus: float = 1.35) -> None:
        self.phrase_bonus = phrase_bonus
        self._df: dict[str, int] = {}
        self._n = 0

    def fit(self, texts: list[str]) -> None:
        self._df = {}
        self._n = len(texts)
        for text in texts:
            for term in set(tokenize(text)):
                self._df[term] = self._df.get(term, 0) + 1

    def idf(self, term: str) -> float:
        if not self._n:
            return 1.0
        n_t = self._df.get(term, 0)
        return math.log(1.0 + (self._n - n_t + 0.5) / (n_t + 0.5))

    def score(self, query: str, text: str) -> float:
        q_terms = tokenize(query)
        if not q_terms:
            return 0.0
        d_terms = tokenize(text)
        if not d_terms:
            return 0.0
        d_counts: dict[str, int] = {}
        for term in d_terms:
            d_counts[term] = d_counts.get(term, 0) + 1
        total = 0.0
        for term in set(q_terms):
            f = d_counts.get(term, 0)
            if f:
                total += self.idf(term) * (1.0 + math.log(f))
        norm = math.sqrt(len(d_terms))
        base = total / norm if norm else 0.0
        lowered = text.lower()
        if query.strip().lower() in lowered or _longest_phrase(query) in lowered:
            base *= self.phrase_bonus
        return base

    def rerank(self, query: str, scored: list[Scored], top_k: int = 6) -> list[Scored]:
        if not scored:
            return []
        if not self._n:
            self.fit([s.chunk.text for s in scored])
        out: list[Scored] = []
        for item in scored:
            detail = dict(item.detail)
            detail["rerank"] = self.score(query, item.chunk.text)
            out.append(
                Scored(chunk=item.chunk, score=detail["rerank"], source=item.source, detail=detail)
            )
        out.sort(key=lambda s: (-s.score, s.chunk.id))
        return out[: max(0, top_k)]


def _longest_phrase(query: str, min_len: int = 4) -> str:
    parts = [p for p in query.split() if len(p) >= min_len]
    return max(parts, key=len).lower() if parts else ""


class CrossEncoderReranker:
    """fastembed cross-encoder (bge-reranker-base) when installed."""

    name = "cross-encoder"

    def __init__(self, model: str = "BAAI/bge-reranker-base") -> None:
        self.model_name = model
        self._model = None

    def _ensure(self):
        if self._model is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("fastembed reranker unavailable") from exc
            self._model = TextCrossEncoder(model_name=self.model_name)
        return self._model

    def warmup(self) -> None:
        """Force model download/initialisation so failures surface at build time."""
        model = self._ensure()
        list(model.rerank("warmup", ["warmup passage", "another passage"]))

    def rerank(self, query: str, scored: list[Scored], top_k: int = 6) -> list[Scored]:
        if not scored:
            return []
        model = self._ensure()
        scores = list(model.rerank(query, [s.chunk.text for s in scored]))
        out: list[Scored] = []
        for item, value in zip(scored, scores):
            detail = dict(item.detail)
            detail["rerank"] = float(value)
            out.append(Scored(chunk=item.chunk, score=float(value), source=item.source, detail=detail))
        out.sort(key=lambda s: (-s.score, s.chunk.id))
        return out[: max(0, top_k)]


def build_reranker(
    enabled: bool = True,
    model: str = "BAAI/bge-reranker-base",
    provider: str = "auto",
) -> Reranker | None:
    """Pick a reranker.

    provider=auto attempts the cross-encoder, forces the download with a warmup
    call and falls back to the lexical reranker when the model is unavailable
    (offline machine, blocked registry). A cold start failure must never break
    the retrieval path, so the fallback is silent by design.
    """
    if not enabled:
        return None
    if provider == "lexical":
        return LexicalReranker()
    if provider == "cross-encoder":
        return CrossEncoderReranker(model)
    if _PROBE["failed"]:
        # Never retry a failed download more than once per process.
        return LexicalReranker()
    try:
        reranker = CrossEncoderReranker(model)
        reranker.warmup()
        return reranker
    except Exception:  # noqa: BLE001 - offline fallback is intentional
        _PROBE["failed"] = True
        return LexicalReranker()


_PROBE: dict[str, bool] = {"failed": False}
