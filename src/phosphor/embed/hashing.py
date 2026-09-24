"""Zero dependency embedder: signed hashing trick over bilingual tokens.

Why a hashing embedder is the default rather than a stub:

* it is a real implementation - cosine similarity between two texts behaves
  like lexical overlap, which is enough for hybrid retrieval to be exercised
  end to end on a bare interpreter with no model download;
* it is deterministic - byte identical across machines, so golden evaluations
  never drift because of float noise from a different ONNX runtime build.

The trade-off (no semantic generalisation) is documented in docs/SPEC.md and
the production path is one env var away.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter

from ..core.mathx import l2_normalize
from ..core.text import tokenize
from .protocol import Embedder


class HashingEmbedder:
    """Sign-consistent random projection, two probes per token."""

    name = "hashing"

    def __init__(self, dim: int = 384, probes: int = 2) -> None:
        self.dim = dim
        self.probes = probes

    def embed_one(self, text: str) -> list[float]:
        tokens = tokenize(text)
        if not tokens:
            return [0.0] * self.dim
        counts = Counter(tokens)
        vec = [0.0] * self.dim
        for token, count in counts.items():
            weight = 1.0 + math.log(count)
            digest = hashlib.blake2b(
                token.encode("utf-8"), digest_size=8
            ).digest()
            value = int.from_bytes(digest, "big")
            sign = 1.0 if (value & 1) else -1.0
            for probe in range(self.probes):
                idx = (value >> (probe * 13)) % self.dim
                vec[idx] += sign * weight / (probe + 1.0)
        return l2_normalize(vec)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


def _self_check() -> None:
    assert isinstance(HashingEmbedder(16), Embedder)


_self_check()
