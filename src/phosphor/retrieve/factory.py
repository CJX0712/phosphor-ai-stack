"""Retriever assembly."""

from __future__ import annotations

from ..core.config import RetrieveConfig
from ..embed.protocol import Embedder
from ..store.protocol import VectorStore
from .dense import DenseRetriever
from .hybrid import HybridRetriever
from .protocol import Reranker
from .sparse import SparseRetriever


def build_retriever(
    cfg: RetrieveConfig,
    embedder: Embedder,
    store: VectorStore,
    reranker: Reranker | None = None,
) -> HybridRetriever:
    dense = DenseRetriever(embedder, store)
    sparse = SparseRetriever(expand=cfg.bilingual_expand)
    return HybridRetriever(dense=dense, sparse=sparse, cfg=cfg, reranker=reranker)
