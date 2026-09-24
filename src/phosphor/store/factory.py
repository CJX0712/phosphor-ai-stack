"""Vector store assembly."""

from __future__ import annotations

from ..core.config import StoreConfig
from ..core.errors import ConfigError
from .faiss_impl import FaissStore
from .memory import MemoryStore
from .protocol import VectorStore


def build_store(cfg: StoreConfig | None = None, dim: int = 384) -> VectorStore:
    cfg = cfg or StoreConfig()
    key = cfg.provider.lower()
    if key == "memory":
        return MemoryStore(dim=dim)
    if key == "faiss":
        return FaissStore(dim=dim, path=cfg.path)
    raise ConfigError(f"unknown store provider: {cfg.provider}")


def available_providers() -> list[str]:
    return ["faiss", "memory"]
