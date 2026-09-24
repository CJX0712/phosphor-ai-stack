"""Embedder assembly."""

from __future__ import annotations

from ..core.config import EmbedConfig
from ..core.errors import ConfigError
from .hashing import HashingEmbedder
from .protocol import Embedder
from .remote import FastEmbedProvider, OllamaEmbedder, OpenAIEmbedder

_REGISTRY: dict[str, type] = {
    "hashing": HashingEmbedder,
    "fastembed": FastEmbedProvider,
    "openai": OpenAIEmbedder,
    "ollama": OllamaEmbedder,
}


def build_embedder(cfg: EmbedConfig | None = None) -> Embedder:
    cfg = cfg or EmbedConfig()
    key = cfg.provider.lower()
    if key == "hashing":
        return HashingEmbedder(dim=cfg.dim)
    cls = _REGISTRY.get(key)
    if cls is None:
        raise ConfigError(
            f"unknown embed provider: {cfg.provider}",
            available=sorted(_REGISTRY),
        )
    return cls(cfg)


def available_providers() -> list[str]:
    return sorted(_REGISTRY)
