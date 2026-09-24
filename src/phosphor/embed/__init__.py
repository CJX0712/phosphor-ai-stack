"""Embedding boundary."""

from ..core.config import EmbedConfig
from .factory import available_providers, build_embedder
from .hashing import HashingEmbedder
from .protocol import Embedder
from .remote import FastEmbedProvider, OllamaEmbedder, OpenAIEmbedder

__all__ = [
    "EmbedConfig",
    "Embedder",
    "FastEmbedProvider",
    "HashingEmbedder",
    "OllamaEmbedder",
    "OpenAIEmbedder",
    "available_providers",
    "build_embedder",
]
