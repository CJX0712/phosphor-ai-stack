"""LLM assembly."""

from __future__ import annotations

from ..core.config import LLMConfig
from ..core.errors import ConfigError
from .mock import MockLLM
from .protocol import LLM
from .remote import LlamaCppLLM, OllamaLLM, OpenAICompatLLM


def build_llm(cfg: LLMConfig | None = None) -> LLM:
    cfg = cfg or LLMConfig()
    key = cfg.provider.lower()
    if key == "mock":
        return MockLLM()
    if key == "ollama":
        return OllamaLLM(cfg)
    if key == "openai":
        return OpenAICompatLLM(cfg)
    if key == "llamacpp":
        return LlamaCppLLM(cfg)
    raise ConfigError(f"unknown llm provider: {cfg.provider}")


def available_providers() -> list[str]:
    return ["llamacpp", "mock", "ollama", "openai"]
