"""LLM boundary."""

from ..core.config import LLMConfig
from .factory import available_providers, build_llm
from .mock import MockLLM, parse_evidence
from .protocol import LLM
from .remote import LlamaCppLLM, OllamaLLM, OpenAICompatLLM

__all__ = [
    "LLM",
    "LLMConfig",
    "LlamaCppLLM",
    "MockLLM",
    "OllamaLLM",
    "OpenAICompatLLM",
    "available_providers",
    "build_llm",
    "parse_evidence",
]
