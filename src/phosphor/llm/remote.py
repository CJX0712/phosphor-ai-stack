"""Production LLM backends: Ollama, OpenAI compatible, llama.cpp.

trust_env=False is mandatory: with a system SOCKS/HTTP proxy configured, httpx
otherwise routes loopback traffic through the proxy and Ollama calls fail with
WinError 10054 connection reset.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

from ..core.config import LLMConfig
from ..core.errors import ConfigError, LLMError
from ..core.types import Message
from .protocol import LLM


def _to_payload(messages: list[Message]) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in messages]


class OllamaLLM:
    name = "ollama"

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self.model = cfg.model
        self.base = (cfg.base_url or "http://127.0.0.1:11434").rstrip("/")

    def complete(self, messages: list[Message], **kwargs: object) -> str:
        import httpx

        payload = {
            "model": self.model,
            "messages": _to_payload(messages),
            "stream": False,
            "options": {
                "temperature": float(kwargs.get("temperature", self.cfg.temperature)),
                "num_predict": int(kwargs.get("max_tokens", self.cfg.max_tokens)),
            },
        }
        try:
            with httpx.Client(timeout=self.cfg.timeout_s, trust_env=False) as client:
                resp = client.post(self.base + "/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"ollama call failed: {exc}") from exc
        return str(data.get("message", {}).get("content", "")).strip()

    def stream(self, messages: list[Message], **kwargs: object) -> Iterator[str]:
        import httpx

        payload = {
            "model": self.model,
            "messages": _to_payload(messages),
            "stream": True,
        }
        with httpx.Client(timeout=self.cfg.timeout_s, trust_env=False) as client:
            with client.stream("POST", self.base + "/api/chat", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    piece = data.get("message", {}).get("content", "")
                    if piece:
                        yield str(piece)


class OpenAICompatLLM:
    """Any /v1/chat/completions endpoint: OpenAI, vLLM, LM Studio, OpenRouter."""

    name = "openai"

    def __init__(self, cfg: LLMConfig) -> None:
        if not cfg.base_url:
            raise ConfigError("llm.base_url required for provider=openai")
        self.cfg = cfg
        self.model = cfg.model

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        return headers

    def complete(self, messages: list[Message], **kwargs: object) -> str:
        import httpx

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _to_payload(messages),
            "temperature": float(kwargs.get("temperature", self.cfg.temperature)),
            "max_tokens": int(kwargs.get("max_tokens", self.cfg.max_tokens)),
            "stream": False,
        }
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        try:
            with httpx.Client(timeout=self.cfg.timeout_s, trust_env=False) as client:
                resp = client.post(url, json=payload, headers=self._headers())
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"openai-compatible call failed: {exc}") from exc
        choices = data.get("choices") or []
        if not choices:
            raise LLMError("empty completion")
        return str(choices[0].get("message", {}).get("content", "")).strip()

    def stream(self, messages: list[Message], **kwargs: object) -> Iterator[str]:
        import httpx

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _to_payload(messages),
            "stream": True,
        }
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        with httpx.Client(timeout=self.cfg.timeout_s, trust_env=False) as client:
            with client.stream("POST", url, json=payload, headers=self._headers()) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        break
                    try:
                        data = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    piece = delta.get("content")
                    if piece:
                        yield str(piece)


class LlamaCppLLM:
    """In-process GGUF inference through llama-cpp-python."""

    name = "llamacpp"

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self.model = cfg.model_path or cfg.model
        self._llm: Any = None

    def _ensure(self) -> Any:
        if self._llm is None:
            try:
                from llama_cpp import Llama  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise ConfigError("llama-cpp-python not installed") from exc
            self._llm = Llama(model_path=self.cfg.model_path, n_ctx=4096, verbose=False, n_threads=4)
        return self._llm

    def complete(self, messages: list[Message], **kwargs: object) -> str:
        llm = self._ensure()
        out = llm.create_chat_completion(
            messages=_to_payload(messages),
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        return str(out["choices"][0]["message"]["content"]).strip()

    def stream(self, messages: list[Message], **kwargs: object) -> Iterator[str]:
        llm = self._ensure()
        for chunk in llm.create_chat_completion(
            messages=_to_payload(messages), stream=True
        ):
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            piece = delta.get("content")
            if piece:
                yield str(piece)


def _self_check() -> None:
    assert isinstance(OllamaLLM(LLMConfig()), LLM)


_self_check()
