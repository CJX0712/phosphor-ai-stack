"""Production embedders behind the same Protocol.

All of them import lazily so a missing optional package is a clear runtime
error at first use, never an import error that breaks the whole package.
"""

from __future__ import annotations

from typing import Any

from ..core.config import EmbedConfig
from ..core.errors import ConfigError
from .protocol import Embedder


class FastEmbedProvider:
    """Qdrant fastembed: ONNX runtime, no torch, CPU friendly."""

    name = "fastembed"

    def __init__(self, cfg: EmbedConfig) -> None:
        self.cfg = cfg
        self._model: Any = None
        self.dim = cfg.dim

    def _ensure(self) -> Any:
        if self._model is None:
            try:
                from fastembed import TextEmbedding  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise ConfigError(
                    "fastembed not installed; pip install -r requirements-ai.txt"
                ) from exc
            try:
                self._model = TextEmbedding(model_name=self.cfg.model, cache_dir=self.cfg.cache_dir or None)
            except Exception as exc:
                raise ConfigError(f"fastembed model load failed: {self.cfg.model}") from exc
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure()
        vecs = [list(map(float, v)) for v in model.embed(texts, batch_size=self.cfg.batch_size)]
        if vecs:
            self.dim = len(vecs[0])
        return vecs

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class OpenAIEmbedder:
    """OpenAI compatible /embeddings endpoint (also served by vLLM, Ollama)."""

    name = "openai"

    def __init__(self, cfg: EmbedConfig) -> None:
        if not cfg.base_url:
            raise ConfigError("embed.base_url required for provider=openai")
        self.cfg = cfg
        self.dim = cfg.dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        import httpx

        url = self.cfg.base_url.rstrip("/") + "/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        payload = {"model": self.cfg.model, "input": texts}
        with httpx.Client(timeout=60.0, trust_env=False) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        items = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
        vecs = [[float(x) for x in item["embedding"]] for item in items]
        if vecs:
            self.dim = len(vecs[0])
        return vecs

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class OllamaEmbedder:
    """Ollama native /api/embed endpoint."""

    name = "ollama"

    def __init__(self, cfg: EmbedConfig) -> None:
        self.cfg = cfg
        self.dim = cfg.dim
        self.base = (cfg.base_url or "http://127.0.0.1:11434").rstrip("/")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        import httpx

        vecs: list[list[float]] = []
        with httpx.Client(timeout=120.0, trust_env=False) as client:
            for text in texts:
                resp = client.post(
                    self.base + "/api/embed",
                    json={"model": self.cfg.model, "input": text},
                )
                resp.raise_for_status()
                emb = resp.json().get("embeddings") or []
                if not emb:
                    raise ConfigError("ollama returned no embedding")
                vecs.append([float(x) for x in emb[0]])
        if vecs:
            self.dim = len(vecs[0])
        return vecs

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


def _self_check() -> None:
    assert isinstance(FastEmbedProvider(EmbedConfig()), Embedder)


_self_check()
