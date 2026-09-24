"""Configuration objects.

One nested dataclass per module, populated from env with the PHOSPHOR_ prefix.
No module reads os.environ directly except this one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any

from .errors import ConfigError


@dataclass
class ChunkConfig:
    size: int = 700
    overlap: int = 120
    min_size: int = 24
    respect_headings: bool = True


@dataclass
class EmbedConfig:
    provider: str = "hashing"  # hashing | fastembed | openai | ollama
    model: str = "BAAI/bge-small-zh-v1.5"
    dim: int = 384
    batch_size: int = 32
    api_key: str = ""
    base_url: str = ""
    cache_dir: str = ".cache/embeddings"


@dataclass
class StoreConfig:
    provider: str = "memory"  # memory | faiss
    path: str = ".cache/index"
    metric: str = "cosine"


@dataclass
class RetrieveConfig:
    top_k: int = 6
    candidate_multiplier: int = 4
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    rrf_k: int = 60
    rerank: bool = True
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_provider: str = "auto"  # auto | lexical | cross-encoder
    max_evidence_chars: int = 2400
    bilingual_expand: bool = True


@dataclass
class LLMConfig:
    provider: str = "mock"  # mock | ollama | openai | llamacpp
    model: str = "qwen2.5:0.5b-instruct"
    base_url: str = "http://127.0.0.1:11434"
    api_key: str = ""
    temperature: float = 0.1
    max_tokens: int = 512
    timeout_s: float = 60.0
    model_path: str = ""


@dataclass
class AgentConfig:
    max_iterations: int = 6
    deterministic_routing: bool = True
    require_evidence: bool = True
    converge_threshold: float = 0.55
    stream: bool = False


@dataclass
class OrchestrationConfig:
    enabled: bool = True
    max_workers: int = 4
    max_rounds: int = 2
    critic_enabled: bool = True


@dataclass
class APISettings:
    host: str = "127.0.0.1"
    port: int = 8080
    prefix: str = "/api/v1"
    cors_origins: tuple[str, ...] = ("*",)
    api_token: str = ""


@dataclass
class Config:
    env: str = "dev"
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    retrieve: RetrieveConfig = field(default_factory=RetrieveConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    orch: OrchestrationConfig = field(default_factory=OrchestrationConfig)
    api: APISettings = field(default_factory=APISettings)
    data_dir: str = ".cache/phosphor"
    telemetry_enabled: bool = True

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Config:
        src = dict(os.environ if env is None else env)
        cfg = cls()
        cfg.env = src.get("PHOSPHOR_ENV", cfg.env)
        cfg.data_dir = src.get("PHOSPHOR_DATA_DIR", cfg.data_dir)
        cfg.telemetry_enabled = _bool(src.get("PHOSPHOR_TELEMETRY"), True)

        mapping = {
            "PHOSPHOR_CHUNK_SIZE": (cfg.chunk, "size", int),
            "PHOSPHOR_CHUNK_OVERLAP": (cfg.chunk, "overlap", int),
            "PHOSPHOR_EMBED_PROVIDER": (cfg.embed, "provider", str),
            "PHOSPHOR_EMBED_MODEL": (cfg.embed, "model", str),
            "PHOSPHOR_EMBED_DIM": (cfg.embed, "dim", int),
            "PHOSPHOR_STORE_PROVIDER": (cfg.store, "provider", str),
            "PHOSPHOR_STORE_PATH": (cfg.store, "path", str),
            "PHOSPHOR_TOP_K": (cfg.retrieve, "top_k", int),
            "PHOSPHOR_RERANK": (cfg.retrieve, "rerank", _bool),
            "PHOSPHOR_RERANK_PROVIDER": (cfg.retrieve, "rerank_provider", str),
            "PHOSPHOR_LLM_PROVIDER": (cfg.llm, "provider", str),
            "PHOSPHOR_LLM_MODEL": (cfg.llm, "model", str),
            "PHOSPHOR_LLM_BASE_URL": (cfg.llm, "base_url", str),
            "PHOSPHOR_LLM_API_KEY": (cfg.llm, "api_key", str),
            "PHOSPHOR_LLM_MODEL_PATH": (cfg.llm, "model_path", str),
            "PHOSPHOR_AGENT_MAX_ITER": (cfg.agent, "max_iterations", int),
            "PHOSPHOR_ORCH_ENABLED": (cfg.orch, "enabled", _bool),
            "PHOSPHOR_API_HOST": (cfg.api, "host", str),
            "PHOSPHOR_API_PORT": (cfg.api, "port", int),
            "PHOSPHOR_API_TOKEN": (cfg.api, "api_token", str),
        }
        for key, (holder, attr, cast) in mapping.items():
            raw = src.get(key)
            if raw is None or raw == "":
                continue
            try:
                setattr(holder, attr, cast(raw))
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"invalid value for {key}", key=key, value=raw) from exc
        return cfg

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"env": self.env, "data_dir": self.data_dir}
        for f in fields(self):
            if f.name in ("env", "data_dir", "telemetry_enabled"):
                continue
            value = getattr(self, f.name)
            if hasattr(value, "__dataclass_fields__"):
                out[f.name] = {
                    g.name: getattr(value, g.name) for g in fields(value)
                }
            else:
                out[f.name] = value
        out["telemetry_enabled"] = self.telemetry_enabled
        return out


def _bool(raw: str, default: bool = False) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _int(raw: str) -> int:
    return int(raw)
