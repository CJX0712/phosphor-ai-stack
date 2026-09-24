"""Kernel of the stack: types, config, errors, events, text and math.

Nothing in here may import from any other phosphor package.
"""

from .config import Config
from .errors import (
    ConfigError,
    EvalError,
    GuardError,
    IngestError,
    LLMError,
    OrchestrationError,
    PhosphorError,
    RetrievalError,
    StorageError,
    ToolError,
    normalize_exception,
)
from .events import Event, EventBus, bus
from .ids import content_hash, stable_id
from .mathx import cosine, l2_normalize
from .text import normalize, strip_citations, tokenize
from .types import (
    AgentAnswer,
    AgentStep,
    CaseResult,
    Chunk,
    Document,
    EvalCase,
    EvalReport,
    IngestReport,
    Message,
    Scored,
    ToolCall,
    ToolResult,
)

__all__ = [
    "AgentAnswer",
    "AgentStep",
    "CaseResult",
    "Chunk",
    "ConfigError",
    "Config",
    "Document",
    "EvalCase",
    "EvalError",
    "EvalReport",
    "Event",
    "EventBus",
    "GuardError",
    "IngestError",
    "IngestReport",
    "LLMError",
    "Message",
    "OrchestrationError",
    "PhosphorError",
    "RetrievalError",
    "Scored",
    "StorageError",
    "ToolCall",
    "ToolError",
    "ToolResult",
    "bus",
    "content_hash",
    "cosine",
    "l2_normalize",
    "normalize",
    "normalize_exception",
    "stable_id",
    "strip_citations",
    "tokenize",
]
