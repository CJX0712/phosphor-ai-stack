"""Phosphor error taxonomy.

Single place where every failure mode of the stack is named, so callers can
catch by semantic category instead of by exception class from a third party.
"""

from __future__ import annotations

from typing import Any


class PhosphorError(Exception):
    """Base class for every error raised by the stack."""

    code = "phosphor_error"

    def __init__(self, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "detail": self.detail}


class ConfigError(PhosphorError):
    code = "config_error"


class IngestError(PhosphorError):
    code = "ingest_error"


class RetrievalError(PhosphorError):
    code = "retrieval_error"


class ToolError(PhosphorError):
    """Raised when a tool refuses or fails to produce a result."""

    code = "tool_error"


class LLMError(PhosphorError):
    code = "llm_error"


class OrchestrationError(PhosphorError):
    code = "orchestration_error"


class StorageError(PhosphorError):
    code = "storage_error"


class EvalError(PhosphorError):
    code = "eval_error"


class GuardError(PhosphorError):
    """Raised by safety guards (sandbox limits, budget, content policy)."""

    code = "guard_error"


def normalize_exception(exc: BaseException) -> PhosphorError:
    """Wrap any foreign exception into the Phosphor taxonomy."""
    if isinstance(exc, PhosphorError):
        return exc
    return PhosphorError(str(exc) or exc.__class__.__name__, origin=exc.__class__.__name__)
