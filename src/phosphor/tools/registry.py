"""Tool registry with sandboxing and budget accounting."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ..core.errors import GuardError, ToolError
from ..core.ids import random_id
from ..core.types import ToolCall, ToolResult
from .protocol import ToolSpec


class ToolRegistry:
    def __init__(self, allow_network: bool = False, max_output_chars: int = 2000) -> None:
        self.allow_network = allow_network
        self.max_output_chars = max_output_chars
        self._handlers: dict[str, Callable[[dict[str, Any]], str]] = {}
        self._specs: dict[str, ToolSpec] = {}
        self._calls = 0

    def register(
        self,
        spec: ToolSpec,
        handler: Callable[[dict[str, Any]], str] | None = None,
    ) -> None:
        fn = handler or spec.handler
        if fn is None:
            raise ToolError(f"tool {spec.name} has no handler")
        self._specs[spec.name] = spec
        self._handlers[spec.name] = fn

    def has(self, name: str) -> bool:
        return name in self._handlers

    def list_specs(self) -> list[ToolSpec]:
        return [self._specs[n] for n in sorted(self._specs)]

    def invoke(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        args = dict(arguments or {})
        call = ToolCall(name=name, arguments=args, call_id=random_id("call_"))
        started = time.perf_counter()
        handler = self._handlers.get(name)
        if handler is None:
            return ToolResult(
                call=call,
                ok=False,
                error=f"unknown tool: {name} (available: {', '.join(sorted(self._handlers))})",
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )
        spec = self._specs[name]
        if spec.requires_network and not self.allow_network:
            return ToolResult(
                call=call,
                ok=False,
                error=f"tool {name} requires network but networking is disabled",
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )
        self._calls += 1
        try:
            output = handler(args)
        except ToolError as exc:
            return ToolResult(
                call=call, ok=False, error=exc.message,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )
        except ZeroDivisionError:
            return ToolResult(call=call, ok=False, error="division by zero",
                              elapsed_ms=int((time.perf_counter() - started) * 1000))
        except Exception as exc:  # noqa: BLE001 - tools are untrusted boundaries
            return ToolResult(
                call=call, ok=False, error=f"{exc.__class__.__name__}: {exc}",
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )
        output = str(output)
        if len(output) > self.max_output_chars:
            output = output[: self.max_output_chars - 1] + "…"
        return ToolResult(call=call, output=output,
                          elapsed_ms=int((time.perf_counter() - started) * 1000))

    @property
    def call_count(self) -> int:
        return self._calls

    def guard(self, condition: bool, message: str) -> None:
        if not condition:
            raise GuardError(message)
