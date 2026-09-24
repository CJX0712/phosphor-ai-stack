"""Tool contract.

A tool is a pure callable plus a JSON schema; the agent never imports a tool
module directly, it only sees the schema and the registry.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..core.types import ToolResult


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    category: str = "general"
    requires_network: bool = False
    handler: Callable[[dict[str, Any]], str] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "category": self.category,
            "requires_network": self.requires_network,
        }


@runtime_checkable
class ToolRegistryProtocol(Protocol):
    def register(self, spec: ToolSpec, handler: Callable[[dict[str, Any]], str]) -> None:
        ...

    def invoke(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        ...

    def list_specs(self) -> list[ToolSpec]:
        ...
