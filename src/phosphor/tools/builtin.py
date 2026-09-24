"""Built-in tools: calculator, clock, unit conversion, knowledge search, HTTP."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from ..core.errors import ToolError
from .arith import detect_arithmetic, format_number, safe_eval
from .protocol import ToolSpec
from .registry import ToolRegistry

_UNITS: dict[str, dict[str, float]] = {
    "length": {"m": 1.0, "km": 1000.0, "cm": 0.01, "mm": 0.001, "mi": 1609.344, "ft": 0.3048, "in": 0.0254},
    "mass": {"kg": 1.0, "g": 0.001, "t": 1000.0, "lb": 0.45359237, "oz": 0.028349523125},
    "data": {"b": 1.0, "kb": 1024.0, "mb": 1048576.0, "gb": 1073741824.0, "tb": 1099511627776.0},
    "time": {"s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0, "ms": 0.001},
}


def _calc(args: dict[str, Any]) -> str:
    expr = str(args.get("expression", "")).strip()
    if not expr:
        raise ToolError("expression is required")
    return f"{expr} = {format_number(safe_eval(expr))}"


def _clock(args: dict[str, Any]) -> str:
    offset_h = float(args.get("offset_hours", 0) or 0)
    tz = timezone(timedelta(hours=offset_h))
    now = datetime.now(tz)
    fmt = str(args.get("format", "%Y-%m-%d %H:%M:%S")) or "%Y-%m-%d %H:%M:%S"
    return now.strftime(fmt)


def _convert(args: dict[str, Any]) -> str:
    value = float(args.get("value", 0))
    src = str(args.get("from", "")).lower()
    dst = str(args.get("to", "")).lower()
    for family, table in _UNITS.items():
        if src in table and dst in table:
            result = value * table[src] / table[dst]
            return f"{value:g} {src} = {format_number(result)} {dst}"
    if src == "c" and dst == "f":
        return f"{value:g} C = {format_number(value * 9 / 5 + 32)} F"
    if src == "f" and dst == "c":
        return f"{value:g} F = {format_number((value - 32) * 5 / 9)} C"
    raise ToolError(f"unsupported conversion: {src} -> {dst}")


def _json_tool(args: dict[str, Any]) -> str:
    """Serialise/deserialise helper used by the agent for structured output."""
    raw = args.get("data")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ToolError(f"invalid json: {exc}") from exc
    return json.dumps(raw, ensure_ascii=False, indent=2)


def _noop(args: dict[str, Any]) -> str:
    return str(args.get("text", ""))


def make_kb_search(retriever: Any, default_k: int = 6) -> Callable[[dict[str, Any]], str]:
    """Knowledge base search tool bound to a retriever instance."""

    def _search(args: dict[str, Any]) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            raise ToolError("query is required")
        k = int(args.get("k", default_k) or default_k)
        scored = retriever.search(query, k=k)
        if not scored:
            return "no evidence found"
        lines = []
        for i, item in enumerate(scored, start=1):
            lines.append(f"{i}. {item.chunk.render()}")
        return "\n".join(lines)

    return _search


def make_http_get(allow_network: bool = False) -> Callable[[dict[str, Any]], str]:
    def _http_get(args: dict[str, Any]) -> str:
        if not allow_network:
            raise ToolError("network access is disabled")
        url = str(args.get("url", "")).strip()
        if not url.startswith(("http://", "https://")):
            raise ToolError("only http(s) urls are allowed")
        import httpx

        with httpx.Client(timeout=15.0, follow_redirects=True, trust_env=False) as client:
            resp = client.get(url)
        return resp.text[:4000]

    return _http_get


def register_builtins(
    registry: ToolRegistry,
    retriever: Any | None = None,
    allow_network: bool = False,
) -> ToolRegistry:
    specs: list[tuple[ToolSpec, Callable[[dict[str, Any]], str]]] = [
        (
            ToolSpec(
                name="calculator",
                description="Evaluate an arithmetic expression, e.g. 12*(3+4).",
                parameters={"expression": "string"},
                category="math",
            ),
            _calc,
        ),
        (
            ToolSpec(
                name="clock",
                description="Current date/time, optionally offset by hours.",
                parameters={"offset_hours": "number", "format": "string"},
                category="time",
            ),
            _clock,
        ),
        (
            ToolSpec(
                name="convert",
                description="Convert units: length, mass, data, time, C/F.",
                parameters={"value": "number", "from": "string", "to": "string"},
                category="math",
            ),
            _convert,
        ),
        (
            ToolSpec(
                name="json",
                description="Parse or serialise JSON and pretty-print it.",
                parameters={"data": "any"},
                category="utility",
            ),
            _json_tool,
        ),
        (
            ToolSpec(
                name="echo",
                description="Return the input text unchanged.",
                parameters={"text": "string"},
                category="utility",
            ),
            _noop,
        ),
        (
            ToolSpec(
                name="http_get",
                description="Fetch a URL (disabled unless networking is enabled).",
                parameters={"url": "string"},
                category="network",
                requires_network=True,
            ),
            make_http_get(allow_network),
        ),
    ]
    for spec, handler in specs:
        registry.register(spec, handler)

    if retriever is not None:
        registry.register(
            ToolSpec(
                name="kb_search",
                description="Search the knowledge base and return evidence snippets.",
                parameters={"query": "string", "k": "number"},
                category="retrieval",
            ),
            make_kb_search(retriever),
        )
    return registry


def route_deterministic(query: str) -> tuple[str, dict[str, str]] | None:
    """Deterministic pre-routing: (tool_name, arguments).

    Returns None when no deterministic tool applies and the model should decide.
    """
    expr = detect_arithmetic(query)
    if expr is not None:
        try:
            safe_eval(expr)
        except ToolError:
            return None
        return "calculator", {"expression": expr}
    return None
