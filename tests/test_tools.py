import pytest

from phosphor.core.errors import ToolError
from phosphor.tools import (
    ToolRegistry,
    detect_arithmetic,
    register_builtins,
    route_deterministic,
    safe_eval,
)


def registry() -> ToolRegistry:
    return register_builtins(ToolRegistry())


def test_calculator_evaluates():
    result = registry().invoke("calculator", {"expression": "12*(3+4)"})
    assert result.ok
    assert "84" in result.output


def test_fullwidth_arithmetic():
    assert safe_eval("１２×（３＋４）") == 84.0


def test_division_by_zero_is_tool_error_not_exception():
    result = registry().invoke("calculator", {"expression": "1/0"})
    assert not result.ok
    assert "zero" in result.error.lower()


def test_invalid_expression_is_rejected():
    with pytest.raises(ToolError):
        safe_eval("__import__('os').system('echo')")


def test_detect_arithmetic_from_chinese_sentence():
    assert detect_arithmetic("请计算 12*(3+4) 的结果是多少？") is not None


def test_route_deterministic_returns_calculator_args():
    routed = route_deterministic("12*(3+4) 等于多少？")
    assert routed is not None
    name, args = routed
    assert name == "calculator"
    assert "expression" in args


def test_route_deterministic_returns_none_for_prose():
    assert route_deterministic("混合检索的融合策略是什么？") is None


def test_unit_conversion():
    result = registry().invoke("convert", {"value": 1, "from": "km", "to": "m"})
    assert result.ok and "1000" in result.output


def test_network_tool_is_blocked_by_default():
    result = registry().invoke("http_get", {"url": "https://example.com"})
    assert not result.ok


def test_unknown_tool_reports_error():
    result = registry().invoke("nope")
    assert not result.ok and "unknown tool" in result.error


def test_kb_search_tool_uses_retriever():
    class FakeRetriever:
        def search(self, query, k=6, trace_id=""):
            from phosphor.core.types import Chunk

            return [Scored(chunk=Chunk(id="x", doc_id="d", text="证据文本"), score=1.0)]

    from phosphor.core.types import Scored

    reg = register_builtins(ToolRegistry(), retriever=FakeRetriever())
    result = reg.invoke("kb_search", {"query": "任意"})
    assert result.ok and "证据文本" in result.output


def test_output_is_truncated():
    reg = ToolRegistry(max_output_chars=10)
    reg.register(registry().list_specs()[0], lambda args: "x" * 100)
    result = reg.invoke("calculator")
    assert len(result.output) <= 10
