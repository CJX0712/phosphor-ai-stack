"""Arithmetic: detection, normalisation and sandboxed evaluation.

Deterministic pre-routing exists because small models cannot be trusted with
arithmetic: a 0.5B model answers 12*(3+4) with 72 often enough to matter. Any
query containing an evaluable expression is computed here and injected as an
observation, so the model never has to recompute it.
"""

from __future__ import annotations

import ast
import math
import re

from ..core.errors import ToolError

# Code points only: full width digit and operator literals written directly in
# the source get mangled when the file is saved through a non UTF-8 codepage.
_FULL_DIGITS = {0xFF10 + i: 0x30 + i for i in range(10)}
# Operator mapping, kept separate from the trim set below.
_OP_MAP = {
    0xFF0B: "+",  # fullwidth plus
    0xFF0D: "-",  # fullwidth hyphen-minus
    0x00D7: "*",  # multiplication sign
    0x2715: "*",  # multiplication x
    0x2716: "*",  # heavy multiplication x
    0x00F7: "/",  # division sign
    0xFF0F: "/",  # fullwidth solidus
    0xFF1D: "=",  # fullwidth equals
    0xFF08: "(",  # fullwidth left paren
    0xFF09: ")",  # fullwidth right paren
    0xFF05: "%",  # fullwidth percent
    0x2212: "-",  # minus sign
    0x2013: "-",  # en dash
    0x2014: "-",  # em dash
}
# Full width punctuation to strip from the tail of a question. This set must
# never contain operator characters, otherwise legal expressions get chopped.
_TRIM = "？?。．.！!，,、；;：: 　\t\r\n\"'“”‘’"

_EXPR_CHARS = re.compile(r"[0-9+\-*/().% ]+")


def normalize_expr(raw: str) -> str:
    text = raw.translate(_FULL_DIGITS)
    text = "".join(_OP_MAP.get(ord(ch), ch) for ch in text)
    return text.strip(_TRIM).strip()


def detect_arithmetic(text: str) -> str | None:
    """Return the first evaluable arithmetic fragment found in a question.

    The whole sentence is not parseable as Python, so candidate fragments are
    extracted first and then validated - doing it the other way round makes
    every Chinese question fail detection.
    """
    if not text:
        return None
    raw = normalize_expr(text)
    candidates: list[str] = []
    whole = raw.strip()
    if whole and _looks_like_expr(whole):
        candidates.append(whole)
    for match in _EXPR_CHARS.finditer(raw):
        frag = match.group(0).strip()
        if len(frag) >= 3 and _looks_like_expr(frag):
            candidates.append(frag)
    # Longest first: "12 * (3 + 4)" must win over "12" and "3 + 4".
    for cand in sorted(set(candidates), key=len, reverse=True):
        try:
            safe_eval(cand)
        except ToolError:
            continue
        return cand
    return None


def _looks_like_expr(text: str) -> bool:
    return any(ch.isdigit() for ch in text) and any(
        ch in "+-*/%" for ch in text
    )


def safe_eval(expr: str) -> float:
    """Evaluate an arithmetic expression with no name resolution."""
    expr = normalize_expr(expr)
    if not expr:
        raise ToolError("empty expression")
    if len(expr) > 200:
        raise ToolError("expression too long")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"invalid expression: {expr}") from exc
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError("only numbers are allowed")
        return float(node.value)
    if isinstance(node, ast.UnaryOp):
        value = _eval_node(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        raise ToolError("unsupported unary operator")
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _apply(node.op, left, right)
    raise ToolError("unsupported expression node")


def _apply(op: ast.operator, left: float, right: float) -> float:
    try:
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            if right == 0:
                raise ToolError("division by zero")
            return left / right
        if isinstance(op, ast.FloorDiv):
            if right == 0:
                raise ToolError("division by zero")
            return float(left // right)
        if isinstance(op, ast.Mod):
            if right == 0:
                raise ToolError("division by zero")
            return math.fmod(left, right)
        if isinstance(op, ast.Pow):
            return float(left**right)
    except OverflowError as exc:
        raise ToolError("numeric overflow") from exc
    raise ToolError("unsupported operator")


def format_number(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6g}"
