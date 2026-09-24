"""ReAct agent with deterministic pre-routing.

Two properties are enforced by contract:

* an observation always follows an action, so the loop has a hard upper bound
  on iterations and cannot spin;
* when a deterministic tool can answer exactly (arithmetic, unit conversion),
  the tool result is injected as an observation before the model sees the
  question, so the model never has to recompute it.
"""

from __future__ import annotations

import json
import re
import time

from ..core.config import AgentConfig
from ..core.errors import ToolError
from ..core.events import bus
from ..core.ids import random_id
from ..core.text import strip_citations, word_set, cosine_sets
from ..core.types import AgentAnswer, AgentStep, Message, Scored, ToolResult
from ..llm.protocol import LLM
from ..retrieve.protocol import Retriever
from ..tools.protocol import ToolRegistryProtocol
from ..tools.registry import ToolRegistry
from .prompts import build_messages

_ACTION_RE = re.compile(r"Action:\s*([A-Za-z_][\w\-]*)\s*(.*)?", re.I)
_FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.I | re.S)
_THOUGHT_RE = re.compile(r"Thought:\s*(.+)", re.I)
_PROTOCOL_PREFIX = ("action:", "thought:", "observation:", "final answer:")


def parse_action(text: str) -> tuple[str, dict] | None:
    match = _ACTION_RE.search(text)
    if not match:
        return None
    name = match.group(1).strip()
    raw_args = (match.group(2) or "").strip()
    args: dict = {}
    if raw_args:
        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                args = parsed
        except json.JSONDecodeError:
            args = {"input": raw_args}
    return name, args


def parse_final(text: str) -> str:
    """Extract the final answer.

    The single line fallback must reject protocol lines, otherwise the last
    'Action:' line of a truncated turn is returned to the caller as an answer.
    """
    match = _FINAL_RE.search(text)
    if match:
        return match.group(1).strip()
    for line in reversed([l.strip() for l in text.splitlines()]):
        if not line:
            continue
        if line.lower().startswith(_PROTOCOL_PREFIX):
            continue
        return line
    return ""


def parse_thought(text: str) -> str:
    match = _THOUGHT_RE.search(text)
    return match.group(1).strip() if match else ""


class ReActAgent:
    def __init__(
        self,
        llm: LLM,
        retriever: Retriever | None = None,
        tools: ToolRegistryProtocol | None = None,
        cfg: AgentConfig | None = None,
        deterministic_router=None,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.tools = tools if tools is not None else ToolRegistry()
        self.cfg = cfg or AgentConfig()
        self._router = deterministic_router

    # -- public -----------------------------------------------------------
    def run(self, query: str, top_k: int | None = None, trace_id: str = "") -> AgentAnswer:
        started = time.perf_counter()
        trace_id = trace_id or random_id("tr_")
        steps: list[AgentStep] = []
        observations: list[str] = []
        evidence: list[Scored] = []
        tool_calls = 0

        if self.retriever is not None:
            evidence = self.retriever.search(query, k=top_k or 6, trace_id=trace_id)
            if evidence:
                observations.append(
                    "Evidence:\n"
                    + "\n".join(f"{i}. {s.chunk.render()}" for i, s in enumerate(evidence, 1))
                )

        if self.cfg.deterministic_routing and self._router is not None:
            routed = self._router(query)
            if routed is not None:
                name, args = routed
                if isinstance(self.tools, ToolRegistry) and self.tools.has(name):
                    result = self.tools.invoke(name, args)
                    tool_calls += 1
                    observations.append(f"{name}: {result.output or result.error}")
                    bus().emit("agent.route", trace_id=trace_id, tool=name, ok=result.ok)

        final = ""
        for index in range(max(1, self.cfg.max_iterations)):
            evidence_text = (
                "\n".join(s.chunk.render() for s in evidence) if evidence else ""
            )
            messages = build_messages(query, evidence_text, observations)
            raw = self.llm.complete(messages)
            action = parse_action(raw)
            thought = parse_thought(raw)

            if action is not None and _has_tool(self.tools, action[0]) and "Final Answer:" not in raw:
                name, args = action
                result = _invoke(self.tools, name, args)
                tool_calls += 1
                observations.append(result.observation())
                steps.append(
                    AgentStep(
                        index=index,
                        thought=thought,
                        action=f"{name}({json.dumps(args, ensure_ascii=False)})",
                        observation=result.observation(),
                        tool_result=result,
                    )
                )
                bus().emit("agent.tool", trace_id=trace_id, tool=name, ok=result.ok)
                if _converged(query, result.observation(), self.cfg.converge_threshold):
                    final = result.output or result.observation()
                    break
                continue

            candidate = parse_final(raw)
            if candidate:
                final = candidate
                steps.append(AgentStep(index=index, thought=thought, action="final", observation=""))
                break

        if not final:
            final = self._fallback(query, evidence, observations)

        elapsed = int((time.perf_counter() - started) * 1000)
        answer = AgentAnswer(
            answer=final,
            steps=steps,
            evidence=evidence,
            tool_calls=tool_calls,
            iterations=len(steps),
            converged=bool(final),
            elapsed_ms=elapsed,
            trace_id=trace_id,
            meta={"llm": getattr(self.llm, "name", "unknown")},
        )
        bus().emit(
            "agent.done",
            trace_id=trace_id,
            tool_calls=tool_calls,
            iterations=len(steps),
            elapsed_ms=elapsed,
        )
        return answer

    def _fallback(self, query: str, evidence: list[Scored], observations: list[str]) -> str:
        if observations:
            for line in reversed(observations):
                if line.startswith("calculator:"):
                    return line.split("calculator:", 1)[1].strip()
        if evidence:
            return evidence[0].chunk.text
        return f"cannot answer from available evidence: {query}"


def _has_tool(tools: ToolRegistryProtocol | ToolRegistry, name: str) -> bool:
    has = getattr(tools, "has", None)
    if callable(has):
        return bool(has(name))
    return any(spec.name == name for spec in tools.list_specs())


def _invoke(tools: ToolRegistryProtocol | ToolRegistry, name: str, args: dict) -> ToolResult:
    try:
        return tools.invoke(name, args)
    except ToolError as exc:
        from ..core.types import ToolCall

        return ToolResult(call=ToolCall(name=name, arguments=args), ok=False, error=exc.message)


def _converged(query: str, observation: str, threshold: float) -> bool:
    """Set cosine between question and observation, length insensitive."""
    if not observation:
        return False
    return cosine_sets(word_set(query), word_set(strip_citations(observation))) >= threshold
