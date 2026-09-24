"""Prompt construction for the ReAct loop.

The context delimiter is globally unique (kb-context) and never appears inside
the instruction text: an earlier version used a plain marker that collided with
the marker used inside the instruction itself, and the mock reader then quoted
the instructions back as if they were evidence.
"""

from __future__ import annotations

from ..core.types import Message

CTX_START = "<kb-context>"
CTX_END = "</kb-context>"

SYSTEM_PROMPT = (
    "You are Phosphor, a retrieval augmented assistant.\n"
    "Rules:\n"
    "1. Answer only from the supplied evidence when evidence is present.\n"
    "2. Cite the evidence identifiers you used.\n"
    "3. If the evidence does not contain the answer, say so explicitly.\n"
    "4. Use a tool when a deterministic tool can answer exactly.\n"
    "Protocol, one turn at a time:\n"
    "Thought: <one sentence of reasoning>\n"
    "Action: <tool_name> {\"arg\": \"value\"}\n"
    "Observation: is appended for you by the runtime.\n"
    "Finish with:\n"
    "Final Answer: <answer>\n"
)

NO_EVIDENCE_HINT = "No evidence was retrieved; rely on tools or state that you cannot answer."


def build_messages(
    query: str,
    evidence_text: str,
    observations: list[str] | None = None,
    history: list[Message] | None = None,
    system: str = SYSTEM_PROMPT,
) -> list[Message]:
    messages: list[Message] = [Message(role="system", content=system)]
    if history:
        messages.extend(history[-6:])
    if evidence_text.strip():
        body = f"{CTX_START}\n{evidence_text}\n{CTX_END}"
        content = (
            f"Question: {query}\n\nEvidence block follows.\n{body}"
        )
    else:
        content = f"Question: {query}\n\n{NO_EVIDENCE_HINT}"
    messages.append(Message(role="user", content=content))
    if observations:
        joined = "\n".join(f"Observation: {o}" for o in observations[-6:])
        messages.append(Message(role="user", content=joined))
    return messages


def render_tool_catalog(lines: list[str]) -> str:
    return "\n".join(f"- {line}" for line in lines)
