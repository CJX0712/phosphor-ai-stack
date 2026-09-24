"""Deterministic extractive LLM.

Why the default model is extractive rather than empty:

* the whole stack can be exercised and scored offline - retrieval, grounding
  and evaluation all produce real numbers with no API key and no download;
* it is deterministic, so golden evaluation thresholds are meaningful;
* it refuses to invent: when evidence does not support an answer it says so,
  which is the failure mode a production model is most likely to hide.

It is not a language model and is labelled as such in every response metadata.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from ..core.text import cosine_sets, split_sentences, strip_citations, tokenize, word_set
from ..core.types import Message
from ..lexical.expand import expand_query
from .protocol import LLM

_CONTEXT_START = "<kb-context>"
_CONTEXT_END = "</kb-context>"
_EVIDENCE_ID_RE = re.compile(r"\[id#([A-Za-z0-9_\-]+)\]")


def _extract_question(messages: list[Message]) -> str:
    """Recover the user question.

    Taking the last user message is wrong: the runtime appends observations as
    additional user turns, and an extractive reader would then score sentences
    against its own evidence instead of against the question.
    """
    for msg in messages:
        if msg.role == "user" and _CONTEXT_START in msg.content:
            head = msg.content.split(_CONTEXT_START)[0]
            for line in head.splitlines():
                if line.lower().startswith("question:"):
                    return line.split(":", 1)[1].strip()
            return head.strip()
    for msg in messages:
        if msg.role == "user":
            return msg.content
    return ""


def _deterministic_observation(messages: list[Message]) -> str:
    """Return a verified tool result that was injected by deterministic routing."""
    for msg in messages:
        if msg.role != "user" or "Observation:" not in msg.content:
            continue
        for line in msg.content.splitlines():
            stripped = line.replace("Observation:", "").strip()
            if stripped.startswith("calculator:"):
                return stripped.split("calculator:", 1)[1].strip()
    return ""


def parse_evidence(messages: list[Message]) -> list[tuple[str, str]]:
    """Return [(chunk_id, text)] parsed from the context block.

    Continuation lines are appended to the previous entry: rendering is
    guaranteed single line per chunk, but defensive parsing keeps the reader
    correct even if a caller renders differently.
    """
    text = "\n".join(m.content for m in messages)
    if _CONTEXT_START not in text:
        return []
    body = text.split(_CONTEXT_START, 1)[1].split(_CONTEXT_END, 1)[0]
    items: list[tuple[str, str]] = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _EVIDENCE_ID_RE.search(line)
        if m:
            items.append((m.group(1), _EVIDENCE_ID_RE.sub("", line).strip()))
        elif items:
            cid, prev = items[-1]
            items[-1] = (cid, f"{prev} {line}")
    return items


class MockLLM:
    """Extractive reader over the context block."""

    name = "mock"
    model = "extractive-v1"

    def __init__(self, max_sentences: int = 3) -> None:
        self.max_sentences = max_sentences

    def complete(self, messages: list[Message], **kwargs: object) -> str:
        question = _extract_question(messages)
        # A deterministic tool result outranks extractive reading: the value is
        # computed, not inferred, so echoing it verbatim is strictly better.
        routed = _deterministic_observation(messages)
        if routed:
            return f"{routed} (source: calculator)"
        evidence = parse_evidence(messages)
        if not evidence:
            return f"[mock] no evidence available for: {question}"
        # Query understanding (bilingual expansion) belongs to the reader too:
        # without it an English question cannot match a Chinese passage even
        # after retrieval succeeded on the expanded sparse query.
        q_terms = word_set(expand_query(question))
        best: list[tuple[float, str, str]] = []
        for cid, text in evidence:
            clean = strip_citations(text)
            for sentence in split_sentences(clean):
                if len(sentence) < 4:
                    continue
                score = cosine_sets(q_terms, word_set(sentence))
                if score > 0:
                    best.append((score, sentence, cid))
        if not best:
            return f"[mock] evidence does not answer: {question}"
        best.sort(key=lambda t: (-t[0], t[2]))
        # De-duplicate: repeated corpora produce identical sentences and an
        # answer that repeats the same line N times scores as well as a real one.
        picked: list[tuple[float, str, str]] = []
        seen: set[str] = set()
        for score, sentence, cid in best:
            key = sentence.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            picked.append((score, sentence, cid))
            if len(picked) >= self.max_sentences:
                break
        answer = " ".join(sentence for _score, sentence, _cid in picked)
        cites = ", ".join(sorted({cid for _score, _sentence, cid in picked}))
        return f"{answer} (source: {cites})"

    def stream(self, messages: list[Message], **kwargs: object) -> Iterator[str]:
        final = self.complete(messages, **kwargs)
        for token in tokenize(final) or [final]:
            yield token + " "


def _self_check() -> None:
    assert isinstance(MockLLM(), LLM)


_self_check()
