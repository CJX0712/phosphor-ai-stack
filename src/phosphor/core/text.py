"""Text primitives shared by chunking, lexical search and evaluation.

Deliberately dependency free and bilingual: the same tokenizer must handle
"检索增强生成" and "retrieval augmented generation" without a segmenter.
"""

from __future__ import annotations

import re
import unicodedata

# CJK ranges used instead of literal characters, so the source file carries no
# symbol literals that a non UTF-8 codepage could mangle.
_CJK_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2A6DF),
)
_LATIN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_DIGIT_RE = re.compile(r"\d+(?:[.,]\d+)*")
_CJK_RUN_RE = re.compile(
    "[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in _CJK_RANGES) + "]+"
)

# Sentence terminators: a full width mark always ends a sentence, an ASCII dot
# only when followed by whitespace or end of string (so "Python 3.10" survives).
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？；])|(?<=[.!?])(?=\s|$)")

# Quote/annotation markers that must never be treated as an assertion.
_CITATION_RE = re.compile(r"\[[^\]\n]{0,40}#[^\]\n]{0,20}\]")
_SOURCE_RE = re.compile(
    r"[(（](?:依据|来源|参考|source|ref| cite)[^)）]{0,60}[)）]", re.IGNORECASE
)


def is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(a <= cp <= b for a, b in _CJK_RANGES)


def has_cjk(text: str) -> bool:
    return any(is_cjk(ch) for ch in text)


def normalize(text: str) -> str:
    """NFKC normalize, collapse whitespace, unify full width punctuation."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Bilingual tokenizer: latin words, digits, and CJK character bigrams."""
    if not text:
        return []
    text = normalize(text).lower()
    tokens: list[str] = []
    for m in _LATIN_RE.finditer(text):
        tokens.append(m.group(0))
    for m in _DIGIT_RE.finditer(text):
        tokens.append(m.group(0).replace(",", ""))
    for m in _CJK_RUN_RE.finditer(text):
        run = m.group(0)
        if len(run) == 1:
            tokens.append(run)
            continue
        # Unigram plus bigram: recall of single characters matters in Chinese.
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def char_ngrams(text: str, n: int = 3) -> list[str]:
    clean = normalize(text).lower().replace(" ", "")
    if len(clean) <= n:
        return [clean] if clean else []
    return [clean[i : i + n] for i in range(len(clean) - n + 1)]


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(normalize(text))
    return [p.strip() for p in parts if p and p.strip()]


def strip_citations(text: str) -> str:
    """Remove provenance markers before grounding/faithfulness scoring.

    A citation such as (source: doc#3) is metadata about an assertion, not an
    assertion itself; leaving it in makes every answer look unsupported.
    """
    text = _CITATION_RE.sub(" ", text)
    text = _SOURCE_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def word_set(text: str) -> set[str]:
    return set(tokenize(text))


def cosine_sets(a: set[str], b: set[str]) -> float:
    """Length insensitive set cosine.

    Coverage based scoring (|A∩B|/|A|) collapses for long questions against
    short answers; set cosine does not, which is why the agent loop can be
    given a hard convergence bound.
    """
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / ((len(a) * len(b)) ** 0.5)


def truncate(text: str, limit: int = 400) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"
