"""Deterministic identifiers.

Every id in the stack is content derived so that re-ingesting the same bytes
twice produces the same ids: this is what makes the system reproducible and
lets tests assert on stable keys.
"""

from __future__ import annotations

import hashlib
import uuid

_NAMESPACE = uuid.UUID("6f0f5c1e-1d2a-5b3c-9e77-0a1b2c3d4e5f")


def stable_id(*parts: str, prefix: str = "", length: int = 16) -> str:
    """Hash based id, stable across processes and machines."""
    h = hashlib.blake2b("\x1f".join(parts).encode("utf-8"), digest_size=16).hexdigest()
    short = h[:length]
    return f"{prefix}{short}" if prefix else short


def uuid5_id(*parts: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, "\x1f".join(parts)))


def random_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:16]}"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
