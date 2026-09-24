"""Phosphor AI Stack.

An end to end retrieval augmented agent platform: ingest, chunk, embed, index,
hybrid retrieve, rerank, reason with tools, orchestrate, evaluate, serve.

Every external capability sits behind a Protocol with a real zero dependency
default implementation, so the default configuration is the configuration that
is continuously verified.
"""

from .core.config import Config
from .core.errors import PhosphorError
from .pipeline import Pipeline, build_pipeline

__version__ = "1.0.0"
__author__ = "晨星"

__all__ = ["Config", "Pipeline", "PhosphorError", "__author__", "__version__", "build_pipeline"]
