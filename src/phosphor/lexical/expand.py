"""Query expansion for the sparse branch only.

Sparse retrieval is monolingual by construction: a Chinese question has zero
token overlap with an English passage. A small domain dictionary bridges that
on the lexical side while the dense side keeps its own signal.

The expansion is deliberately NOT fed to the dense retriever: doing so makes
the two branches agree with each other and destroys the complementarity that
hybrid fusion depends on.
"""

from __future__ import annotations

_GLOSSARY: dict[str, list[str]] = {
    "retrieval": ["检索", "召回"],
    "检索": ["retrieval", "search"],
    "rag": ["检索增强生成", "检索增强"],
    "检索增强生成": ["rag"],
    "embedding": ["嵌入", "向量", "embedding"],
    "嵌入": ["embedding", "向量"],
    "向量": ["vector", "embedding"],
    "vector": ["向量"],
    "agent": ["智能体", "代理"],
    "智能体": ["agent"],
    "rerank": ["重排", "重排序"],
    "重排": ["rerank"],
    "chunk": ["分块", "切片"],
    "分块": ["chunk"],
    "评估": ["evaluation", "eval"],
    "evaluation": ["评估"],
    "架构": ["architecture"],
    "architecture": ["架构"],
    "部署": ["deploy", "deployment"],
    "deploy": ["部署"],
    "延迟": ["latency"],
    "latency": ["延迟"],
    "成本": ["cost"],
    "cost": ["成本"],
    "安全": ["security", "safety"],
    "security": ["安全"],
    "memory": ["记忆", "内存"],
    "记忆": ["memory"],
}


def expand_terms(query: str, limit: int = 4) -> list[str]:
    """Return extra lexical terms for a query, order stable and dedup applied."""
    lowered = query.lower()
    extras: list[str] = []
    for src, targets in _GLOSSARY.items():
        if src in lowered:
            for target in targets:
                if target not in extras and target.lower() not in lowered:
                    extras.append(target)
                if len(extras) >= limit:
                    return extras
    return extras


def expand_query(query: str, limit: int = 4) -> str:
    extras = expand_terms(query, limit)
    return f"{query} {' '.join(extras)}".strip() if extras else query


def glossary_size() -> int:
    return len(_GLOSSARY)
