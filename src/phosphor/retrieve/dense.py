"""Dense retrieval: embed the query, search the vector index."""

from __future__ import annotations

from ..core.types import Chunk, Scored
from ..embed.protocol import Embedder
from ..store.protocol import VectorStore


class DenseRetriever:
    name = "dense"

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def index(self, chunks: list[Chunk], vectors: list[list[float]] | None = None) -> None:
        if not chunks:
            return
        vecs = vectors if vectors is not None else self.embedder.embed([c.text for c in chunks])
        self.store.upsert(
            [c.id for c in chunks],
            vecs,
            [{"doc_id": c.doc_id, "order": c.order} for c in chunks],
        )

    def drop_document(self, doc_id: str) -> int:
        return self.store.drop_document(doc_id)

    def search(self, query: str, k: int = 6) -> list[tuple[str, float]]:
        vec = self.embedder.embed_one(query)
        return self.store.search(vec, k)

    def search_scored(self, query: str, k: int, chunk_of: dict[str, Chunk]) -> list[Scored]:
        out: list[Scored] = []
        for cid, score in self.search(query, k):
            chunk = chunk_of.get(cid)
            if chunk is None:
                continue
            out.append(Scored(chunk=chunk, score=score, source=self.name, detail={"cosine": score}))
        return out

    def stats(self) -> dict:
        base = self.store.stats()
        base["embedder"] = getattr(self.embedder, "name", "unknown")
        return base
