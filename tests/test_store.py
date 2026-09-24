import importlib.util

import pytest

from phosphor.core.config import StoreConfig
from phosphor.core.mathx import l2_normalize
from phosphor.store import MemoryStore, build_store


def test_memory_search_is_exact_cosine():
    store = MemoryStore(dim=3)
    store.upsert(["a", "b"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                 [{"doc_id": "d"}, {"doc_id": "d"}])
    hits = store.search([1.0, 0.0, 0.0], k=2)
    assert hits[0][0] == "a"
    assert abs(hits[0][1] - 1.0) < 1e-6


def test_drop_document_removes_every_vector():
    store = MemoryStore(dim=2)
    store.upsert(["a", "b"], [[1.0, 0.0], [0.0, 1.0]],
                 [{"doc_id": "x"}, {"doc_id": "y"}])
    assert store.drop_document("x") == 1
    assert [cid for cid, _ in store.search([1.0, 0.0], k=5)] == ["b"]


def test_dimension_mismatch_is_rejected():
    store = MemoryStore(dim=4)
    with pytest.raises(ValueError):
        store.upsert(["a"], [[1.0, 0.0]])


def test_stats_counts_documents():
    store = MemoryStore(dim=2)
    store.upsert(["a", "b"], [[1.0, 0.0], [0.0, 1.0]],
                 [{"doc_id": "x"}, {"doc_id": "x"}])
    assert store.stats()["vectors"] == 2
    assert store.stats()["documents"] == 1


def test_build_store_default_is_memory():
    assert isinstance(build_store(StoreConfig()), MemoryStore)


@pytest.mark.skipif(importlib.util.find_spec("faiss") is None, reason="faiss optional")
def test_faiss_matches_memory_ordering():
    import numpy as np

    from phosphor.store import faiss_impl

    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(40, 16)).astype("float32")
    vecs = [l2_normalize([float(x) for x in row]) for row in matrix]
    ids = [f"v{i}" for i in range(40)]
    query = vecs[3]

    mem = MemoryStore(dim=16)
    mem.upsert(ids, vecs, [{"doc_id": "d"}] * 40)
    faiss_store = faiss_impl.FaissStore(dim=16)
    faiss_store.upsert(ids, vecs, [{"doc_id": "d"}] * 40)

    mem_top = [cid for cid, _ in mem.search(query, 5)]
    faiss_top = [cid for cid, _ in faiss_store.search(query, 5)]
    assert mem_top[0] == faiss_top[0]
    assert set(mem_top[:3]) == set(faiss_top[:3])
