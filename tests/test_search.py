import faiss
import numpy as np
import pytest

import search


def test_mmr_retrieval_selects_diverse_results(monkeypatch):
    embeddings = np.array(
        [
            [0.95, 0.31225],
            [0.90, 0.43589],
            [0.80, -0.60]
        ],
        dtype="float32"
    )

    index = faiss.IndexFlatIP(2)
    index.add(embeddings)

    chunks = [
        {"page": 1, "text": "Passage A"},
        {"page": 2, "text": "Passage B"},
        {"page": 3, "text": "Passage C"}
    ]

    monkeypatch.setattr(search, "active_chunks", chunks)
    monkeypatch.setattr(search, "active_index", index)
    monkeypatch.setattr(search, "active_embeddings", embeddings)

    monkeypatch.setattr(
        search.model,
        "encode",
        lambda *args, **kwargs: np.array(
            [[1.0, 0.0]],
            dtype="float32"
        )
    )

    results = search.search_documents_mmr(
        "test question",
        top_k=2,
        fetch_k=3,
        lambda_param=0.5
    )

    assert len(results) == 2
    assert results[0]["text"] == "Passage A"
    assert results[1]["text"] == "Passage C"


def test_mmr_rejects_invalid_top_k():
    with pytest.raises(ValueError, match="top_k must be positive"):
        search.search_documents_mmr("test question", top_k=0)


def test_mmr_rejects_fetch_k_smaller_than_top_k():
    with pytest.raises(ValueError, match="fetch_k must be >= top_k"):
        search.search_documents_mmr(
            "test question",
            top_k=5,
            fetch_k=3
        )


def test_mmr_rejects_invalid_lambda():
    with pytest.raises(ValueError, match="lambda_param must be between 0 and 1"):
        search.search_documents_mmr(
            "test question",
            lambda_param=1.5
        )