
import pytest

from chunk import chunk_pages


def test_chunk_pages_returns_multiple_chunks():
    pages = [
        {
            "page": 1,
            "text": "This is a test sentence. " * 100
        }
    ]

    chunks = chunk_pages(pages, chunk_size=100, overlap=20)

    assert len(chunks) > 1


def test_chunk_pages_preserves_page_numbers():
    pages = [
        {"page": 3, "text": "This is page three."},
        {"page": 7, "text": "This is page seven."}
    ]

    chunks = chunk_pages(pages)

    assert {chunk["page"] for chunk in chunks} == {3, 7}


def test_chunk_pages_respects_chunk_size():
    pages = [
        {"page": 1, "text": "Python programming is interesting. " * 50}
    ]

    chunks = chunk_pages(pages, chunk_size=100, overlap=20)

    assert all(len(chunk["text"]) <= 100 for chunk in chunks)


def test_chunk_pages_handles_empty_text():
    pages = [{"page": 1, "text": "   "}]

    chunks = chunk_pages(pages)

    assert chunks == []


def test_chunk_pages_rejects_invalid_parameters():
    pages = [{"page": 1, "text": "Some text"}]

    with pytest.raises(ValueError):
        chunk_pages(pages, chunk_size=0)

    with pytest.raises(ValueError):
        chunk_pages(pages, chunk_size=100, overlap=100)
