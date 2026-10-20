
import pytest
from fastapi.testclient import TestClient

import app as app_module

client = TestClient(app_module.app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert "AI Document Assistant" in response.text


def test_invalid_upload():
    response = client.post(
        "/upload",
        files={
            "file": (
                "notes.txt",
                b"Not a PDF",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400


def test_large_upload():
    response = client.post(
        "/upload",
        files={
            "file": (
                "large.pdf",
                b"%PDF-" + b"x" * (10 * 1024 * 1024),
                "application/pdf"
            )
        }
    )

    assert response.status_code == 413


def test_upload_success(monkeypatch):

    def fake_build_index(content, filename):
        return {
            "filename": filename,
            "pages": 2,
            "chunks": 5
        }

    monkeypatch.setattr(
        app_module,
        "build_index",
        fake_build_index
    )

    response = client.post(
        "/upload",
        files={
            "file": (
                "test.pdf",
                b"%PDF-fake-test-content",
                "application/pdf"
            )
        }
    )

    assert response.status_code == 200
    assert response.json()["chunks"] == 5


def test_ask(monkeypatch):

    monkeypatch.setattr(
        app_module,
        "get_document_info",
        lambda: {"ready": True}
    )

    monkeypatch.setattr(
        app_module,
        "generate_answer",
        lambda question: (
            "Attention uses queries, keys and values.",
            [{
                "page": 2,
                "score": 0.85,
                "text": "Attention mechanism example."
            }]
        )
    )

    response = client.post(
        "/ask",
        json={"question": "What is attention?"}
    )

    assert response.status_code == 200

    data = response.json()

    assert "Attention" in data["answer"]
    assert len(data["sources"]) == 1
    assert data["sources"][0]["page"] == 2


def test_empty_question():
    response = client.post(
        "/ask",
        json={"question": ""}
    )

    assert response.status_code == 422
