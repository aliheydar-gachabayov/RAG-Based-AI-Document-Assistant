
from pathlib import Path
from threading import RLock

import faiss
import numpy as np
import pymupdf

from sentence_transformers import SentenceTransformer
from chunk import chunk_pages

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "sample.pdf"

model = SentenceTransformer("all-MiniLM-L6-v2")

active_chunks = []
active_index = None
active_filename = None
active_embeddings = None

index_lock = RLock()


def build_index(pdf_bytes, filename):
    """Process a PDF and replace the active search index."""

    global active_chunks, active_index, active_filename, active_embeddings

    pages = []

    with pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    ) as document:

        if document.needs_pass:
            raise ValueError("Password-protected PDFs are unsupported.")

        if document.page_count > 200:
            raise ValueError("PDF cannot exceed 200 pages.")

        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text")

            if text.strip():
                pages.append({
                    "page": page_number,
                    "text": text.strip()
                })

    if not pages:
        raise ValueError(
            "No readable text found. Scanned PDFs are not supported."
        )

    chunks = chunk_pages(pages)

    if not chunks:
        raise ValueError("Unable to create text chunks.")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    chunk_embeddings = embeddings.copy()

    with index_lock:
        active_chunks = chunks
        active_index = index
        active_filename = filename
        active_embeddings = chunk_embeddings
    return {
        "filename": filename,
        "pages": len(pages),
        "chunks": len(chunks)
    }


def search_documents(question, top_k=3):
    """Retrieve passages from the active PDF."""

    if top_k < 1:
        raise ValueError("top_k must be positive.")

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )

    with index_lock:
        if active_index is None:
            raise ValueError("No document has been indexed.")

        scores, indices = active_index.search(
            question_embedding,
            min(top_k, len(active_chunks))
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "page": active_chunks[idx]["page"],
                "score": float(score),
                "text": active_chunks[idx]["text"]
            })

    return results


def get_document_info():
    with index_lock:
        return {
            "filename": active_filename,
            "chunks": len(active_chunks),
            "ready": active_index is not None
        }


def search_documents_mmr(question, top_k=3, fetch_k=15, lambda_param=0.7):
    """Retrieve relevant and diverse passages using MMR."""

    if top_k < 1:
        raise ValueError("top_k must be positive.")

    if fetch_k < top_k:
        raise ValueError("fetch_k must be >= top_k.")

    if not 0 <= lambda_param <= 1:
        raise ValueError("lambda_param must be between 0 and 1.")

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )[0]

    with index_lock:
        if active_index is None:
            raise ValueError("No document has been indexed.")

        candidate_count = min(fetch_k, len(active_chunks))

        scores, indices = active_index.search(
            question_embedding.reshape(1, -1),
            candidate_count
        )

        candidate_indices = [
            int(idx) for idx in indices[0] if idx >= 0
        ]

        selected = []
        remaining = candidate_indices.copy()

        while remaining and len(selected) < top_k:
            best_idx = None
            best_score = float("-inf")

            for idx in remaining:
                relevance = float(
                    np.dot(active_embeddings[idx], question_embedding)
                )

                if selected:
                    diversity_penalty = max(
                        float(np.dot(
                            active_embeddings[idx],
                            active_embeddings[selected_idx]
                        ))
                        for selected_idx in selected
                    )
                else:
                    diversity_penalty = 0.0

                mmr_score = (
                    lambda_param * relevance
                    - (1 - lambda_param) * diversity_penalty
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            selected.append(best_idx)
            remaining.remove(best_idx)

        return [
            {
                "page": active_chunks[idx]["page"],
                "score": float(np.dot(
                    active_embeddings[idx],
                    question_embedding
                )),
                "text": active_chunks[idx]["text"]
            }
            for idx in selected
        ]

if PDF_PATH.exists():
    print("Indexing default PDF...")

    result = build_index(
        PDF_PATH.read_bytes(),
        PDF_PATH.name
    )

    print(
        f"Indexed {result['chunks']} chunks "
        f"from {result['filename']}"
    )
