# RAG-Based AI Document Assistant

A local PDF question-answering application built with retrieval-augmented generation (RAG). Upload a PDF, retrieve relevant passages with semantic search, and generate an answer with page references using a locally hosted language model.

The project is intended as an educational AI engineering project demonstrating document ingestion, embeddings, vector search, API development, and evaluation.

## Features

- Upload PDF documents through a web interface.
- Extract text and preserve original page numbers with PyMuPDF.
- Split extracted text into overlapping chunks.
- Generate embeddings with Sentence Transformers.
- Retrieve relevant passages using FAISS inner-product similarity.
- Generate answers with a local Ollama model (default: `qwen3:1.7b`).
- Return retrieved passages and page numbers alongside each answer.
- Expose document upload, status, and question-answering endpoints through FastAPI.
- Validate file type, PDF signature, upload size, and question length.
- Include API tests for core endpoints and validation behavior.

## Architecture

```text
                 Browser
                    |
                    v
             FastAPI application
               /     |      \
              /      |       \
          /upload   /ask   /document
             |       |
             v       v
        PyMuPDF   Embed question
             |       |
        Extract text  v
             |      FAISS
        Chunk pages   |
             |        v
        Embed chunks  Relevant passages
             |        |
             v        v
            FAISS   Ollama (Qwen3)
                       |
                       v
                 Answer + sources
```

### Retrieval and generation flow

1. The user uploads a PDF.
2. The backend validates the upload and extracts text page by page.
3. Text is split into overlapping character-based chunks while retaining page numbers.
4. Sentence Transformers creates normalized embeddings for each chunk.
5. FAISS stores the embeddings in an inner-product index.
6. When a question arrives, the system embeds it and retrieves the top three passages.
7. The retrieved passages are supplied to Ollama with instructions to answer only from the provided context.
8. The API returns the generated answer and retrieved source passages, including page numbers.

The default embedding model is `all-MiniLM-L6-v2`. The default generation model is `qwen3:1.7b`.

## Tech stack

| Area | Technology |
|---|---|
| Language | Python |
| API | FastAPI, Pydantic, Uvicorn |
| PDF processing | PyMuPDF |
| Embeddings | Sentence Transformers |
| Vector search | FAISS |
| Generation | Ollama, Qwen3 |
| Numerical computing | NumPy |
| Frontend | HTML, CSS, JavaScript |
| Testing | Pytest, FastAPI TestClient |
| Containerization | Docker |

## Project structure

```text
RAG-Based-AI-Document-Assistant/
├── frontend/
│   ├── index.html
│   └── tests/
│       └── test_api.py
├── app.py
├── chunk.py
├── extract.py
├── search.py
├── rag.py
├── evaluate.py
├── evaluate_retrieval.py
├── evaluation_dataset.json
├── requirements.txt
├── Dockerfile
└── .gitignore
```

## Getting started

### Prerequisites

- Python 3.14 (or a compatible Python version supported by the pinned dependencies)
- [Ollama](https://ollama.com/)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/aliheydar-gachabayov/RAG-Based-AI-Document-Assistant.git
cd RAG-Based-AI-Document-Assistant
```

### 2. Create and activate a virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start Ollama and download the model

Make sure the Ollama application or service is running, then execute:

```bash
ollama pull qwen3:1.7b
```

The application defaults to the Ollama server at `http://localhost:11434`. You can override the model and server address with environment variables:

macOS / Linux:

```bash
export OLLAMA_MODEL=qwen3:1.7b
export OLLAMA_HOST=http://localhost:11434
```

Windows PowerShell:

```powershell
$env:OLLAMA_MODEL = "qwen3:1.7b"
$env:OLLAMA_HOST = "http://localhost:11434"
```

### 5. Run the application

```bash
uvicorn app:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

Useful endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /` | Web interface |
| `GET /health` | Health check |
| `GET /document` | Active document status |
| `POST /upload` | Upload and index a PDF |
| `POST /ask` | Ask a question about the active PDF |

Interactive API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 6. Run tests

```bash
python -m pytest frontend/tests -v
```

The API tests cover the health endpoint, frontend response, invalid and oversized uploads, successful upload behavior with a mocked indexer, question handling, and empty-question validation.

## API examples

### Upload a PDF

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@your-document.pdf"
```

A successful response includes the uploaded filename, number of pages with extractable text, and number of indexed chunks.

### Ask a question

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main topics in this document?"}'
```

The response contains:
- The original question.
- The generated answer.
- Retrieved source passages, with page numbers and similarity scores.

## Configuration and current constraints

- Maximum upload size: 10 MB.
- Maximum PDF length: 200 pages.
- Only PDFs with extractable text are supported; scanned PDFs require OCR, which is not currently implemented.
- Password-protected PDFs are rejected.
- One active document index is held in application memory. Uploading a new PDF replaces the active index.
- The default retrieval setting is top-`k` = 3.
- Chunking currently uses character-based chunks (800 characters, with 150 characters of overlap).
- The default generation model is configurable through `OLLAMA_MODEL`.
- The application is designed for local experimentation, not multi-user production deployment.

## Evaluation

Two evaluation scripts are included:

- `evaluate_retrieval.py` evaluates retrieval against manually labeled relevant page numbers in `evaluation_dataset.json` and reports Hit Rate@3 and MRR@3.
- `evaluate.py` runs a small keyword-match retrieval check for example questions.

To run either script, first start the application or initialize the document index in the same Python process with a suitable PDF. The FAISS index is in memory and is not automatically shared between separate processes.

The current evaluation data is a starting point, not a comprehensive benchmark. Results should be interpreted as development checks rather than evidence of production-level retrieval or answer quality.

## Docker

Build the image:

```bash
docker build -t rag-document-assistant .
```

Run the container:

```bash
docker run --rm -p 8000:8000 \
  -e OLLAMA_MODEL=qwen3:1.7b \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  rag-document-assistant
```

The Ollama server must be reachable from inside the container. The `host.docker.internal` hostname is supported by Docker Desktop on macOS and Windows; Linux setups may require a host-gateway configuration or another reachable host address.

Then visit [http://localhost:8000](http://localhost:8000).

## Security and limitations

- Uploaded PDF content is passed to a language model. Treat documents as untrusted input.
- The generation prompt instructs the model to use retrieved passages only, but this does not guarantee that hallucinations or prompt-injection behavior are impossible.
- Similarity scores are retrieval signals, not calibrated probabilities of relevance.
- The in-memory index is lost when the application process restarts.
- The current single-active-document design is not suitable for concurrent users or tenant isolation.
- Do not expose the development server publicly without adding authentication, resource limits, persistent storage, and production deployment controls.

## Future improvements

- Add OCR support for scanned PDFs.
- Use token-aware or semantic chunking and compare chunking strategies.
- Persist indexes and support multiple documents.
- Add a reranking stage and compare retrieval configurations.
- Expand the labeled evaluation set and report Recall@K, MRR, and nDCG.
- Evaluate answer faithfulness, citation correctness, and answer relevance.
- Add automated CI testing and a reproducible evaluation workflow.
- Add authentication and production-grade upload controls.

## Author

**Aliheydar Gachabayov**

- [GitHub](https://github.com/aliheydar-gachabayov)
- [LinkedIn](https://www.linkedin.com/in/aliheydar-gachabayov)
