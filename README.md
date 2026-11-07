# RAG-Based AI Document Assistant

A local PDF question-answering application built with retrieval-augmented generation (RAG). Upload a PDF, retrieve relevant passages with semantic search, and generate an answer with page references using a locally hosted language model.

The project is intended as an educational AI engineering project demonstrating document ingestion, embeddings, vector search, retrieval evaluation, API development, and local LLM integration.

## Features

- Upload PDF documents through a web interface.
- Extract text and preserve original page numbers with PyMuPDF.
- Split extracted text into overlapping chunks.
- Generate embeddings with Sentence Transformers.
- Retrieve relevant passages using FAISS inner-product similarity.
- Support diversity-aware retrieval using Maximal Marginal Relevance (MMR).
- Compare standard and MMR retrieval using Hit Rate@K and Mean Reciprocal Rank (MRR).
- Generate answers with a local Ollama model (default: `qwen3:1.7b`).
- Return retrieved passages and page numbers alongside each answer.
- Expose document upload, status, and question-answering endpoints through FastAPI.
- Validate file type, PDF signature, upload size, and question length.
- Include API and chunking tests.

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
             PyMuPDF  Embed question
                |       |
           Extract text |
                |       v
           Chunk pages FAISS
                |       |
           Embed chunks |
                |       v
                v   Relevant passages
              FAISS      |
                         v
                   Ollama (Qwen3)
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
6. When a question arrives, the system embeds it and retrieves relevant passages.
7. Standard retrieval ranks passages by similarity. MMR retrieval balances relevance and diversity.
8. The retrieved passages are supplied to Ollama with instructions to answer only from the provided context.
9. The API returns the generated answer and retrieved source passages, including page numbers.

The default embedding model is `all-MiniLM-L6-v2`. The default generation model is `qwen3:1.7b`.

## Tech stack

| Area | Technology |
|---|---|
| Language | Python |
| API | FastAPI, Pydantic, Uvicorn |
| PDF processing | PyMuPDF |
| Embeddings | Sentence Transformers |
| Vector search | FAISS |
| Retrieval optimization | Maximal Marginal Relevance (MMR) |
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
├── tests/
│   ├── test_chunk.py
│   └── test_search.py
├── data/                    # Optional local folder (ignored by Git)
├── app.py
├── chunk.py
├── extract.py
├── search.py
├── rag.py
├── evaluate.py
├── evaluate_retrieval.py
├── evaluation_dataset.json
├── pytest.ini
├── requirements.txt
├── requirements-dev.txt
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

On the first application or test run, Sentence Transformers downloads the
`all-MiniLM-L6-v2` embedding model. This requires an internet connection;
the model is cached locally for later runs.

### 4. Start Ollama and download the model

Make sure the Ollama application or service is running, then execute:

```bash
ollama pull qwen3:1.7b
```

The application defaults to the Ollama server at `http://localhost:11434`. You can override the model and server address with environment variables.

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
python -m pip install -r requirements-dev.txt
python -m pytest -v
```

The test suite covers API endpoints, upload validation, question handling, document chunking, and MMR retrieval behavior.

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
- Standard FAISS retrieval and MMR retrieval are implemented.
- Chunking currently uses character-based chunks (800 characters, with 150 characters of overlap).
- The default generation model is configurable through `OLLAMA_MODEL`.
- The application is designed for local experimentation, not multi-user production deployment.

## Evaluation

Two evaluation scripts are included:

- `evaluate_retrieval.py` compares standard FAISS retrieval with Maximal Marginal Relevance (MMR) retrieval.
- `evaluate.py` runs a small keyword-match retrieval check for example questions.

### Retrieval evaluation methodology

Retrieval is evaluated using `evaluation_dataset.json`, which contains 18 questions about the sample document and manually assigned relevant PDF page numbers.

The evaluation reports two metrics:

- **Hit Rate@K:** The proportion of questions for which at least one relevant page appears in the top K retrieved results.
- **Mean Reciprocal Rank (MRR@K):** Measures how highly the first relevant result is ranked. Higher values indicate better ranking.

Both retrieval methods are evaluated at K = 3 and K = 10.

### Results

| Metric | Standard Retrieval | MMR Retrieval |
|---|---:|---:|
| Hit Rate@3 | 0.889 | 0.889 |
| MRR@3 | 0.741 | 0.741 |
| Hit Rate@10 | 1.000 | 1.000 |
| MRR@10 | 0.757 | 0.769 |

### Observations

- Both methods retrieve at least one relevant page for all 18 questions within the top 10 results.
- MMR achieves a slightly higher MRR@10 than standard retrieval (0.769 compared with 0.757).
- At K = 3, both methods achieve identical overall metrics, although they rank some individual questions differently.
- MMR improves the ranking of some queries but can reduce the ranking of others.

These results suggest that MMR can improve retrieval ranking for certain queries, but does not consistently outperform standard retrieval.

### Limitations

The evaluation uses manually labeled page-level relevance rather than exact chunk-level relevance. It covers one sample PDF and 18 questions, so it should be considered an initial development experiment rather than a comprehensive benchmark.

### Running the retrieval evaluation

```bash
python evaluate_retrieval.py
```

The script automatically indexes `data/sample.pdf` when `search.py` is imported, if that file exists. The `data/` directory is excluded from Git, so the sample PDF is not included in this repository. To run this evaluation, create the `data/` directory and place your own text-based PDF there as `sample.pdf`. The questions and relevant page numbers in `evaluation_dataset.json` must match that PDF.

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
