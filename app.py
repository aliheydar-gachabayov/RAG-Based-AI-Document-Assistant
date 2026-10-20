
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from ollama import ResponseError

from rag import generate_answer
from search import build_index, get_document_info

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_PATH = BASE_DIR / "frontend" / "index.html"

MAX_FILE_SIZE = 10 * 1024 * 1024

request_lock = Lock()

app = FastAPI(
    title="RAG Document Assistant",
    description="PDF question answering with FAISS and Ollama",
    version="1.1.0"
)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


@app.get("/")
def home():
    return FileResponse(FRONTEND_PATH)


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/document")
def document_info():
    return get_document_info()


@app.post("/upload")
def upload_document(file: UploadFile = File(...)):

    filename = Path(
        (file.filename or "").replace("\\", "/")
    ).name

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    content = file.file.read(MAX_FILE_SIZE + 1)

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="PDF exceeds the 10 MB limit."
        )

    if not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="Invalid PDF file."
        )

    with request_lock:
        try:
            result = build_index(content, filename)
            return {
                "message": "Document indexed successfully.",
                **result
            }

        except (ValueError, RuntimeError) as error:
            raise HTTPException(
                status_code=400,
                detail=str(error)
            )


@app.post("/ask")
def ask_document(request: QuestionRequest):

    with request_lock:
        if not get_document_info()["ready"]:
            raise HTTPException(
                status_code=400,
                detail="Upload a PDF before asking questions."
            )

        try:
            answer, sources = generate_answer(
                request.question
            )

            return {
                "question": request.question,
                "answer": answer,
                "sources": [
                    {
                        "page": source["page"],
                        "similarity": round(source["score"], 4),
                        "text": source["text"]
                    }
                    for source in sources
                ]
            }

        except ResponseError:
            raise HTTPException(
                status_code=502,
                detail="The language model request failed."
            )

        except ConnectionError:
            raise HTTPException(
                status_code=503,
                detail="Cannot connect to Ollama."
            )
