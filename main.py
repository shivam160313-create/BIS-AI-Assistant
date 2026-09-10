import asyncio
import os
import shutil
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    UploadFile,
    File
)

from fastapi.middleware.cors import (
    CORSMiddleware
)

from pydantic import BaseModel

from backend.ai_assistant import (
    ask_bis,
    certification_guide,
    compliance_guide,
    search_standards
)

from backend.document_qa import (
    ask_document
)

from backend.vector_store import (
    extract_text,
    chunk_text,
    collection,
    ensure_knowledge_base,
    BASE_DIR,
    KB_STATUS
)


UPLOAD_FOLDER = BASE_DIR / "data" / "uploads"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # IMPORTANT: this must never block.
    #
    # ensure_knowledge_base() is a synchronous, potentially slow function
    # (it reads 19 PDFs and calls the Gemini embeddings API). Running it
    # directly here - or with `await` on a coroutine that does the work
    # inline - would delay FastAPI's ASGI "startup complete" signal, which
    # is exactly what made FastAPI Cloud's readiness check fail and loop
    # the container.
    #
    # asyncio.to_thread() runs it in a worker thread; asyncio.create_task()
    # schedules that without waiting for it. This coroutine reaches `yield`
    # (and the app becomes ready to serve /health and everything else)
    # essentially instantly, while indexing continues in the background.
    asyncio.create_task(asyncio.to_thread(ensure_knowledge_base))

    yield


app = FastAPI(
    title="BIS AI Assistant",
    description=(
        "AI-powered assistant for "
        "Indian Standards and BIS Services"
    ),
    version="5.0",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):

    question: str


class StandardSearch(BaseModel):

    product: str


@app.get("/")
def root():

    return {
        "message": "BIS AI Assistant is running 🚀",
        "version": "5.0"
    }


@app.get("/health")
def health():
    # Always returns fast and always returns 200 - this endpoint reports
    # status, it doesn't gate on the knowledge base being ready. FastAPI
    # Cloud's readiness check hits this immediately on startup.

    try:
        indexed_chunks = collection.count()
    except Exception:
        indexed_chunks = 0

    return {
        "status": "healthy",
        "knowledge_base": KB_STATUS["state"],
        "knowledge_base_detail": KB_STATUS["detail"],
        "indexed_chunks": indexed_chunks,
        "ai_provider": "google-gemini",
        "google_api_configured": bool(
            os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        )
    }


@app.post("/admin/rebuild-index")
def rebuild_index():
    """
    Manually re-trigger a knowledge-base build in the background, e.g.
    after setting GOOGLE_API_KEY without wanting to wait for a restart.

    Note: this has no authentication. If you expose this publicly long
    term, put an auth dependency on it - it's included here as an
    operational convenience, not a hardened admin API.
    """

    asyncio.create_task(asyncio.to_thread(ensure_knowledge_base))

    return {
        "message": "Knowledge base rebuild triggered in the background.",
        "current_status": KB_STATUS["state"]
    }


@app.post("/ask")
def ask_question(
    data: Question
):

    return ask_bis(
        data.question
    )


@app.post("/standards")
def standards_search(
    data: StandardSearch
):

    return search_standards(
        data.product
    )


@app.post("/certification")
def certification_question(
    data: Question
):

    return certification_guide(
        data.question
    )


@app.post("/compliance")
def compliance_question(
    data: Question
):

    return compliance_guide(
        data.question
    )


@app.post("/document-qa")
def document_question(
    data: Question
):

    return ask_document(
        data.question
    )


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    allowed_extensions = (
        ".pdf",
        ".txt"
    )

    filename = file.filename or ""

    if not filename.lower().endswith(
        allowed_extensions
    ):

        return {
            "success": False,
            "message": (
                "Only PDF and TXT files "
                "are supported."
            )
        }

    UPLOAD_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = UPLOAD_FOLDER / filename

    with open(
        file_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    try:

        text = extract_text(
            file_path
        )

        if not text.strip():

            return {
                "success": False,
                "message": (
                    "Document contains no "
                    "extractable text."
                ),
                "filename": filename
            }

        chunks = chunk_text(
            text
        )

        ids = []
        documents = []
        metadatas = []

        safe_name = (
            filename
            .replace("\\", "_")
            .replace("/", "_")
            .replace(" ", "_")
        )

        for index, chunk in enumerate(
            chunks
        ):

            ids.append(
                f"upload_{safe_name}_{index}"
            )

            documents.append(
                chunk
            )

            metadatas.append({
                "source": str(file_path),
                "category": "uploads",
                "filename": filename
            })

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        return {
            "success": True,
            "message": (
                "Document uploaded and "
                "indexed successfully."
            ),
            "filename": filename,
            "chunks": len(chunks)
        }

    except Exception as error:

        return {
            "success": False,
            "message": (
                "Document uploaded but "
                "indexing failed."
            ),
            "filename": filename,
            "error": str(error)
        }
