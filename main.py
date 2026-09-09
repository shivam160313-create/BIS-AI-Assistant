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
    BASE_DIR
)


UPLOAD_FOLDER = BASE_DIR / "data" / "uploads"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup: if the Chroma collection is empty - which is exactly what
    # happens on a brand-new FastAPI Cloud deployment with no persisted
    # chroma_db - automatically build it from the bundled BIS PDFs in data/.
    # Any failure here is logged, not raised, so a knowledge-base problem
    # never prevents the API itself from coming up (check /health instead).
    try:
        ensure_knowledge_base()
    except Exception as error:
        print(f"Knowledge base build failed on startup: {error}")

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

    try:
        indexed_chunks = collection.count()
        knowledge_base_status = "ready" if indexed_chunks > 0 else "empty"
    except Exception as error:
        indexed_chunks = 0
        knowledge_base_status = f"error: {error}"

    return {
        "status": "healthy",
        "knowledge_base": knowledge_base_status,
        "indexed_chunks": indexed_chunks,
        "openai_configured": bool(os.getenv("OPENAI_API_KEY"))
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
