import os
import shutil

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
    collection
)


app = FastAPI(
    title="BIS AI Assistant",
    description=(
        "AI-powered assistant for "
        "Indian Standards and BIS Services"
    ),
    version="5.0"
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

    return {
        "status": "healthy",
        "knowledge_base": "connected"
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

    upload_folder = "data/uploads"

    os.makedirs(
        upload_folder,
        exist_ok=True
    )

    file_path = os.path.join(
        upload_folder,
        filename
    )

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
                "source": file_path,
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