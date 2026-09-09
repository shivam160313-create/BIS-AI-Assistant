import os
import threading
from pathlib import Path

import chromadb
from openai import OpenAI
from pypdf import PdfReader


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "manuals"
CHROMA_DIR = BASE_DIR / "chroma_db"


# ============================================================
# CONFIG
# ============================================================

COLLECTION_NAME = "bis_knowledge_base"
EMBEDDING_MODEL = "text-embedding-3-small"

_build_lock = threading.Lock()


# ============================================================
# KNOWLEDGE BASE STATUS
# ============================================================

KB_STATUS = {
    "state": "not_started",
    "detail": "",
    "indexed_chunks": 0
}


# ============================================================
# CHROMA
# IMPORTANT:
# Do NOT use Chroma's OpenAIEmbeddingFunction.
# We provide embeddings manually using OpenAI SDK.
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)


# ============================================================
# OPENAI
# ============================================================

def _get_openai_client():

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    return OpenAI(
        api_key=api_key
    )


def _embed_texts(texts):

    client = _get_openai_client()

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )

    return [
        item.embedding
        for item in response.data
    ]


def embed_query(query):

    embeddings = _embed_texts(
        [query]
    )

    return embeddings[0]


# ============================================================
# DOCUMENT TEXT EXTRACTION
# ============================================================

def extract_text(file_path):

    file_path = Path(file_path)

    extension = file_path.suffix.lower()

    # TXT
    if extension == ".txt":

        return file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    # PDF
    if extension == ".pdf":

        reader = PdfReader(
            str(file_path)
        )

        pages = []

        for page in reader.pages:

            text = page.extract_text() or ""

            if text.strip():
                pages.append(
                    text
                )

        return "\n\n".join(pages)

    raise ValueError(
        "Unsupported file type."
    )


# ============================================================
# TEXT CHUNKING
# ============================================================

def chunk_text(
    text,
    chunk_size=1500,
    overlap=200
):

    text = text.strip()

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        chunk = text[
            start:end
        ].strip()

        if chunk:
            chunks.append(
                chunk
            )

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# ============================================================
# LOAD BIS MANUALS
# ============================================================

def _load_documents():

    documents = []
    metadatas = []
    ids = []

    files = sorted(
        DATA_DIR.glob("*")
    )

    chunk_counter = 0

    for file_path in files:

        if file_path.suffix.lower() != ".pdf":
            continue

        try:

            reader = PdfReader(
                str(file_path)
            )

            for page_number, page in enumerate(
                reader.pages,
                start=1
            ):

                text = page.extract_text() or ""

                text = text.strip()

                if not text:
                    continue

                chunks = chunk_text(
                    text
                )

                for chunk in chunks:

                    documents.append(
                        chunk
                    )

                    metadatas.append({
                        "filename": file_path.name,
                        "source": str(file_path),
                        "page": page_number,
                        "category": "bis_manual"
                    })

                    ids.append(
                        f"bis_{file_path.stem}_"
                        f"{page_number}_"
                        f"{chunk_counter}"
                    )

                    chunk_counter += 1

        except Exception as error:

            print(
                f"Warning: failed to read "
                f"{file_path.name}: {error}"
            )

    return (
        documents,
        metadatas,
        ids
    )


# ============================================================
# BUILD KNOWLEDGE BASE
# ============================================================

def ensure_knowledge_base():

    global KB_STATUS

    with _build_lock:

        try:

            existing_count = (
                collection.count()
            )

            if existing_count > 0:

                KB_STATUS["state"] = "ready"

                KB_STATUS["detail"] = (
                    "Existing indexed data found."
                )

                KB_STATUS["indexed_chunks"] = (
                    existing_count
                )

                return

            if not os.getenv("OPENAI_API_KEY"):

                KB_STATUS["state"] = (
                    "skipped_no_api_key"
                )

                KB_STATUS["detail"] = (
                    "OPENAI_API_KEY is not configured."
                )

                KB_STATUS["indexed_chunks"] = 0

                return

            KB_STATUS["state"] = "building"

            KB_STATUS["detail"] = (
                "Reading BIS manuals..."
            )

            documents, metadatas, ids = (
                _load_documents()
            )

            if not documents:

                KB_STATUS["state"] = "empty"

                KB_STATUS["detail"] = (
                    "No BIS PDF documents were found."
                )

                KB_STATUS["indexed_chunks"] = 0

                return

            batch_size = 32

            total = len(documents)

            for start in range(
                0,
                total,
                batch_size
            ):

                end = min(
                    start + batch_size,
                    total
                )

                batch_documents = (
                    documents[start:end]
                )

                batch_metadatas = (
                    metadatas[start:end]
                )

                batch_ids = (
                    ids[start:end]
                )

                KB_STATUS["detail"] = (
                    f"Embedding chunks "
                    f"{start + 1}-{end} "
                    f"of {total}..."
                )

                embeddings = _embed_texts(
                    batch_documents
                )

                collection.upsert(
                    ids=batch_ids,
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                    embeddings=embeddings
                )

            indexed = collection.count()

            KB_STATUS["state"] = "ready"

            KB_STATUS["detail"] = (
                f"Knowledge base ready with "
                f"{indexed} chunks."
            )

            KB_STATUS["indexed_chunks"] = (
                indexed
            )

            print(
                f"Knowledge base ready: "
                f"{indexed} chunks"
            )

        except Exception as error:

            KB_STATUS["state"] = "error"

            KB_STATUS["detail"] = (
                f"Knowledge base build failed: "
                f"{error}"
            )

            try:
                KB_STATUS["indexed_chunks"] = (
                    collection.count()
                )
            except Exception:
                KB_STATUS["indexed_chunks"] = 0

            print(
                f"Knowledge base error: {error}"
            )


# ============================================================
# STATUS MESSAGE
# ============================================================

def get_status_message_if_not_ready():

    state = KB_STATUS["state"]

    if state == "ready":
        return None

    if state == "building":
        return (
            "The BIS knowledge base is still "
            "being built. Please try again "
            "in a moment."
        )

    if state == "skipped_no_api_key":
        return (
            "OPENAI_API_KEY is not configured."
        )

    if state == "empty":
        return (
            "No BIS documents were found "
            "in the knowledge base."
        )

    if state == "error":
        return (
            "The knowledge base failed to build: "
            + KB_STATUS["detail"]
        )

    return (
        "The BIS knowledge base is not ready yet. "
        "Please try again shortly."
    )


# ============================================================
# GET COLLECTION
# ============================================================

def get_collection():

    return collection