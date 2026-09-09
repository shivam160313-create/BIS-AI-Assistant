import os
import threading
from pathlib import Path

import chromadb
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "manuals"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "bis_knowledge_base"
EMBEDDING_MODEL = "text-embedding-3-small"

_build_lock = threading.Lock()

KB_STATUS = "not_started"
KB_STATUS_DETAIL = ""
KB_INDEXED_CHUNKS = 0


client = chromadb.PersistentClient(path=str(CHROMA_DIR))

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


def _get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None

    return OpenAI(api_key=api_key)


def _embed_texts(texts):
    openai_client = _get_openai_client()

    if openai_client is None:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )

    return [item.embedding for item in response.data]


def embed_query(query):
    embeddings = _embed_texts([query])
    return embeddings[0]


def _load_documents():
    documents = []
    metadatas = []
    ids = []

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        return documents, metadatas, ids

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "pypdf is required to read BIS PDF manuals."
        ) from exc

    chunk_id = 0

    for pdf_file in pdf_files:

        try:
            reader = PdfReader(str(pdf_file))

            for page_number, page in enumerate(reader.pages, start=1):

                text = page.extract_text() or ""
                text = text.strip()

                if not text:
                    continue

                chunk_size = 1500
                overlap = 200

                start = 0

                while start < len(text):

                    end = min(start + chunk_size, len(text))
                    chunk = text[start:end].strip()

                    if chunk:
                        documents.append(chunk)

                        metadatas.append({
                            "filename": pdf_file.name,
                            "page": page_number
                        })

                        ids.append(f"{pdf_file.stem}-{page_number}-{chunk_id}")
                        chunk_id += 1

                    if end >= len(text):
                        break

                    start = end - overlap

        except Exception as exc:
            print(
                f"Warning: failed to read {pdf_file.name}: {exc}"
            )

    return documents, metadatas, ids


def build_knowledge_base():
    global KB_STATUS
    global KB_STATUS_DETAIL
    global KB_INDEXED_CHUNKS

    with _build_lock:

        if KB_STATUS == "ready" and collection.count() > 0:
            return

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            KB_STATUS = "skipped_no_api_key"
            KB_STATUS_DETAIL = (
                "OPENAI_API_KEY is not set, so the knowledge base "
                "was not built."
            )
            KB_INDEXED_CHUNKS = 0
            return

        try:
            if collection.count() > 0:
                KB_STATUS = "ready"
                KB_STATUS_DETAIL = "Existing indexed data found."
                KB_INDEXED_CHUNKS = collection.count()
                return

            KB_STATUS = "building"
            KB_STATUS_DETAIL = "Reading BIS manuals..."

            documents, metadatas, ids = _load_documents()

            if not documents:
                KB_STATUS = "empty"
                KB_STATUS_DETAIL = "No document chunks were found."
                KB_INDEXED_CHUNKS = 0
                return

            batch_size = 32

            for start in range(0, len(documents), batch_size):

                end = min(start + batch_size, len(documents))

                batch_documents = documents[start:end]
                batch_metadatas = metadatas[start:end]
                batch_ids = ids[start:end]

                KB_STATUS_DETAIL = (
                    f"Embedding chunks {start + 1}-{end} "
                    f"of {len(documents)}..."
                )

                embeddings = _embed_texts(batch_documents)

                collection.upsert(
                    ids=batch_ids,
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                    embeddings=embeddings
                )

            KB_INDEXED_CHUNKS = collection.count()
            KB_STATUS = "ready"
            KB_STATUS_DETAIL = (
                f"Knowledge base ready with "
                f"{KB_INDEXED_CHUNKS} chunks."
            )

        except Exception as exc:
            KB_STATUS = "error"
            KB_STATUS_DETAIL = (
                f"Knowledge base build failed: {exc}"
            )
            KB_INDEXED_CHUNKS = collection.count()

            raise


def get_collection():
    return collection


def get_kb_status():
    return {
        "status": KB_STATUS,
        "detail": KB_STATUS_DETAIL,
        "indexed_chunks": KB_INDEXED_CHUNKS
    }