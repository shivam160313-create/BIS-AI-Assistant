import os
import threading
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader


# ---------------------------------------------------------------------------
# Reliable, working-directory-independent paths.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FOLDER = BASE_DIR / "data"
CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "bis_documents"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
UPSERT_BATCH_SIZE = 100

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

DATA_FOLDER.mkdir(parents=True, exist_ok=True)
CHROMA_PATH.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Embedding function.
#
# Deliberately NOT using Chroma's default local embedding model
# (all-MiniLM-L6-v2 via onnxruntime). That model has to be downloaded
# (~80MB) and loaded into memory on every cold start, which on a small
# cloud instance is exactly what was blowing past FastAPI Cloud's
# readiness window / memory limit and causing the restart loop.
#
# Using OpenAI's hosted embeddings instead means: no model download, no
# local inference cost, and one consistent place (OpenAI) for both
# embeddings and chat completions.
#
# If OPENAI_API_KEY isn't set, we pass no embedding_function at all -
# Chroma will only try to lazily load its own default the first time
# something actually calls .add()/.query() on the collection. Since
# ensure_knowledge_base() below refuses to build when the key is missing,
# that call never happens, so no download is ever triggered in that case
# either.
# ---------------------------------------------------------------------------
def _build_embedding_function():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None

    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBEDDING_MODEL,
    )


client = chromadb.PersistentClient(path=str(CHROMA_PATH))

_embedding_function = _build_embedding_function()

if _embedding_function is not None:
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_function,
    )
else:
    collection = client.get_or_create_collection(name=COLLECTION_NAME)


# ---------------------------------------------------------------------------
# Knowledge-base build status + concurrency guard.
#
# FastAPI Cloud doesn't provide a persistent local disk (their docs point
# you at external managed services - Neon/Supabase/Redis - for state that
# needs to survive a restart). So chroma_db/ is treated as an ephemeral,
# per-instance cache: rebuilt automatically on cold start, but ALWAYS in a
# background thread so it can never block app startup or /health again.
# ---------------------------------------------------------------------------
_build_lock = threading.Lock()

KB_STATUS = {
    "state": "not_started",  # not_started | building | ready | empty | error | skipped_no_api_key
    "chunks": 0,
    "detail": "Knowledge base build has not started yet."
}


def extract_text(file_path):

    text = ""
    file_path = str(file_path)

    if file_path.lower().endswith(".pdf"):

        reader = PdfReader(file_path)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    elif file_path.lower().endswith(".txt"):

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

    return text


def chunk_text(text):

    chunks = []
    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - CHUNK_OVERLAP

    return chunks


def clear_collection():

    existing = collection.get()
    ids = existing.get("ids", [])

    if ids:
        collection.delete(ids=ids)
        print(f"Removed {len(ids)} old chunks")


def build_knowledge_base(force: bool = False):
    """
    Scan DATA_FOLDER recursively for .pdf / .txt files, chunk them, and
    upsert them into the Chroma collection in batches (never one giant
    request). Safe to call more than once.
    """

    if force:
        clear_collection()

    all_ids = []
    all_documents = []
    all_metadatas = []
    processed_files = 0

    for root, dirs, files in os.walk(DATA_FOLDER):

        # Runtime user uploads are indexed individually by /upload, not as
        # part of a bundled-knowledge-base rebuild.
        if Path(root).name == "uploads":
            continue

        for filename in files:

            if not filename.lower().endswith((".pdf", ".txt")):
                continue

            file_path = os.path.join(root, filename)

            try:
                text = extract_text(file_path)

                if not text.strip():
                    print(f"Empty document, skipping: {filename}")
                    continue

                chunks = chunk_text(text)
                category = os.path.basename(root)

                safe_name = (
                    filename
                    .replace("\\", "_")
                    .replace("/", "_")
                    .replace(" ", "_")
                )

                for index, chunk in enumerate(chunks):
                    all_ids.append(f"{safe_name}_{index}")
                    all_documents.append(chunk)
                    all_metadatas.append({
                        "source": file_path,
                        "category": category,
                        "filename": filename
                    })

                processed_files += 1
                print(f"Prepared {filename} ({len(chunks)} chunks, category={category})")

            except Exception as error:
                print(f"Failed to read {filename}: {error}")

    total_chunks = len(all_ids)

    # Upsert in batches so we never send one enormous request to the
    # embeddings API, and so a failure partway through still leaves earlier
    # batches indexed and usable.
    for start in range(0, total_chunks, UPSERT_BATCH_SIZE):
        end = min(start + UPSERT_BATCH_SIZE, total_chunks)

        collection.upsert(
            ids=all_ids[start:end],
            documents=all_documents[start:end],
            metadatas=all_metadatas[start:end],
        )

        print(f"Indexed chunks {start}-{end} of {total_chunks}")

    print("=" * 55)
    print("BIS KNOWLEDGE BASE BUILD COMPLETE")
    print(f"Files processed : {processed_files}")
    print(f"Total chunks    : {total_chunks}")
    print(f"Embedding model : {EMBEDDING_MODEL}")
    print("=" * 55)

    return total_chunks


def ensure_knowledge_base():
    """
    Build the knowledge base if it's currently empty.

    This is a BLOCKING, synchronous function - callers (main.py's lifespan)
    are responsible for running it in a background thread so it never
    blocks the event loop or delays application startup.

    Safe to call more than once: a lock ensures only one build ever runs
    per process; a second call while a build is in progress (or already
    finished) just returns immediately.
    """

    if not _build_lock.acquire(blocking=False):
        return

    try:
        if not os.getenv("OPENAI_API_KEY"):
            KB_STATUS["state"] = "skipped_no_api_key"
            KB_STATUS["detail"] = (
                "OPENAI_API_KEY is not set, so the knowledge base was not "
                "built (embeddings require it). Set the key and restart "
                "the app, or call it again once configured."
            )
            print(KB_STATUS["detail"])
            return

        try:
            existing_count = collection.count()
        except Exception:
            existing_count = 0

        if existing_count > 0:
            KB_STATUS["state"] = "ready"
            KB_STATUS["chunks"] = existing_count
            KB_STATUS["detail"] = "Knowledge base already populated."
            print(KB_STATUS["detail"])
            return

        KB_STATUS["state"] = "building"
        KB_STATUS["detail"] = "Indexing bundled BIS documents in the background..."
        print(KB_STATUS["detail"])

        total_chunks = build_knowledge_base()

        if total_chunks > 0:
            KB_STATUS["state"] = "ready"
            KB_STATUS["chunks"] = total_chunks
            KB_STATUS["detail"] = "Knowledge base build complete."
        else:
            KB_STATUS["state"] = "empty"
            KB_STATUS["detail"] = "No indexable documents were found in data/."

    except Exception as error:
        KB_STATUS["state"] = "error"
        KB_STATUS["detail"] = f"Knowledge base build failed: {error}"
        print(KB_STATUS["detail"])

    finally:
        _build_lock.release()


def get_status_message_if_not_ready():
    """
    Returns a user-facing message if the knowledge base has no documents to
    query yet (still building, skipped for lack of an API key, or errored).
    Returns None once there's something to query, so callers can proceed.
    """

    try:
        if collection.count() > 0:
            return None
    except Exception:
        pass

    if KB_STATUS["state"] == "skipped_no_api_key":
        return (
            "The knowledge base could not be built because OPENAI_API_KEY "
            "is not configured on the server. Set it and restart the app."
        )

    if KB_STATUS["state"] == "error":
        return f"The knowledge base failed to build: {KB_STATUS['detail']}"

    return (
        "The knowledge base is still being indexed in the background. "
        "Please try again in a minute."
    )


if __name__ == "__main__":
    # Manual rebuild: python -m backend.vector_store
    print("Rebuilding BIS Knowledge Base...")
    build_knowledge_base(force=True)
