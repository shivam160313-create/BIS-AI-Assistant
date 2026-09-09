import os
from pathlib import Path

import chromadb
from pypdf import PdfReader


# ---------------------------------------------------------------------------
# Reliable, working-directory-independent paths.
#
# BASE_DIR is the project root (the parent of this "backend" package), so
# these paths resolve correctly no matter where the process is launched
# from - local machine, Docker, or FastAPI Cloud.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FOLDER = BASE_DIR / "data"
CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "bis_documents"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Make sure these directories exist even on a brand-new checkout / container.
DATA_FOLDER.mkdir(parents=True, exist_ok=True)
CHROMA_PATH.mkdir(parents=True, exist_ok=True)


# A single Chroma client/collection shared by the whole app.
# get_or_create_collection() never raises if the collection (or the whole
# chroma_db directory) doesn't exist yet - which is exactly the situation on
# a fresh cloud deployment with no persisted database.
client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_or_create_collection(name=COLLECTION_NAME)


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

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

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
    Scan DATA_FOLDER recursively for .pdf / .txt files and index them into
    the Chroma collection.

    Safe to call more than once: upsert() means re-running just refreshes
    the same chunk ids. Pass force=True to wipe the collection first.
    """

    if force:
        clear_collection()

    all_ids = []
    all_documents = []
    all_metadatas = []

    processed_files = 0
    total_chunks = 0

    for root, dirs, files in os.walk(DATA_FOLDER):

        # Never index files a user uploads at runtime as part of the
        # "bundled knowledge base" rebuild - those are indexed individually
        # by the /upload endpoint instead.
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
                total_chunks += len(chunks)

                print(f"Indexed {filename} ({len(chunks)} chunks, category={category})")

            except Exception as error:
                print(f"Failed to index {filename}: {error}")

    if all_ids:
        collection.upsert(
            ids=all_ids,
            documents=all_documents,
            metadatas=all_metadatas
        )

    print("=" * 55)
    print("BIS KNOWLEDGE BASE BUILD COMPLETE")
    print(f"Files processed : {processed_files}")
    print(f"Total chunks    : {total_chunks}")
    print(f"Collection      : {COLLECTION_NAME}")
    print(f"Database path   : {CHROMA_PATH}")
    print("=" * 55)

    return total_chunks


def ensure_knowledge_base():
    """
    Called on application startup. If the collection is empty (a brand-new
    ChromaDB on a fresh deployment), automatically build it from the PDFs
    bundled in data/. If it already has data, do nothing - this keeps
    restarts fast.
    """

    try:
        existing_count = collection.count()
    except Exception:
        existing_count = 0

    if existing_count == 0:
        print("Knowledge base is empty - building it from bundled BIS documents...")
        build_knowledge_base()
    else:
        print(f"Knowledge base already populated with {existing_count} chunks.")


if __name__ == "__main__":
    # Manual rebuild: python -m backend.vector_store
    print("Rebuilding BIS Knowledge Base...")
    build_knowledge_base(force=True)
