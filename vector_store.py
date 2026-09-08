import os
import shutil

import chromadb
from pypdf import PdfReader


DATA_FOLDER = "data"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "bis_documents"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


def extract_text(file_path):

    text = ""

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

        chunk = text[
            start:end
        ].strip()

        if chunk:
            chunks.append(chunk)

        start = end - CHUNK_OVERLAP

    return chunks


def clear_collection():

    existing = collection.get()

    ids = existing.get(
        "ids",
        []
    )

    if ids:

        collection.delete(
            ids=ids
        )

        print(
            f"🗑 Removed {len(ids)} old chunks"
        )


def build_knowledge_base():

    all_ids = []
    all_documents = []
    all_metadatas = []

    processed_files = 0

    total_chunks = 0

    for root, dirs, files in os.walk(
        DATA_FOLDER
    ):

        for filename in files:

            if not filename.lower().endswith(
                (".pdf", ".txt")
            ):
                continue

            file_path = os.path.join(
                root,
                filename
            )

            try:

                text = extract_text(
                    file_path
                )

                if not text.strip():

                    print(
                        f"⚠ Empty document: {filename}"
                    )

                    continue

                chunks = chunk_text(
                    text
                )

                category = os.path.basename(
                    root
                )

                safe_name = (
                    filename
                    .replace("\\", "_")
                    .replace("/", "_")
                    .replace(" ", "_")
                )

                for index, chunk in enumerate(
                    chunks
                ):

                    chunk_id = (
                        f"{safe_name}_{index}"
                    )

                    all_ids.append(
                        chunk_id
                    )

                    all_documents.append(
                        chunk
                    )

                    all_metadatas.append({
                        "source": file_path,
                        "category": category,
                        "filename": filename
                    })

                processed_files += 1

                total_chunks += len(chunks)

                print(
                    f"📄 {filename}"
                )

                print(
                    f"   Category: {category}"
                )

                print(
                    f"   Chunks: {len(chunks)}"
                )

            except Exception as error:

                print(
                    f"❌ Failed: {filename}"
                )

                print(
                    f"   Error: {error}"
                )

    if all_ids:

        collection.upsert(
            ids=all_ids,
            documents=all_documents,
            metadatas=all_metadatas
        )

    print()
    print("=" * 55)
    print("🚀 BIS KNOWLEDGE BASE READY")
    print("=" * 55)
    print(
        f"📚 Files processed : {processed_files}"
    )
    print(
        f"🧩 Total chunks    : {total_chunks}"
    )
    print(
        f"🧠 Collection      : {COLLECTION_NAME}"
    )
    print(
        f"💾 Database        : {CHROMA_PATH}"
    )
    print("=" * 55)

    return total_chunks


if __name__ == "__main__":

    print()
    print("🔄 Building BIS Knowledge Base...")
    print()

    build_knowledge_base()