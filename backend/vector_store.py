from pathlib import Path
import threading
import numpy as np
import chromadb
from pypdf import PdfReader
from sklearn.feature_extraction.text import HashingVectorizer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"

vectorizer = HashingVectorizer(
    n_features=768,
    alternate_sign=False,
    norm="l2"
)

_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = _client.get_or_create_collection("bis_knowledge")
collection = _collection

KB_STATUS = {"state": "not_ready", "detail": "Knowledge base not built yet."}
_lock = threading.Lock()


def _embed_texts(texts, task_type=None):
    matrix = vectorizer.transform(texts).toarray().astype(np.float32)
    return matrix.tolist()


def embed_query(query):
    return _embed_texts([query])[0]


def get_collection():
    return _collection


def get_kb_status():
    return KB_STATUS.copy()


def get_status_message_if_not_ready():
    status = get_kb_status()
    if status["state"] == "ready":
        return None
    return status["detail"]


def build_knowledge_base():
    with _lock:
        KB_STATUS["state"] = "building"
        KB_STATUS["detail"] = "Building BIS knowledge base..."

        documents = []
        metadatas = []
        ids = []

        pdf_files = list(BASE_DIR.glob("*.pdf")) + list(DATA_DIR.glob("*.pdf"))

        for pdf in pdf_files:
            try:
                reader = PdfReader(str(pdf))
                for page_no, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    text = text.strip()

                    if not text:
                        continue

                    chunk_size = 1200
                    overlap = 200

                    start = 0
                    chunk_no = 0

                    while start < len(text):
                        chunk = text[start:start + chunk_size]

                        if len(chunk.strip()) > 50:
                            documents.append(chunk)
                            metadatas.append({
                                "source": pdf.name,
                                "page": page_no + 1
                            })
                            ids.append(
                                f"{pdf.stem}-{page_no}-{chunk_no}"
                            )

                        start += chunk_size - overlap
                        chunk_no += 1

            except Exception as e:
                print(f"Skipping {pdf.name}: {e}")

        if not documents:
            KB_STATUS["state"] = "error"
            KB_STATUS["detail"] = "No PDF documents found."
            return

        print(f"Prepared {len(documents)} chunks. Creating local embeddings...")

        embeddings = _embed_texts(documents)

        try:
            _collection.delete(where={})
        except Exception:
            pass

        batch_size = 100

        for i in range(0, len(documents), batch_size):
            end = min(i + batch_size, len(documents))
            print(f"Indexing chunks {i + 1}-{end} of {len(documents)}...")

            _collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                embeddings=embeddings[i:end]
            )

        KB_STATUS["state"] = "ready"
        KB_STATUS["detail"] = f"BIS knowledge base ready: {len(documents)} chunks"
        print(KB_STATUS["detail"])


def ensure_knowledge_base():
    return build_knowledge_base()

def extract_text(pdf_path):
    reader = PdfReader(str(pdf_path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)

def chunk_text(text, chunk_size=1200, overlap=200):
    text = text or ""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
