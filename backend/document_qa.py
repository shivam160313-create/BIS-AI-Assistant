import time

from backend.openai_client import get_client, DEFAULT_MODEL
from backend.vector_store import get_status_message_if_not_ready
from backend.retriever import (
    retrieve_documents,
    build_context,
    get_source_names
)


def ask_document(question: str):

    start_time = time.perf_counter()

    question = question.strip()

    if not question:

        return {
            "question": question,
            "answer": "Please enter a question.",
            "sources": [],
            "timing": {"retrieval": 0, "ai": 0, "total": 0}
        }

    not_ready = get_status_message_if_not_ready()

    if not_ready:
        return {
            "question": question,
            "answer": not_ready,
            "sources": [],
            "timing": {"retrieval": 0, "ai": 0, "total": 0}
        }

    retrieval_start = time.perf_counter()

    retrieved_documents = retrieve_documents(
        question,
        n_results=5
    )

    retrieval_time = time.perf_counter() - retrieval_start

    if not retrieved_documents:

        return {
            "question": question,
            "answer": (
                "I could not find relevant information "
                "in the available BIS documents."
            ),
            "sources": [],
            "timing": {
                "retrieval": round(retrieval_time, 2),
                "ai": 0,
                "total": round(time.perf_counter() - start_time, 2)
            }
        }

    context = build_context(retrieved_documents)
    sources = get_source_names(retrieved_documents)

    prompt = f"""
You are the Document Q&A module of a
BIS AI Assistant.

Your job is to answer questions using ONLY
the provided BIS document content.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not invent facts.
3. Do not invent IS numbers.
4. Do not invent certification requirements.
5. Do not assume something that is not present
   in the provided documents.
6. If the answer cannot be found in the sources,
   clearly say:
   "I could not find this information in the
   available BIS documents."
7. Keep the answer concise.
8. Use simple language.
9. Use numbered steps when explaining a process.
10. Mention the relevant document name when useful.

USER QUESTION:
{question}

BIS DOCUMENT SOURCES:
{context}
"""

    try:
        client = get_client()
    except RuntimeError as error:
        return {
            "question": question,
            "answer": f"The AI service is not available right now: {error}",
            "sources": sources,
            "timing": {
                "retrieval": round(retrieval_time, 2),
                "ai": 0,
                "total": round(time.perf_counter() - start_time, 2)
            }
        }

    ai_start = time.perf_counter()

    try:
        response = client.responses.create(
            model=DEFAULT_MODEL,
            input=prompt,
            max_output_tokens=350
        )
    except Exception as error:
        return {
            "question": question,
            "answer": f"The AI service returned an error: {error}",
            "sources": sources,
            "timing": {
                "retrieval": round(retrieval_time, 2),
                "ai": round(time.perf_counter() - ai_start, 2),
                "total": round(time.perf_counter() - start_time, 2)
            }
        }

    ai_time = time.perf_counter() - ai_start
    total_time = time.perf_counter() - start_time

    return {
        "question": question,
        "answer": response.output_text,
        "sources": sources,
        "timing": {
            "retrieval": round(retrieval_time, 2),
            "ai": round(ai_time, 2),
            "total": round(total_time, 2)
        }
    }
