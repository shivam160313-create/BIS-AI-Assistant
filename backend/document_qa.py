import time

from dotenv import load_dotenv
from openai import OpenAI

from backend.retriever import (
    retrieve_documents,
    build_context,
    get_source_names
)


load_dotenv()


client = OpenAI(
    timeout=30.0
)


def ask_document(question: str):

    start_time = time.perf_counter()

    question = question.strip()

    if not question:

        return {
            "question": question,
            "answer": "Please enter a question.",
            "sources": [],
            "timing": {
                "retrieval": 0,
                "ai": 0,
                "total": 0
            }
        }

    retrieval_start = time.perf_counter()

    retrieved_documents = retrieve_documents(
        question,
        n_results=5
    )

    retrieval_time = (
        time.perf_counter()
        - retrieval_start
    )

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
                "total": round(
                    time.perf_counter() - start_time,
                    2
                )
            }
        }

    context = build_context(
        retrieved_documents
    )

    sources = get_source_names(
        retrieved_documents
    )

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

    ai_start = time.perf_counter()

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt,
        reasoning={
            "effort": "none"
        },
        max_output_tokens=350
    )

    ai_time = (
        time.perf_counter()
        - ai_start
    )

    total_time = (
        time.perf_counter()
        - start_time
    )

    return {
        "question": question,
        "answer": response.output_text,
        "sources": sources,
        "timing": {
            "retrieval": round(
                retrieval_time,
                2
            ),
            "ai": round(
                ai_time,
                2
            ),
            "total": round(
                total_time,
                2
            )
        }
    }