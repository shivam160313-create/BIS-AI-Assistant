import time

from backend.ai_client import get_client
from backend.vector_store import collection, get_status_message_if_not_ready

CHAT_MODEL = "gemini-3.6-flash"


def get_sources(results):
    sources = []
    for metadata in results.get("metadatas", [[]])[0]:
        source = metadata.get("source")
        if source:
            filename = source.replace("\\", "/").split("/")[-1]
            if filename not in sources:
                sources.append(filename)
    return sources


def get_context(query):
    start = time.perf_counter()
    results = collection.query(query_texts=[query], n_results=3)
    retrieval_time = time.perf_counter() - start

    documents = results["documents"][0]
    context = "\n\n".join(
        f"SOURCE {i + 1}: {doc}"
        for i, doc in enumerate(documents)
    )
    return results, context, retrieval_time


def _empty_timing():
    return {"retrieval": 0, "openai": 0, "total": 0}


def _not_ready_response(query, message, key="question"):
    return {
        key: query,
        "answer": message,
        "sources": [],
        "timing": _empty_timing()
    }


def _error_response(query, results, retrieval_time, total_start, error, key="question"):
    return {
        key: query,
        "answer": f"The AI service is not available right now: {error}",
        "sources": get_sources(results),
        "timing": {
            "retrieval": round(retrieval_time, 2),
            "openai": 0,
            "total": round(time.perf_counter() - total_start, 2)
        }
    }


def ask_ai(query, mode="assistant"):
    not_ready = get_status_message_if_not_ready()

    if not_ready:
        return _not_ready_response(query, not_ready)

    total_start = time.perf_counter()
    results, context, retrieval_time = get_context(query)

    if mode == "certification":
        instruction = "Explain the BIS certification process in short numbered steps."
    elif mode == "compliance":
        instruction = "Explain BIS compliance requirements clearly and briefly."
    else:
        instruction = "Answer the user's BIS question clearly in a concise manner."

    prompt = f"""
You are BIS AI Assistant.

{instruction}

Rules:
- Use ONLY the provided BIS sources.
- Never invent facts.
- Never invent IS numbers.
- Never invent QCO information.
- If the sources do not contain the answer, say:
  "I could not find this information in the available BIS sources."
- Keep the answer under 150 words.
- Use simple language.

QUESTION:
{query}

BIS SOURCES:
{context}
"""

    try:
        client = get_client()
    except RuntimeError as error:
        return _error_response(query, results, retrieval_time, total_start, error)

    ai_start = time.perf_counter()

    try:
        response = client.responses.create(
            model=CHAT_MODEL,
            input=prompt,
            max_output_tokens=300
        )
    except Exception as error:
        return _error_response(query, results, retrieval_time, total_start, error)

    ai_time = time.perf_counter() - ai_start
    total_time = time.perf_counter() - total_start

    return {
        "question": query,
        "answer": response.output_text,
        "sources": get_sources(results),
        "timing": {
            "retrieval": round(retrieval_time, 2),
            "openai": round(ai_time, 2),
            "total": round(total_time, 2)
        }
    }


def ask_bis(query):
    return ask_ai(query, "assistant")


def certification_guide(query):
    return ask_ai(query, "certification")


def compliance_guide(query):
    return ask_ai(query, "compliance")


def search_standards(product):
    return ask_ai(product, "assistant")
