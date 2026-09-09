import os
import time
import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=30.0
)

chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_collection(
    name="bis_documents"
)


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

    results = collection.query(
        query_texts=[query],
        n_results=3
    )

    retrieval_time = time.perf_counter() - start

    documents = results["documents"][0]

    context = "\n\n".join(
        f"SOURCE {i + 1}: {doc}"
        for i, doc in enumerate(documents)
    )

    return results, context, retrieval_time


def ask_ai(query, mode="assistant"):

    total_start = time.perf_counter()

    results, context, retrieval_time = get_context(query)

    if mode == "certification":
        instruction = """
Explain the BIS certification process in short numbered steps.
Only include information supported by the sources.
"""

    elif mode == "compliance":
        instruction = """
Explain BIS compliance requirements clearly and briefly.
Only include information supported by the sources.
"""

    else:
        instruction = """
Answer the user's BIS question clearly in a concise manner.
"""

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
- Do not explain your reasoning.
- Do not mention internal AI processes.

QUESTION:
{query}

BIS SOURCES:
{context}
"""

    openai_start = time.perf_counter()

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt,
        reasoning={
            "effort": "none"
        },
        max_output_tokens=300
    )

    openai_time = time.perf_counter() - openai_start
    total_time = time.perf_counter() - total_start

    print("\n" + "=" * 50)
    print(f"⏱ Retrieval : {retrieval_time:.2f}s")
    print(f"⏱ OpenAI    : {openai_time:.2f}s")
    print(f"⏱ Total     : {total_time:.2f}s")
    print("=" * 50 + "\n")

    return {
        "question": query,
        "answer": response.output_text,
        "sources": get_sources(results),
        "timing": {
            "retrieval": round(retrieval_time, 2),
            "openai": round(openai_time, 2),
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

    total_start = time.perf_counter()

    results, context, retrieval_time = get_context(product)

    prompt = f"""
You are a BIS Standards Search Assistant.

Product:
{product}

Rules:
- Use ONLY the provided BIS sources.
- Never guess an IS number.
- Never invent a standard title.
- Mention an IS number only if directly present.
- If no specific standard is confirmed, clearly say so.
- Keep the answer under 120 words.

BIS SOURCES:
{context}
"""

    openai_start = time.perf_counter()

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt,
        reasoning={
            "effort": "none"
        },
        max_output_tokens=250
    )

    openai_time = time.perf_counter() - openai_start
    total_time = time.perf_counter() - total_start

    print("\n" + "=" * 50)
    print(f"⏱ Retrieval : {retrieval_time:.2f}s")
    print(f"⏱ OpenAI    : {openai_time:.2f}s")
    print(f"⏱ Total     : {total_time:.2f}s")
    print("=" * 50 + "\n")

    return {
        "product": product,
        "answer": response.output_text,
        "sources": get_sources(results),
        "official_portal": "https://standards.bis.gov.in/",
        "timing": {
            "retrieval": round(retrieval_time, 2),
            "openai": round(openai_time, 2),
            "total": round(total_time, 2)
        }
    }