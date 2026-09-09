from backend.vector_store import embed_query, collection


def retrieve_documents(query: str, n_results: int = 5):

    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved = []

    for i, document in enumerate(documents):

        metadata = (
            metadatas[i]
            if i < len(metadatas)
            else {}
        )

        distance = (
            distances[i]
            if i < len(distances)
            else None
        )

        retrieved.append({
            "content": document,
            "metadata": metadata,
            "distance": distance
        })

    return retrieved


def build_context(retrieved_documents):

    context_parts = []

    for index, item in enumerate(retrieved_documents, start=1):

        content = item["content"]

        filename = item["metadata"].get(
            "filename",
            "BIS Document"
        )

        context_parts.append(
            f"""
SOURCE {index}
DOCUMENT: {filename}

{content}
"""
        )

    return "\n".join(context_parts)


def get_source_names(retrieved_documents):

    sources = []

    for item in retrieved_documents:

        filename = item["metadata"].get("filename")

        if filename and filename not in sources:
            sources.append(filename)

    return sources