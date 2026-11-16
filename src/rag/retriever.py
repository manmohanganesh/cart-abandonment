from src.config import Config


def retrieve_context(vectorstore, query: str, k: int = None, config: Config = None):
    if k is None:
        k = config.get("rag.retriever.k", 3) if config else 3
    docs = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]
