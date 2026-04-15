def retrieve_context(vectorstore,query,k=3):
    docs = vectorstore.similarity_search(query,k=k)
    return [doc.page_content for doc in docs]
