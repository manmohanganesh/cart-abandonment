from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from src.config import Config


def load_documents(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return text


def create_vector_store(text: str, config: Config):
    rag_config = config.get("rag", {})

    embeddings = HuggingFaceEmbeddings(
        model_name = rag_config.get(
            "embeddings.model_name", 
            "sentence-transformers/all-MiniLM-L6-v2")
    )

    splitter_config = rag_config.get("text_splitter", {})
    splitter = CharacterTextSplitter(
        separator=splitter_config.get("separator", "\n"),
        chunk_size=splitter_config.get("chunk_size", 500),
        chunk_overlap=splitter_config.get("chunk_overlap", 100),
    )
    docs = splitter.split_text(text)

    vectorstore_config = rag_config.get("vectorstore", {})
    persist_dir = vectorstore_config.get("persist_directory")

    if persist_dir:
        vectorstore = Chroma.from_texts(
            docs,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_name=vectorstore_config.get("collection_name", "cart_abandonment_reviews"),
        )
    else:
        vectorstore = Chroma.from_texts(docs, embedding=embeddings)

    return vectorstore