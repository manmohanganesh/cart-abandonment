"""Shared service layer for Streamlit app - singleton services with caching."""

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import load_config
from src.models.predict import load_model
from src.rag.ingest import create_vector_store, load_documents
from src.rag.retriever import retrieve_context

_config = None
_model = None
_columns = None
_vectorstore = None


def get_config():
    """Get or create config singleton."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_model():
    """Get or create model singleton."""
    global _model, _columns
    if _model is None or _columns is None:
        _model, _columns = load_model()
    return _model, _columns


def get_vectorstore():
    """Get or create vectorstore singleton with persistence."""
    global _vectorstore
    config = get_config()

    if _vectorstore is None:
        persist_dir = config.get("rag.vectorstore.persist_directory")
        if persist_dir:
            Path(persist_dir).mkdir(parents=True, exist_ok=True)

        docs = load_documents(config.get("paths.reviews_file", "data/external/reviews.txt"))
        _vectorstore = create_vector_store(docs, config)

    return _vectorstore


def get_rag_context(query: str, k: int = None):
    """Retrieve RAG context using shared vectorstore."""
    vectorstore = get_vectorstore()
    config = get_config()
    return retrieve_context(vectorstore, query, k=k, config=config)


def clear_cache():
    """Clear all cached services (for testing)."""
    global _config, _model, _columns, _vectorstore
    _config = None
    _model = None
    _columns = None
    _vectorstore = None


# Streamlit-specific cached versions (import streamlit lazily)
def _get_streamlit():
    import streamlit as st
    return st


@_get_streamlit().cache_resource
def get_model_cached():
    """Streamlit-cached model loader."""
    return load_model()
    

@_get_streamlit().cache_resource
def get_vectorstore_cached():
    """Streamlit-cached vectorstore loader with persistence."""
    config = load_config()
    persist_dir = config.get("rag.vectorstore.persist_directory")
    if persist_dir:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

    docs = load_documents(config.get("paths.reviews_file", "data/external/reviews.txt"))
    return create_vector_store(docs, config)


@_get_streamlit().cache_resource
def get_config_cached():
    """Streamlit-cached config loader."""
    return load_config()