"""raglet - a tiny, local-first RAG toolkit."""

from .core import RAG, RAGConfig
from .loaders import SUPPORTED_EXTENSIONS, load_document, load_documents
from .chunking import chunk_text
from .store import VectorStore
from .embeddings import EmbeddingProvider, get_embedder
from .llm import LLMProvider, get_llm
from .rerank import Reranker, get_reranker

__version__ = "0.1.0"

__all__ = [
    "RAG",
    "RAGConfig",
    "SUPPORTED_EXTENSIONS",
    "load_document",
    "load_documents",
    "chunk_text",
    "VectorStore",
    "EmbeddingProvider",
    "get_embedder",
    "LLMProvider",
    "get_llm",
    "Reranker",
    "get_reranker",
    "__version__",
]
