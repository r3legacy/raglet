"""raglet - a tiny, local-first RAG toolkit."""

from .chunking import chunk_text
from .core import RAG, RAGConfig
from .embeddings import EmbeddingProvider, get_embedder
from .eval import evaluate
from .llm import LLMProvider, get_llm
from .loaders import SUPPORTED_EXTENSIONS, load_document, load_documents
from .rerank import Reranker, get_reranker
from .store import VectorStore

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
    "evaluate",
    "__version__",
]
