"""High-level RAG orchestration: ingest -> retrieve -> answer."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import loaders
from .chunking import chunk_text
from .embeddings import EmbeddingProvider, get_embedder
from .fusion import reciprocal_rank_fusion
from .llm import LLMProvider, build_prompt, get_llm
from .rerank import Reranker, get_reranker
from .sparse import BM25
from .store import VectorStore

DEFAULT_STORE = os.path.join(os.path.expanduser("~"), ".raglet", "store")


@dataclass
class RAGConfig:
    """Configuration for a :class:`RAG` instance."""

    chunk_size: int = 500
    overlap: int = 50
    embedder: str = "hash"
    embedder_kwargs: Dict[str, Any] = field(default_factory=dict)
    llm: str = "extractive"
    llm_kwargs: Dict[str, Any] = field(default_factory=dict)
    store_path: str = DEFAULT_STORE
    use_sparse: bool = True
    use_rerank: bool = False
    reranker: str = "score"
    reranker_kwargs: Dict[str, Any] = field(default_factory=dict)
    top_k: int = 5
    rerank_top_n: int = 5


class RAG:
    """A tiny, local-first retrieval-augmented generation pipeline."""

    def __init__(self, config: Optional[RAGConfig] = None, **kwargs: Any):
        self.config = config or RAGConfig()
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self.embedder: EmbeddingProvider = get_embedder(
            self.config.embedder, **self.config.embedder_kwargs
        )
        self.llm: LLMProvider = get_llm(self.config.llm, **self.config.llm_kwargs)
        self.store = VectorStore()
        self.bm25 = BM25()
        self._bm25_corpus: List[str] = []
        self._reranker: Optional[Reranker] = None
        if self.config.use_rerank:
            self._reranker = get_reranker(self.config.reranker, **self.config.reranker_kwargs)

    def ingest(self, path: str, glob: Optional[str] = None) -> int:
        """Load documents from ``path`` and index them. Returns chunk count."""
        docs = loaders.load_documents(path, glob=glob)
        if not docs:
            return 0

        all_chunks: List[Dict[str, Any]] = []
        for doc in docs:
            for piece in chunk_text(doc["text"], self.config.chunk_size, self.config.overlap):
                all_chunks.append(
                    {
                        "text": piece,
                        "source": doc["source"],
                        "metadata": doc.get("metadata", {}),
                    }
                )
        if not all_chunks:
            return 0

        embeddings = self.embedder.embed([chunk["text"] for chunk in all_chunks])
        self.store.add(all_chunks, embeddings)

        if self.config.use_sparse:
            self._bm25_corpus = [chunk["text"] for chunk in self.store.chunks]
            self.bm25.fit(self._bm25_corpus)

        self.save()
        return len(all_chunks)

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve the most relevant chunks for ``query``."""
        k = k or self.config.top_k
        rankings: List[List[int]] = []

        query_vec = self.embedder.embed_query(query)
        dense = self.store.search(query_vec, k=max(k, 10) if self.config.use_sparse else k)
        if dense:
            rankings.append([chunk["_id"] for chunk in dense])

        if self.config.use_sparse and self._bm25_corpus:
            sparse = self.bm25.search(query, k=max(k, 10))
            rankings.append([index for index, _ in sparse])

        if not rankings:
            return []

        fused = reciprocal_rank_fusion(rankings)
        candidates: List[Dict[str, Any]] = []
        seen: set = set()
        for doc_id, score in fused:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            record = dict(self.store.chunks[doc_id])
            record["score"] = score
            candidates.append(record)

        if self._reranker is not None:
            candidates = self._reranker.rerank(query, candidates)[: self.config.rerank_top_n]
        else:
            candidates = candidates[:k]
        return candidates

    def ask(self, query: str, k: Optional[int] = None) -> Dict[str, Any]:
        """Retrieve context for ``query`` and generate an answer."""
        candidates = self.retrieve(query, k=k)
        if not candidates:
            return {"answer": "(no relevant context found)", "sources": [], "context": []}
        prompt = build_prompt(query, candidates)
        answer = self.llm.generate(prompt, context_chunks=candidates)
        sources = [
            {"source": chunk.get("source"), "score": chunk.get("score")}
            for chunk in candidates
        ]
        return {"answer": answer, "sources": sources, "context": candidates}

    def save(self) -> None:
        self.store.save(self.config.store_path)

    def load(self) -> bool:
        """Load a previously built index. Returns ``True`` if one was found."""
        store_file = os.path.join(self.config.store_path, "chunks.json")
        if os.path.isdir(self.config.store_path) and os.path.exists(store_file):
            self.store = VectorStore.load(self.config.store_path)
            if self.config.use_sparse:
                self._bm25_corpus = [chunk["text"] for chunk in self.store.chunks]
                self.bm25.fit(self._bm25_corpus)
            return True
        return False
