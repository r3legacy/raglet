"""High-level RAG orchestration: ingest -> retrieve -> answer."""

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

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

    def ingest(
        self,
        path: str,
        glob: Optional[str] = None,
        reset: bool = False,
    ) -> int:
        """Load documents from ``path`` and index them. Returns chunk count.

        Args:
            path: A file or directory of documents to index.
            glob: Optional glob pattern when ``path`` is a directory.
            reset: When ``True``, wipe any existing index before ingesting.
        """
        if reset:
            self._reset_index()

        docs = loaders.load_documents(path, glob=glob)
        if not docs:
            return 0

        all_chunks: List[Dict[str, Any]] = []
        seen_hashes = {self._chunk_hash(chunk["text"]) for chunk in self.store.chunks}
        for doc in docs:
            for piece in chunk_text(doc["text"], self.config.chunk_size, self.config.overlap):
                chunk_hash = self._chunk_hash(piece)
                if chunk_hash in seen_hashes:
                    continue
                seen_hashes.add(chunk_hash)
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

    @staticmethod
    def _chunk_hash(text: str) -> str:
        """Stable content hash used to de-duplicate chunks across ingests."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _reset_index(self) -> None:
        """Drop the in-memory index and any persisted store on disk."""
        self.store = VectorStore()
        self._bm25_corpus = []
        self.bm25 = BM25()
        store_dir = self.config.store_path
        if os.path.isdir(store_dir):
            for name in ("chunks.json", "embeddings.npy", "config.json"):
                file_path = os.path.join(store_dir, name)
                if os.path.exists(file_path):
                    os.remove(file_path)

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve the most relevant chunks for ``query``.

        Args:
            query: The user question.
            k: Number of candidates to return (defaults to ``config.top_k``).
            filter: Optional predicate applied to each chunk; only chunks for
                which it returns ``True`` are considered.
        """
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

        if filter is not None:
            candidates = [c for c in candidates if filter(c)]
        if not candidates:
            return []

        if self._reranker is not None:
            candidates = self._reranker.rerank(query, candidates)[: self.config.rerank_top_n]
        else:
            candidates = candidates[:k]
        return candidates

    def ask(
        self,
        query: str,
        k: Optional[int] = None,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve context for ``query`` and generate an answer.

        Args:
            query: The user question.
            k: Number of candidates to retrieve.
            source: Optional source basename to limit retrieval to.
        """
        def chunk_filter(chunk: Dict[str, Any]) -> bool:
            return chunk.get("source") == source

        candidates = self.retrieve(query, k=k, filter=chunk_filter if source is not None else None)
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
        self._save_config()

    def _save_config(self) -> None:
        """Persist the pipeline config so an index can be reloaded correctly.

        The embedder/llm used to build the index are recorded so that
        :meth:`load` reconstructs providers matching the stored embeddings
        (dimensions must match). API keys are intentionally omitted from the
        persisted kwargs to avoid writing secrets to disk.
        """
        _RAG_CONFIG_FILE = "rag_config.json"
        config_path = os.path.join(self.config.store_path, _RAG_CONFIG_FILE)
        serializable = {
            "embedder": self.config.embedder,
            "embedder_kwargs": _redact_kwargs(self.config.embedder_kwargs),
            "llm": self.config.llm,
            "llm_kwargs": _redact_kwargs(self.config.llm_kwargs),
            "reranker": self.config.reranker,
            "reranker_kwargs": _redact_kwargs(self.config.reranker_kwargs),
            "use_rerank": self.config.use_rerank,
            "use_sparse": self.config.use_sparse,
            "chunk_size": self.config.chunk_size,
            "overlap": self.config.overlap,
            "top_k": self.config.top_k,
            "rerank_top_n": self.config.rerank_top_n,
        }
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(serializable, handle, ensure_ascii=False, indent=2)

    def load(self) -> bool:
        """Load a previously built index. Returns ``True`` if one was found.

        When a persisted pipeline config exists it is restored so the
        embedder/llm match the stored embeddings, overriding the defaults of
        the config passed to the constructor.
        """
        store_file = os.path.join(self.config.store_path, "chunks.json")
        if os.path.isdir(self.config.store_path) and os.path.exists(store_file):
            self._restore_config()
            self.store = VectorStore.load(self.config.store_path)
            if self.config.use_sparse:
                self._bm25_corpus = [chunk["text"] for chunk in self.store.chunks]
                self.bm25.fit(self._bm25_corpus)
            return True
        return False

    def _restore_config(self) -> None:
        """Rebuild providers from a persisted ``rag_config.json`` if present."""
        config_path = os.path.join(self.config.store_path, "rag_config.json")
        if not os.path.exists(config_path):
            return
        with open(config_path, encoding="utf-8") as handle:
            saved = json.load(handle)
        self.config.embedder = saved.get("embedder", self.config.embedder)
        self.config.embedder_kwargs = saved.get("embedder_kwargs", self.config.embedder_kwargs)
        self.config.llm = saved.get("llm", self.config.llm)
        self.config.llm_kwargs = saved.get("llm_kwargs", self.config.llm_kwargs)
        self.config.reranker = saved.get("reranker", self.config.reranker)
        self.config.reranker_kwargs = saved.get("reranker_kwargs", self.config.reranker_kwargs)
        self.config.use_rerank = saved.get("use_rerank", self.config.use_rerank)
        self.config.use_sparse = saved.get("use_sparse", self.config.use_sparse)
        self.config.chunk_size = saved.get("chunk_size", self.config.chunk_size)
        self.config.overlap = saved.get("overlap", self.config.overlap)
        self.config.top_k = saved.get("top_k", self.config.top_k)
        self.config.rerank_top_n = saved.get("rerank_top_n", self.config.rerank_top_n)

        self.embedder = get_embedder(self.config.embedder, **self.config.embedder_kwargs)
        self.llm = get_llm(self.config.llm, **self.config.llm_kwargs)
        self._reranker = None
        if self.config.use_rerank:
            self._reranker = get_reranker(self.config.reranker, **self.config.reranker_kwargs)

    def remove(self, source: str) -> int:
        """Remove every chunk whose ``source`` matches ``source``.

        Returns the number of chunks removed (0 if none matched). The index is
        rebuilt and persisted.
        """
        ids = [chunk["_id"] for chunk in self.store.chunks if chunk.get("source") == source]
        if not ids:
            return 0
        removed = self.store.remove(ids)
        if self.config.use_sparse:
            self._bm25_corpus = [chunk["text"] for chunk in self.store.chunks]
            if self._bm25_corpus:
                self.bm25.fit(self._bm25_corpus)
            else:
                self.bm25 = BM25()
        self.save()
        return removed


def _redact_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop secrets from provider kwargs before persisting to disk."""
    redacted = dict(kwargs)
    for secret in ("api_key", "token"):
        redacted.pop(secret, None)
    return redacted
