"""Vector store backed by numpy, with optional FAISS acceleration."""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import faiss

    _HAS_FAISS = True
except Exception:  # pragma: no cover - optional dependency
    faiss = None
    _HAS_FAISS = False


class VectorStore:
    """Stores chunk embeddings and performs cosine similarity search."""

    def __init__(self) -> None:
        self.chunks: List[Dict[str, Any]] = []
        self._emb: Optional[np.ndarray] = None
        self._index = None
        self.dim = 0

    def add(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        if not embeddings:
            return
        array = np.asarray(embeddings, dtype="float32")
        if array.ndim != 2:
            raise ValueError("embeddings must be a 2D list")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        array = array / norms

        if self._emb is None:
            self._emb = array
            self.dim = array.shape[1]
        else:
            self._emb = np.vstack([self._emb, array])
            self.dim = self._emb.shape[1]

        start = len(self.chunks)
        for offset, chunk in enumerate(chunks):
            record = dict(chunk)
            record["_id"] = start + offset
            self.chunks.append(record)
        self._build_index()

    def _build_index(self) -> None:
        if _HAS_FAISS and self._emb is not None and self._emb.shape[0] > 0:
            self._index = faiss.IndexFlatIP(self.dim)
            self._index.add(self._emb)
        else:
            self._index = None

    def search(
        self, query_vec: List[float], k: int = 5, allowed: Optional[set] = None
    ) -> List[Dict[str, Any]]:
        if self._emb is None or len(self.chunks) == 0:
            return []
        query = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        k = min(k, self._emb.shape[0])

        if allowed is None:
            if self._index is not None:
                scores, indices = self._index.search(query, k)
                indices = indices[0].tolist()
                scores = scores[0].tolist()
            else:
                sims = self._emb @ query[0]
                order = np.argsort(-sims)[:k]
                indices = order.tolist()
                scores = sims[order].tolist()
        else:
            # Restrict the search to the allowed chunk ids (used for metadata
            # filtering) by scoring only those rows with numpy.
            allowed_rows = [(i, c) for i, c in enumerate(self.chunks) if c.get("_id") in allowed]
            if not allowed_rows:
                return []
            rows = np.asarray([self._emb[i] for i, _ in allowed_rows], dtype="float32")
            sims = rows @ query[0]
            order = np.argsort(-sims)[:k]
            indices = [allowed_rows[i][0] for i in order]
            scores = [float(sims[i]) for i in order]

        results: List[Dict[str, Any]] = []
        for index, score in zip(indices, scores):
            if index < 0:
                continue
            record = dict(self.chunks[index])
            record["score"] = float(score)
            results.append(record)
        return results

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "embeddings.npy"), self._emb)
        meta = [{k: v for k, v in c.items() if k != "_id"} for c in self.chunks]
        with open(os.path.join(path, "chunks.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as handle:
            json.dump({"dim": self.dim, "faiss": _HAS_FAISS}, handle)

    def remove(self, ids: List[int]) -> int:
        """Remove chunks by their ``_id`` and return the count removed.

        Rebuilds the embedding matrix and search index after deletion. Chunk
        ``_id`` values are renumbered sequentially and stay aligned with the
        embedding matrix rows.
        """
        if not ids:
            return 0
        drop = set(ids)
        kept = [chunk for chunk in self.chunks if chunk.get("_id") not in drop]
        removed = len(self.chunks) - len(kept)
        if removed == 0:
            return 0

        keep_rows = []
        self.chunks = []
        for new_id, chunk in enumerate(kept):
            record = dict(chunk)
            record["_id"] = new_id
            self.chunks.append(record)
            keep_rows.append(new_id)

        if self._emb is not None and self._emb.shape[0] > 0:
            self._emb = self._emb[keep_rows]
        else:
            self._emb = None
        self._build_index()
        return removed

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        store = cls()
        store._emb = np.load(os.path.join(path, "embeddings.npy"))
        with open(os.path.join(path, "chunks.json"), encoding="utf-8") as handle:
            chunks = json.load(handle)
        store.dim = store._emb.shape[1]
        store.chunks = []
        for offset, chunk in enumerate(chunks):
            record = dict(chunk)
            record["_id"] = offset
            store.chunks.append(record)
        store._build_index()
        return store
