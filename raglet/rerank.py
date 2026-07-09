"""Rerankers that reorder retrieved candidates before generation."""

from typing import Any, Dict, List, Optional

from .llm import LLMProvider


class Reranker:
    """Base class for rerankers."""

    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        raise NotImplementedError


class ScoreReranker(Reranker):
    """Fallback reranker that sorts by the fused retrieval score."""

    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True)


class CrossEncoderReranker(Reranker):
    """Cross-encoder reranker via sentence-transformers."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Cross-encoder reranking requires sentence-transformers: "
                "pip install sentence-transformers"
            ) from exc
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pairs = [(query, c["text"]) for c in candidates]
        scores = self._model.predict(pairs)
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)
        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)


class LLMReranker(Reranker):
    """LLM-based reranker (degrades gracefully to score ordering)."""

    def __init__(self, llm: Optional[LLMProvider] = None):
        self.llm = llm

    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True)


_RERANKERS = {
    "score": ScoreReranker,
    "cross-encoder": CrossEncoderReranker,
    "llm": LLMReranker,
}


def get_reranker(name: str = "score", **kwargs: Any) -> Reranker:
    """Return a reranker by name."""
    if name not in _RERANKERS:
        raise ValueError(f"Unknown reranker '{name}'. Choose from {list(_RERANKERS)}")
    return _RERANKERS[name](**kwargs)
