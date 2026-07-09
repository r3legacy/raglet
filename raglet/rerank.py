"""Rerankers that reorder retrieved candidates before generation."""

import re
from typing import Any, Dict, List, Optional

from .llm import LLMProvider

_RERANK_PROMPT = (
    "You are a strict relevance judge. Given the QUESTION and a CANDIDATE passage, "
    "rate how relevant the passage is for answering the question on a scale of "
    "0 to 100, where 0 means irrelevant and 100 means fully answers it. "
    "Reply with a single integer and nothing else.\n\n"
    "QUESTION: {query}\n\nCANDIDATE: {candidate}\n\nSCORE:"
)


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
    """LLM-based reranker.

    Scores each (query, candidate) pair with the configured LLM on a 0–100
    scale and reorders by that score. When no real LLM is available it degrades
    gracefully to ordering by the fused retrieval score so the pipeline still
    returns a sensible ranking.
    """

    def __init__(self, llm: Optional[LLMProvider] = None):
        self.llm = llm

    def _can_judge(self) -> bool:
        name = type(self.llm).__name__
        return self.llm is not None and name not in ("ExtractiveLLM", "DummyLLM")

    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self._can_judge():
            return sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True)

        scored: List[Dict[str, Any]] = []
        for candidate in candidates:
            prompt = _RERANK_PROMPT.format(query=query, candidate=candidate.get("text", ""))
            try:
                raw = self.llm.generate(prompt) or ""
            except Exception:
                raw = ""
            match = re.search(r"\d+", raw)
            score = float(match.group(0)) if match else 0.0
            scored.append((candidate, score))
        for candidate, score in scored:
            candidate["rerank_score"] = score
        return [c for c, _ in sorted(scored, key=lambda item: item[1], reverse=True)]


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
