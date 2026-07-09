"""Query expansion to improve retrieval recall.

Two strategies are supported:

* ``multi``  — generate several sub-queries that together cover the intent of
  the original question. With a real LLM configured this uses the model to
  decompose the question; otherwise a dependency-free lexical decomposition is
  used so the zero-download path still benefits.
* ``hyde``   — Hypothetical Document Embeddings: ask the LLM to draft a short
  passage that *would* answer the question, then retrieve against that passage
  (plus the original query). Requires an LLM; it degrades to the original query
  when only the offline ``extractive``/``dummy`` LLMs are available.
"""

import re
from typing import List, Optional

from .llm import LLMProvider

_STOPWORDS = {
    "a", "an", "the", "of", "for", "to", "in", "on", "and", "or", "is", "are",
    "what", "how", "why", "who", "when", "where", "which", "do", "does", "did",
    "can", "could", "should", "would", "with", "about", "that", "this",
}

_DECOMPOSE_PROMPT = (
    "Rewrite the question below as up to 3 shorter, independent sub-questions "
    "that together capture its meaning. Return one sub-question per line, no "
    "numbering.\n\nQUESTION: {query}\n\nSUB-QUESTIONS:"
)

_HYDE_PROMPT = (
    "Write a short, dense passage that would answer the question below. "
    "Be factual and specific; do not ask anything.\n\nQUESTION: {query}\n\nPASSAGE:"
)


class QueryExpander:
    """Produce query variants used to broaden retrieval."""

    def __init__(self, method: str = "none", llm: Optional[LLMProvider] = None):
        self.method = method
        self.llm = llm

    def expand(self, query: str) -> List[str]:
        """Return a list of query strings (always includes the original)."""
        query = (query or "").strip()
        if not query:
            return [query]
        if self.method == "multi":
            return self._multi(query)
        if self.method == "hyde":
            return self._hyde(query)
        return [query]

    def _can_generate(self) -> bool:
        name = type(self.llm).__name__
        return self.llm is not None and name not in ("ExtractiveLLM", "DummyLLM")

    def _multi(self, query: str) -> List[str]:
        variants = [query]
        if self._can_generate():
            prompt = _DECOMPOSE_PROMPT.format(query=query)
            try:
                out = self.llm.generate(prompt) or ""
            except Exception:
                out = ""
            for line in out.splitlines():
                line = line.strip().lstrip("0123456789.-) ").strip()
                if line and line not in variants:
                    variants.append(line)
        else:
            variants.extend(self._lexical_variants(query))
        return variants

    def _hyde(self, query: str) -> List[str]:
        if not self._can_generate():
            return [query]
        prompt = _HYDE_PROMPT.format(query=query)
        try:
            hypo = (self.llm.generate(prompt) or "").strip()
        except Exception:
            hypo = ""
        if not hypo:
            return [query]
        return [query, hypo]

    @staticmethod
    def _lexical_variants(query: str) -> List[str]:
        variants: List[str] = []
        lowered = query.lower()
        for conj in (" and ", " or ", " but ", " while "):
            if conj in lowered:
                for part in re.split(conj, lowered):
                    part = part.strip()
                    if part:
                        variants.append(part)
                break
        compact = " ".join(w for w in query.split() if w.lower() not in _STOPWORDS)
        if compact and compact != query:
            variants.append(compact)
        return [v for v in variants if v != query]
