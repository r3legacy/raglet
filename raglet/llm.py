"""LLM providers and prompt construction."""

import re
from typing import Any, Dict, List, Optional

_TOKEN_SPLIT = re.compile(r"\s+")


class LLMProvider:
    """Base class for answer-generation backends."""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        raise NotImplementedError


class ExtractiveLLM(LLMProvider):
    """Offline baseline: returns the chunk most relevant to the query.

    Relevance is the lexical overlap (Jaccard over token sets) between the
    query and each candidate chunk, falling back to the retrieval score when
    available. This keeps raglet fully functional with zero model downloads.
    """

    def generate(self, prompt: str, context_chunks: Optional[List[Dict[str, Any]]] = None, **kwargs: Any) -> str:
        if not context_chunks:
            return "(no context retrieved)"

        query = _query_from_prompt(prompt)
        query_tokens = set(_TOKEN_SPLIT.split(query.lower()))

        best_chunk = None
        best_score = float("-inf")
        for chunk in context_chunks:
            text_tokens = set(_TOKEN_SPLIT.split(chunk.get("text", "").lower()))
            overlap = len(query_tokens & text_tokens)
            union = len(query_tokens | text_tokens) or 1
            lexical = overlap / union
            retrieved = float(chunk.get("score", 0.0) or 0.0)
            combined = 0.7 * lexical + 0.3 * _normalize(retrieved)
            if combined > best_score:
                best_score = combined
                best_chunk = chunk
        return best_chunk["text"]


def _query_from_prompt(prompt: str) -> str:
    """Extract the QUESTION portion from a prompt built by ``build_prompt``."""
    marker = "QUESTION:"
    index = prompt.rfind(marker)
    if index != -1:
        return prompt[index + len(marker):].split("ANSWER:")[0].strip()
    return prompt


def _normalize(value: float) -> float:
    """Map an arbitrary retrieval score into a 0..1 range for blending."""
    if value <= 0:
        return 0.0
    if value <= 1:
        return value
    return 1.0 / (1.0 + abs(value - 1.0))


class DummyLLM(LLMProvider):
    """Echoes the prompt; useful for debugging pipelines."""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return f"[dummy] {prompt[:160]}"


class OllamaLLM(LLMProvider):
    """Local generation via Ollama's REST API."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        system: Optional[str] = None,
    ):
        try:
            import requests  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Ollama requires requests: pip install requests") from exc
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.system = system

    def generate(self, prompt: str, **kwargs: Any) -> str:
        import requests

        payload: Dict[str, Any] = {"model": self.model, "prompt": prompt, "stream": False}
        if self.system:
            payload["system"] = self.system
        response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "")


class OpenAILLM(LLMProvider):
    """OpenAI chat completions backend."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        system: Optional[str] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("OpenAI requires openai: pip install openai") from exc
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.system = system

    def generate(self, prompt: str, **kwargs: Any) -> str:
        messages: List[Dict[str, str]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})
        response = self._client.chat.completions.create(model=self.model, messages=messages)
        return response.choices[0].message.content or ""


class AnthropicLLM(LLMProvider):
    """Anthropic Messages API backend."""

    def __init__(
        self,
        model: str = "claude-3-5-haiku-latest",
        api_key: Optional[str] = None,
        system: Optional[str] = None,
    ):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Anthropic requires anthropic: pip install anthropic") from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.system = system

    def generate(self, prompt: str, **kwargs: Any) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self.system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


_LLMS = {
    "extractive": ExtractiveLLM,
    "dummy": DummyLLM,
    "ollama": OllamaLLM,
    "openai": OpenAILLM,
    "anthropic": AnthropicLLM,
}


def get_llm(name: str = "extractive", **kwargs: Any) -> LLMProvider:
    """Return an LLM provider by name."""
    if name not in _LLMS:
        raise ValueError(f"Unknown LLM '{name}'. Choose from {list(_LLMS)}")
    return _LLMS[name](**kwargs)


def build_prompt(query: str, chunks: List[Dict[str, Any]], max_context_chars: int = 6000) -> str:
    """Build a grounded prompt that cites the retrieved sources."""
    parts: List[str] = []
    used = 0
    for index, chunk in enumerate(chunks):
        snippet = chunk["text"]
        if used + len(snippet) > max_context_chars:
            snippet = snippet[: max(0, max_context_chars - used)]
        parts.append(f"[Source {index + 1}] {chunk.get('source', 'unknown')}\n{snippet}")
        used += len(snippet)
        if used >= max_context_chars:
            break
    context = "\n\n".join(parts)
    return (
        "Answer the question using ONLY the context below. "
        "Cite the source filenames. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:"
    )
