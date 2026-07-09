"""Embedding providers with a dependency-free default.

The ``hash`` embedder needs no models and is deterministic, which makes it
ideal for tests and for a zero-download first run. For real semantic
retrieval use ``local`` (sentence-transformers) or a cloud provider.
"""

import hashlib
import math
import re
from typing import Any, Dict, List, Optional

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class EmbeddingProvider:
    """Base class for embedding backends."""

    dim: int = 0

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0]


class HashEmbedding(EmbeddingProvider):
    """Deterministic, dependency-free embeddings (signed hashed bag-of-words)."""

    def __init__(self, dim: int = 256, normalize: bool = True):
        self.dim = dim
        self.normalize = normalize

    def embed(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for text in texts:
            vector = [0.0] * self.dim
            for token in _tokens(text):
                digest = int(hashlib.md5(token.encode()).hexdigest(), 16)
                index = digest % self.dim
                sign = 1.0 if (digest >> 7) & 1 else -1.0
                vector[index] += sign
            if self.normalize:
                norm = math.sqrt(sum(value * value for value in vector))
                if norm > 0:
                    vector = [value / norm for value in vector]
            out.append(vector)
        return out


class LocalHFEmbedding(EmbeddingProvider):
    """Local embeddings via sentence-transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Local embeddings require sentence-transformers: "
                "pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI text embeddings."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("OpenAI embeddings require openai: pip install openai") from exc
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.dim = 1536 if "3-small" in model else 3072

    def embed(self, texts: List[str]) -> List[List[float]]:
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


class OllamaEmbedding(EmbeddingProvider):
    """Local embeddings served by Ollama's REST API."""

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        try:
            import requests  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Ollama embeddings require requests: pip install requests") from exc
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.dim = 0

    def embed(self, texts: List[str]) -> List[List[float]]:
        import requests

        out: List[List[float]] = []
        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=30,
            )
            response.raise_for_status()
            out.append(response.json()["embedding"])
        if out and self.dim == 0:
            self.dim = len(out[0])
        return out


_EMBEDDERS: Dict[str, type] = {
    "hash": HashEmbedding,
    "local": LocalHFEmbedding,
    "openai": OpenAIEmbedding,
    "ollama": OllamaEmbedding,
}


def get_embedder(name: str = "hash", **kwargs: Any) -> EmbeddingProvider:
    """Return an embedding provider by name."""
    if name not in _EMBEDDERS:
        raise ValueError(f"Unknown embedder '{name}'. Choose from {list(_EMBEDDERS)}")
    return _EMBEDDERS[name](**kwargs)
