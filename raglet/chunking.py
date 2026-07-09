"""Text chunking with overlap and optional sentence awareness."""

import re
from typing import List

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    split_by: str = "token",
) -> List[str]:
    """Split ``text`` into overlapping chunks.

    Args:
        text: Raw document text.
        chunk_size: Maximum number of tokens (words) per chunk.
        overlap: Number of trailing units carried into the next chunk.
        split_by: ``"token"`` (word based) or ``"sentence"`` (sentence based).

    Returns:
        A list of chunk strings.
    """
    text = (text or "").strip()
    if not text:
        return []

    if split_by == "sentence":
        units = [unit.strip() for unit in _SENTENCE_SPLIT.split(text) if unit.strip()]
    else:
        units = text.split()

    def _unit_tokens(unit: str) -> int:
        return len(unit.split())

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for unit in units:
        unit_tokens = _unit_tokens(unit)
        if current and current_len + unit_tokens > chunk_size:
            chunks.append(" ".join(current))
            if overlap > 0:
                tail = current[-overlap:] if overlap < len(current) else list(current)
            else:
                tail = []
            current = list(tail)
            current_len = sum(_unit_tokens(item) for item in current)
        current.append(unit)
        current_len += unit_tokens

    if current:
        chunks.append(" ".join(current))
    return chunks
