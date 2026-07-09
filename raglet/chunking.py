"""Text chunking with overlap and optional sentence awareness."""

import re
from typing import Any, Dict, List

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


def chunk_parent_child(
    text: str,
    parent_size: int = 1500,
    child_size: int = 500,
    child_overlap: int = 50,
    split_by: str = "token",
) -> List[Dict[str, Any]]:
    """Split ``text`` into parent windows, each broken into child chunks.

    Returns a list of ``{"parent": <parent text>, "children": [<child>...]}``.
    Children are used for precise retrieval; the parent text is used as the
    richer generation context (the small-to-big pattern).
    """
    text = (text or "").strip()
    if not text:
        return []

    units = text.split()
    parents: List[str] = []
    current: List[str] = []
    current_len = 0
    for unit in units:
        if current and current_len + 1 > parent_size:
            parents.append(" ".join(current))
            current = []
            current_len = 0
        current.append(unit)
        current_len += 1
    if current:
        parents.append(" ".join(current))

    blocks: List[Dict[str, Any]] = []
    for parent in parents:
        children = chunk_text(
            parent, chunk_size=child_size, overlap=child_overlap, split_by=split_by
        )
        if children:
            blocks.append({"parent": parent, "children": children})
    return blocks
