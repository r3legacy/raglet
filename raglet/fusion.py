"""Reciprocal Rank Fusion for combining multiple ranked result lists."""

from typing import Dict, List, Optional, Sequence, Tuple


def reciprocal_rank_fusion(
    rankings: List[List[int]],
    weights: Optional[Sequence[float]] = None,
    k: int = 60,
) -> List[Tuple[int, float]]:
    """Fuse several ranked lists of document ids into a single ranking.

    Each ranking contributes ``weight / (k + rank + 1)`` for every position, so
    a signal with a larger ``weight`` (e.g. dense vs. sparse) can be emphasised
    without abandoning the rank-based robustness of RRF.

    Args:
        rankings: Each element is an ordered list of document ids (best first).
        weights: Optional per-ranking weights (default 1.0 each). Must match the
            length of ``rankings`` when provided.
        k: Smoothing constant that down-weights lower-ranked documents.

    Returns:
        A list of ``(doc_id, fused_score)`` tuples sorted by score descending.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("`weights` must have one entry per ranking")

    scores: Dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
