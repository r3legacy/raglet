"""Reciprocal Rank Fusion for combining multiple ranked result lists."""

from typing import Dict, List, Tuple


def reciprocal_rank_fusion(
    rankings: List[List[int]], k: int = 60
) -> List[Tuple[int, float]]:
    """Fuse several ranked lists of document ids into a single ranking.

    Args:
        rankings: Each element is an ordered list of document ids (best first).
        k: Smoothing constant that down-weights lower-ranked documents.

    Returns:
        A list of ``(doc_id, fused_score)`` tuples sorted by score descending.
    """
    scores: Dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
