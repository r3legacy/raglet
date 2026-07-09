"""Evaluation harness for retrieval quality."""

import json
import os
from typing import Any, Dict, List

_DEFAULT_QA = os.path.join(os.path.dirname(__file__), "..", "tests", "sample_qa.json")


def evaluate(rag: Any, data_path: str = None, k: int = None) -> Dict[str, Any]:
    """Evaluate retrieval quality against a labeled QA dataset.

    The dataset is a JSON list of ``{"question": ..., "gold_source": ...}``
    objects. ``gold_source`` should match a chunk's ``source`` basename and may
    be a single string or a list of strings (multiple acceptable answers).

    Returns recall@k, precision@k and MRR over the labeled questions.
    """
    data_path = data_path or os.path.abspath(_DEFAULT_QA)
    if not os.path.exists(data_path):
        return {"error": f"QA file not found: {data_path}"}

    with open(data_path, encoding="utf-8") as handle:
        qa: List[Dict[str, Any]] = json.load(handle)

    k = k or rag.config.top_k
    total = len(qa)
    retrieval_hits = 0
    precision_sum = 0.0
    mrr_sum = 0.0
    answered = 0
    for item in qa:
        candidates = rag.retrieve(item["question"], k=k)
        sources = [c.get("source") for c in candidates]
        if candidates:
            answered += 1
        gold = item.get("gold_source")
        gold_set = {gold} if isinstance(gold, str) else set(gold or [])
        if gold_set:
            found = [s for s in sources if s in gold_set]
            if found:
                retrieval_hits += 1
                best_rank = min(sources.index(s) + 1 for s in found)
                mrr_sum += 1.0 / best_rank
            precision_sum += len(found) / max(k, 1)

    return {
        "questions": total,
        "answered": answered,
        "k": k,
        "retrieval_recall@k": round(retrieval_hits / total, 3) if total else 0.0,
        "precision@k": round(precision_sum / total, 3) if total else 0.0,
        "mrr": round(mrr_sum / total, 3) if total else 0.0,
    }
