"""Evaluation harness for retrieval quality."""

import json
import os
from typing import Any, Dict, List

_DEFAULT_QA = os.path.join(os.path.dirname(__file__), "..", "tests", "sample_qa.json")


def evaluate(rag: Any, data_path: str = None) -> Dict[str, Any]:
    """Evaluate retrieval recall against a labeled QA dataset.

    The dataset is a JSON list of ``{"question": ..., "gold_source": ...}``
    objects. ``gold_source`` should match a chunk's ``source`` basename.
    """
    data_path = data_path or os.path.abspath(_DEFAULT_QA)
    if not os.path.exists(data_path):
        return {"error": f"QA file not found: {data_path}"}

    with open(data_path, encoding="utf-8") as handle:
        qa: List[Dict[str, Any]] = json.load(handle)

    total = len(qa)
    retrieval_hits = 0
    answered = 0
    for item in qa:
        candidates = rag.retrieve(item["question"], k=rag.config.top_k)
        sources = [c.get("source") for c in candidates]
        if candidates:
            answered += 1
        if item.get("gold_source") and item["gold_source"] in sources:
            retrieval_hits += 1

    return {
        "questions": total,
        "answered": answered,
        "retrieval_recall@k": round(retrieval_hits / total, 3) if total else 0.0,
    }
