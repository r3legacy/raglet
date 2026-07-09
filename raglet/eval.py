"""Evaluation harness for retrieval quality."""

import json
import math
import os
import re
from typing import Any, Dict, List, Optional

# Bundled with the package so `raglet eval` works without --data even when
# installed from a wheel (tests/ is not shipped in the distribution).
_DEFAULT_QA = os.path.join(os.path.dirname(__file__), "sample_qa.json")


def _load_qa(data_path: str) -> List[Dict[str, Any]]:
    """Load QA items from a JSON file or a directory of JSON files."""
    if os.path.isdir(data_path):
        items: List[Dict[str, Any]] = []
        for entry in sorted(os.listdir(data_path)):
            if entry.endswith(".json"):
                with open(os.path.join(data_path, entry), encoding="utf-8") as handle:
                    items.extend(json.load(handle))
        return items
    with open(data_path, encoding="utf-8") as handle:
        return json.load(handle)


def _ndcg_at_k(rels: List[int], k: int) -> float:
    """Binary nDCG@k for a ranked relevance list (1=relevant, 0=not)."""
    rels = rels[:k]
    if not rels:
        return 0.0
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    ideal = sorted(rels, reverse=True)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(rag: Any, data_path: str = None, k: int = None) -> Dict[str, Any]:
    """Evaluate retrieval quality against a labeled QA dataset.

    The dataset is a JSON list of ``{"question": ..., "gold_source": ...}``
    objects. ``gold_source`` should match a chunk's ``source`` basename and may
    be a single string or a list of strings (multiple acceptable answers).

    Returns recall@k, precision@k, nDCG@k and MRR over the labeled questions.
    """
    data_path = data_path or os.path.abspath(_DEFAULT_QA)
    if not os.path.exists(data_path):
        return {"error": f"QA file not found: {data_path}"}

    qa = _load_qa(data_path)

    k = k or rag.config.top_k
    total = len(qa)
    retrieval_hits = 0
    precision_sum = 0.0
    mrr_sum = 0.0
    ndcg_sum = 0.0
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
            rels = [1 if s in gold_set else 0 for s in sources]
            ndcg_sum += _ndcg_at_k(rels, k)

    return {
        "questions": total,
        "answered": answered,
        "k": k,
        "retrieval_recall@k": round(retrieval_hits / total, 3) if total else 0.0,
        "precision@k": round(precision_sum / total, 3) if total else 0.0,
        "ndcg@k": round(ndcg_sum / total, 3) if total else 0.0,
        "mrr": round(mrr_sum / total, 3) if total else 0.0,
    }


_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return set(_TOKEN.findall((text or "").lower()))


def _faithfulness(answer: str, context: str) -> float:
    """Offline groundedness: fraction of answer tokens present in context.

    Returns 0.0 for an empty answer, 1.0 when every answer token appears in the
    retrieved context.
    """
    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return 0.0
    context_tokens = _tokens(context)
    overlap = len(answer_tokens & context_tokens)
    return overlap / len(answer_tokens)


def evaluate_answers(
    rag: Any,
    data_path: str = None,
    k: int = None,
    judge: str = "offline",
) -> Dict[str, Any]:
    """Evaluate *generated* answers, not just retrieval.

    The dataset is a JSON list of objects with at least ``question`` and
    optionally ``gold_answer`` (and/or ``gold_source``). For each question the
    pipeline generates an answer and we measure:

    * ``faithfulness`` — offline token overlap between the answer and the
      retrieved context (does the answer stay grounded in the sources?).
    * ``groundedness`` — when ``judge="llm"``, the configured LLM rates support
      on a 1–5 scale (falls back to offline faithfulness otherwise).
    * ``answer_recall`` — overlap between the answer and ``gold_answer``.

    Returns aggregated metrics.
    """
    data_path = data_path or os.path.abspath(_DEFAULT_QA)
    if not os.path.exists(data_path):
        return {"error": f"QA file not found: {data_path}"}

    qa = _load_qa(data_path)

    k = k or rag.config.top_k
    total = len(qa)
    faith_sum = 0.0
    grounded_sum = 0.0
    grounded_count = 0
    recall_sum = 0.0
    has_gold_answer = 0
    answered = 0

    for item in qa:
        result = rag.ask(item["question"], k=k)
        answer = result.get("answer", "")
        if answer:
            answered += 1
        context = "\n".join(c.get("text", "") for c in result.get("context", []))
        faith = _faithfulness(answer, context)
        faith_sum += faith

        gold_answer = item.get("gold_answer")
        if gold_answer:
            has_gold_answer += 1
            ans_tokens = _tokens(answer)
            gold_tokens = _tokens(gold_answer)
            union = ans_tokens | gold_tokens
            recall_sum += len(ans_tokens & gold_tokens) / len(union) if union else 0.0

        if judge == "llm":
            score = _llm_groundedness(rag, answer, context)
            if score is not None:
                grounded_sum += score
                grounded_count += 1
        else:
            grounded_sum += faith
            grounded_count += 1

    return {
        "questions": total,
        "answered": answered,
        "k": k,
        "faithfulness": round(faith_sum / total, 3) if total else 0.0,
        "avg_groundedness": round(grounded_sum / grounded_count, 3) if grounded_count else 0.0,
        "answer_recall": round(recall_sum / has_gold_answer, 3) if has_gold_answer else None,
    }


def _llm_groundedness(rag: Any, answer: str, context: str) -> Optional[float]:
    """Ask the configured LLM to rate 1–5 how well the answer is supported."""
    if answer and not context:
        return 0.0
    if not answer:
        return 0.0
    prompt = (
        "Rate how well the ANSWER is supported by the CONTEXT on a scale of "
        "1 (unsupported) to 5 (fully supported). Reply with a single integer.\n\n"
        f"CONTEXT:\n{context[:4000]}\n\nANSWER:\n{answer}\n\nSCORE:"
    )
    try:
        raw = rag.llm.generate(prompt) or ""
    except Exception:
        return None
    for token in re.findall(r"[1-5]", raw):
        return float(token)
    return None
