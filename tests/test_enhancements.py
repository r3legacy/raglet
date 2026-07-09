"""Tests for the advanced enhancements: fusion weights, LLM reranker, metadata
filtering, confidence/citations, and streaming."""

import shutil
from pathlib import Path

from raglet.core import RAG, RAGConfig
from raglet.fusion import reciprocal_rank_fusion
from raglet.llm import DummyLLM, LLMProvider
from raglet.rerank import LLMReranker
from raglet.store import VectorStore

FIXTURES = Path(__file__).parent / "fixtures"


def _copy_fixtures(dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for file in FIXTURES.glob("*.txt"):
        shutil.copy(file, dst / file.name)


def test_fusion_respects_weights():
    # Doc 5 leads the high-weight ranking; it must win overall.
    rankings = [[5, 1], [2, 3]]
    fused = reciprocal_rank_fusion(rankings, weights=[100.0, 1.0], k=60)
    assert fused[0][0] == 5


def test_fusion_default_weights_equal():
    rankings = [[3, 1, 2], [1, 2, 3]]
    fused = reciprocal_rank_fusion(rankings)
    # With equal weights the union of top items still appears, fused scores > 0.
    ids = {doc_id for doc_id, _ in fused}
    assert ids == {1, 2, 3}


def test_llm_reranker_degrades_without_real_llm():
    reranker = LLMReranker(llm=DummyLLM())
    cands = [{"text": "a", "score": 0.1}, {"text": "b", "score": 0.9}]
    out = reranker.rerank("q", cands)
    # No real LLM -> falls back to score ordering (highest first).
    assert out[0]["score"] == 0.9
    assert "rerank_score" not in out[0]


def test_llm_reranker_uses_llm_scores():
    class Judge(LLMProvider):
        def generate(self, prompt, **kwargs):
            candidate = prompt.split("CANDIDATE:")[1]
            return str(len(candidate))

    reranker = LLMReranker(llm=Judge())
    cands = [{"text": "short", "score": 0.9}, {"text": "much longer text", "score": 0.1}]
    out = reranker.rerank("q", cands)
    # LLM scores by length: longer candidate should rank first.
    assert out[0]["text"] == "much longer text"
    assert "rerank_score" in out[0]


def test_store_search_allowed_filter():
    store = VectorStore()
    store.add(
        [{"text": "alpha", "source": "a"}, {"text": "beta", "source": "b"}],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    allowed = {store.chunks[1]["_id"]}
    results = store.search([0.0, 1.0], k=5, allowed=allowed)
    assert len(results) == 1
    assert results[0]["source"] == "b"


def test_metadata_filtering(tmp_path):
    corpus = tmp_path / "corpus"
    _copy_fixtures(corpus)
    store = tmp_path / "store"
    rag = RAG(RAGConfig(embedder="hash", store_path=str(store), top_k=5))
    rag.ingest(str(corpus))

    # Tag only one chunk with a metadata key.
    target = rag.store.chunks[0]
    target["metadata"] = {"tag": "policy"}
    for chunk in rag.store.chunks[1:]:
        chunk["metadata"] = {"tag": "other"}

    hits = rag.retrieve("RAG", filters={"tag": "policy"})
    assert hits
    assert all(c["metadata"].get("tag") == "policy" for c in hits)


def test_ask_returns_confidence_and_citations(tmp_path):
    corpus = tmp_path / "corpus"
    _copy_fixtures(corpus)
    store = tmp_path / "store"
    rag = RAG(RAGConfig(embedder="hash", llm="extractive", store_path=str(store), top_k=3))
    rag.ingest(str(corpus))

    result = rag.ask("What does RAG stand for?")
    assert "confidence" in result
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0
    assert "citations" in result
    assert result["citations"][0]["index"] == 1
    assert "source" in result["citations"][0]


def test_ask_abstains_below_threshold(tmp_path):
    corpus = tmp_path / "corpus"
    _copy_fixtures(corpus)
    store = tmp_path / "store"
    rag = RAG(
        RAGConfig(
            embedder="hash",
            llm="extractive",
            store_path=str(store),
            top_k=3,
            answer_threshold=0.99,
        )
    )
    rag.ingest(str(corpus))
    result = rag.ask("unrelated gibberish zzqqxx")
    assert "confidently" in result["answer"]


def test_ask_stream_matches_ask(tmp_path):
    corpus = tmp_path / "corpus"
    _copy_fixtures(corpus)
    store = tmp_path / "store"
    rag = RAG(RAGConfig(embedder="hash", llm="dummy", store_path=str(store), top_k=3))
    rag.ingest(str(corpus))

    full = rag.ask("What does RAG stand for?")["answer"]
    streamed = "".join(rag.ask_stream("What does RAG stand for?"))
    assert streamed == full


def test_ask_stream_persists_memory(tmp_path):
    corpus = tmp_path / "corpus"
    _copy_fixtures(corpus)
    store = tmp_path / "store"
    rag = RAG(
        RAGConfig(embedder="hash", llm="dummy", memory_size=3, store_path=str(store), top_k=3)
    )
    rag.ingest(str(corpus))
    list(rag.ask_stream("What does RAG stand for?", session_id="s"))
    assert len(rag.memory.history("s")) == 1
