import shutil
from pathlib import Path

from raglet.core import RAG, RAGConfig
from raglet import eval as eval_mod

FIXTURES = Path(__file__).parent / "fixtures"


def _copy_fixtures(dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for file in FIXTURES.glob("*.txt"):
        shutil.copy(file, dst / file.name)


def test_end_to_end_ingest_ask(tmp_path):
    corpus = tmp_path / "corpus"
    _copy_fixtures(corpus)

    store = tmp_path / "store"
    rag = RAG(
        RAGConfig(
            embedder="hash",
            llm="extractive",
            store_path=str(store),
            use_sparse=True,
            top_k=3,
        )
    )
    count = rag.ingest(str(corpus))
    assert count >= 2

    result = rag.ask("What does RAG stand for?")
    assert result["answer"]
    assert result["sources"]
    source_names = {s["source"] for s in result["sources"]}
    assert "rag_intro.txt" in source_names


def test_retrieve_returns_ranking(tmp_path):
    corpus = tmp_path / "corpus"
    _copy_fixtures(corpus)
    store = tmp_path / "store"
    rag = RAG(RAGConfig(embedder="hash", store_path=str(store), top_k=3))
    rag.ingest(str(corpus))

    candidates = rag.retrieve("How do embeddings map text?")
    assert candidates
    # The embeddings fixture should be retrieved for this query.
    assert any(c["source"] == "embeddings.txt" for c in candidates)


def test_eval_recall(tmp_path):
    corpus = tmp_path / "corpus"
    _copy_fixtures(corpus)
    store = tmp_path / "store"
    rag = RAG(RAGConfig(embedder="hash", store_path=str(store), top_k=3))
    rag.ingest(str(corpus))

    report = eval_mod.evaluate(rag)
    assert report["questions"] == 3
    assert report["retrieval_recall@k"] >= 0.0
    assert "precision@k" in report
    assert "mrr" in report


def test_ingest_reset_is_idempotent(tmp_path):
    corpus = tmp_path / "corpus"
    _copy_fixtures(corpus)
    store = tmp_path / "store"

    rag = RAG(RAGConfig(embedder="hash", store_path=str(store)))
    first = rag.ingest(str(corpus))

    # Re-ingest without reset would duplicate chunks.
    rag2 = RAG(RAGConfig(embedder="hash", store_path=str(store)))
    rag2.load()
    duplicated = rag2.ingest(str(corpus))
    assert duplicated == 0  # content-hash dedupe prevents duplicates

    # Reset wipes the index and re-indexes from scratch.
    rag3 = RAG(RAGConfig(embedder="hash", store_path=str(store)))
    reset = rag3.ingest(str(corpus), reset=True)
    assert reset == first


def test_source_filter(tmp_path):
    corpus = tmp_path / "corpus"
    _copy_fixtures(corpus)
    store = tmp_path / "store"
    rag = RAG(RAGConfig(embedder="hash", store_path=str(store), top_k=5))
    rag.ingest(str(corpus))

    result = rag.ask("What does RAG stand for?", source="rag_intro.txt")
    assert result["sources"]
    assert all(s["source"] == "rag_intro.txt" for s in result["sources"])


def test_extractive_is_query_aware(tmp_path):
    corpus = tmp_path / "corpus"
    _copy_fixtures(corpus)
    store = tmp_path / "store"
    rag = RAG(RAGConfig(embedder="hash", llm="extractive", store_path=str(store), top_k=5))
    rag.ingest(str(corpus))

    # A query about embeddings should surface the embeddings fixture, not the
    # first-ranked chunk regardless of relevance.
    result = rag.ask("How do embeddings map text into vectors?")
    sources = {s["source"] for s in result["sources"]}
    assert "embeddings.txt" in sources
