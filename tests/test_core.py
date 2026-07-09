import shutil
from pathlib import Path

from raglet import eval as eval_mod
from raglet.core import RAG, RAGConfig

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


def test_config_persistence_restores_embedder(tmp_path):
    corpus = tmp_path / "corpus"
    _copy_fixtures(corpus)
    store = tmp_path / "store"
    # Index with a non-default embedder variant (different vector dim).
    rag = RAG(
        RAGConfig(
            embedder="hash",
            embedder_kwargs={"dim": 512},
            llm="extractive",
            store_path=str(store),
            top_k=3,
        )
    )
    rag.ingest(str(corpus))
    rag.save()

    # Reopen with default config (dim 256). Without restoring the persisted
    # config the query dimension (256) would not match the stored embeddings
    # (512) and retrieval would crash; the persisted config must win.
    rag2 = RAG(RAGConfig(embedder="hash", store_path=str(store), top_k=3))
    assert rag2.load() is True
    assert rag2.config.embedder_kwargs.get("dim") == 512
    candidates = rag2.retrieve("What does RAG stand for?")
    assert candidates
    assert any(c["source"] == "rag_intro.txt" for c in candidates)


def test_remove_source(tmp_path):
    corpus = tmp_path / "corpus"
    _copy_fixtures(corpus)
    store = tmp_path / "store"
    rag = RAG(RAGConfig(embedder="hash", store_path=str(store), top_k=5))
    rag.ingest(str(corpus))
    total_before = len(rag.store.chunks)

    removed = rag.remove("rag_intro.txt")
    assert removed > 0
    assert len(rag.store.chunks) == total_before - removed
    assert all(c["source"] != "rag_intro.txt" for c in rag.store.chunks)

    # The removed source must no longer appear in retrieval results.
    result = rag.ask("What does RAG stand for?")
    assert all(s["source"] != "rag_intro.txt" for s in result["sources"])

    # Removal must survive a reload.
    rag2 = RAG(RAGConfig(embedder="hash", store_path=str(store)))
    rag2.load()
    assert all(c["source"] != "rag_intro.txt" for c in rag2.store.chunks)


def test_eval_multi_gold(tmp_path):
    import json

    corpus = tmp_path / "corpus"
    _copy_fixtures(corpus)
    store = tmp_path / "store"
    rag = RAG(RAGConfig(embedder="hash", store_path=str(store), top_k=3))
    rag.ingest(str(corpus))

    qa = [
        {"question": "What does RAG stand for?", "gold_source": ["rag_intro.txt", "embeddings.txt"]},
    ]
    qa_file = tmp_path / "qa.json"
    qa_file.write_text(json.dumps(qa), encoding="utf-8")
    multi = eval_mod.evaluate(rag, str(qa_file))
    assert multi["questions"] == 1
    assert multi["retrieval_recall@k"] >= 0.0
    assert 0.0 <= multi["precision@k"] <= 1.0
