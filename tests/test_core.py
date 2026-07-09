import shutil
from pathlib import Path

from raglet import eval as eval_mod
from raglet.core import RAG, RAGConfig
from raglet.llm import build_prompt

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


def test_parent_child_returns_parent_context(tmp_path):
    corpus = tmp_path / "corpus"
    _copy_fixtures(corpus)
    store = tmp_path / "store"
    rag = RAG(
        RAGConfig(
            embedder="hash",
            chunking_strategy="parent_child",
            parent_size=2000,
            child_size=200,
            store_path=str(store),
            top_k=3,
        )
    )
    count = rag.ingest(str(corpus))
    assert count > 0
    # Children are indexed, parents stored separately.
    assert rag._parents
    assert all(c["chunk_type"] == "child" for c in rag.store.chunks)

    result = rag.ask("What does RAG stand for?")
    assert result["answer"]
    # Generation context must be the (larger) parent text, not a bare child.
    assert any(len(c["text"]) >= 200 for c in result["context"])
    assert any(c.get("source") == "rag_intro.txt" for c in result["sources"])


def test_parent_child_persists_and_reloads(tmp_path):
    corpus = tmp_path / "corpus"
    _copy_fixtures(corpus)
    store = tmp_path / "store"
    rag = RAG(
        RAGConfig(
            embedder="hash",
            chunking_strategy="parent_child",
            child_size=200,
            store_path=str(store),
            top_k=3,
        )
    )
    rag.ingest(str(corpus))
    rag.save()

    rag2 = RAG(RAGConfig(embedder="hash", store_path=str(store), top_k=3))
    assert rag2.load() is True
    assert rag2.config.chunking_strategy == "parent_child"
    assert rag2._parents
    result = rag2.ask("What does RAG stand for?")
    assert result["answer"]


def test_query_expansion_lexical(tmp_path):
    corpus = tmp_path / "corpus"
    _copy_fixtures(corpus)
    store = tmp_path / "store"
    # "multi" with the offline extractive LLM falls back to lexical variants.
    rag = RAG(RAGConfig(embedder="hash", llm="extractive", query_expansion="multi", store_path=str(store), top_k=3))
    rag.ingest(str(corpus))
    candidates = rag.retrieve("What is RAG and how does it work?")
    assert candidates
    # Expansion must not break retrieval; original query is always included.
    assert any(c["source"] == "rag_intro.txt" for c in candidates)


def test_faithfulness_eval_offline(tmp_path):
    import json

    corpus = tmp_path / "corpus"
    _copy_fixtures(corpus)
    store = tmp_path / "store"
    rag = RAG(RAGConfig(embedder="hash", store_path=str(store), top_k=3))
    rag.ingest(str(corpus))

    qa = [
        {
            "question": "What does RAG stand for?",
            "gold_answer": "Retrieval-Augmented Generation",
            "gold_source": "rag_intro.txt",
        }
    ]
    qa_file = tmp_path / "qa.json"
    qa_file.write_text(json.dumps(qa), encoding="utf-8")
    report = eval_mod.evaluate_answers(rag, str(qa_file))
    assert report["questions"] == 1
    assert 0.0 <= report["faithfulness"] <= 1.0
    assert 0.0 <= report["avg_groundedness"] <= 1.0
    assert report["answer_recall"] is not None


def test_conversation_memory_carries_context(tmp_path):
    corpus = tmp_path / "corpus"
    _copy_fixtures(corpus)
    store = tmp_path / "store"
    rag = RAG(RAGConfig(embedder="hash", llm="extractive", memory_size=3, store_path=str(store), top_k=3))
    rag.ingest(str(corpus))

    first = rag.ask("What does RAG stand for?", session_id="s1")
    assert first["answer"]
    # The session should now contain one Q/A pair.
    assert len(rag.memory.history("s1")) == 1

    # Re-asking within the same session keeps the prior turn in memory.
    rag.ask("Can you elaborate on that?", session_id="s1")
    assert len(rag.memory.history("s1")) == 2

    # Memory survives a reload.
    rag2 = RAG(RAGConfig(embedder="hash", memory_size=3, store_path=str(store)))
    rag2.load()
    assert len(rag2.memory.history("s1")) == 2


def test_memory_is_runtime_switch(tmp_path):
    # Build the index WITHOUT memory, then reopen WITH memory enabled at
    # runtime. Memory must not be forced off by the persisted index config.
    corpus = tmp_path / "corpus"
    _copy_fixtures(corpus)
    store = tmp_path / "store"
    rag = RAG(RAGConfig(embedder="hash", store_path=str(store)))
    rag.ingest(str(corpus))
    rag.save()
    assert rag.memory is None

    rag2 = RAG(RAGConfig(embedder="hash", store_path=str(store), memory_size=3))
    assert rag2.load() is True
    assert rag2.memory is not None
    rag2.ask("What does RAG stand for?", session_id="rt")
    assert len(rag2.memory.history("rt")) == 1


def test_summarize_creates_summary_nodes(tmp_path):
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "a.txt").write_text(
        "RAG stands for Retrieval Augmented Generation. "
        "It combines a retriever with a generator. "
        "Dense and sparse signals are fused with RRF. "
        "HyDE drafts a hypothetical document for retrieval. "
        "Parent-child chunking keeps larger context. "
        "Confidence is derived from the retrieval scores. "
        "Citations map each claim back to a source."
    )
    rag = RAG(
        RAGConfig(
            embedder="hash",
            chunking_strategy="parent_child",
            summarize=True,
            store_path=str(tmp_path / "s"),
            top_k=3,
        )
    )
    assert rag.ingest(str(corpus)) > 0
    types = {c["chunk_type"] for c in rag.store.chunks}
    assert "summary" in types
    res = rag.ask("What does RAG stand for?")
    assert res["answer"]


def test_retry_on_weak_does_not_crash(tmp_path):
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "a.txt").write_text("RAG stands for Retrieval Augmented Generation.")
    rag = RAG(
        RAGConfig(
            embedder="hash",
            retry_on_weak=True,
            query_expansion="multi",
            store_path=str(tmp_path / "s"),
            top_k=3,
        )
    )
    rag.ingest(str(corpus))
    res = rag.ask("unrelated gibberish zzqqxx")
    assert "confidence" in res


def test_robust_queries_broadens():
    rag = RAG(RAGConfig(embedder="hash"))
    variants = rag._robust_queries("what is RAG and how does it work")
    assert len(variants) > 1


def test_rewrite_query_offline_concats_history():
    rag = RAG(RAGConfig(embedder="hash"))
    out = rag._rewrite_query("why?", [("What is RAG?", "x")])
    assert "What is RAG?" in out and "why?" in out


def test_build_prompt_token_budget():
    chunks = [
        {"text": "a" * 1000, "source": "s.txt"},
        {"text": "b" * 1000, "source": "t.txt"},
    ]
    prompt = build_prompt("q", chunks, max_tokens=50)
    # The second (large) chunk must be dropped/truncated under the budget.
    assert "b" * 1000 not in prompt


def test_eval_reports_ndcg(tmp_path):
    corpus = tmp_path / "c"
    corpus.mkdir()
    shutil.copy(FIXTURES / "rag_intro.txt", corpus / "rag_intro.txt")
    rag = RAG(RAGConfig(embedder="hash", store_path=str(tmp_path / "s"), top_k=3))
    rag.ingest(str(corpus))
    rep = eval_mod.evaluate(rag)
    assert "ndcg@k" in rep


def test_eval_directory_of_qa(tmp_path):
    corpus = tmp_path / "c"
    corpus.mkdir()
    shutil.copy(FIXTURES / "rag_intro.txt", corpus / "rag_intro.txt")
    rag = RAG(RAGConfig(embedder="hash", store_path=str(tmp_path / "s"), top_k=3))
    rag.ingest(str(corpus))
    qa_dir = tmp_path / "qa"
    qa_dir.mkdir()
    (qa_dir / "a.json").write_text(
        '[{"question": "What does RAG stand for?", "gold_source": "rag_intro.txt"}]',
        encoding="utf-8",
    )
    (qa_dir / "b.json").write_text(
        '[{"question": "What is raglet?", "gold_source": "rag_intro.txt"}]',
        encoding="utf-8",
    )
    rep = eval_mod.evaluate(rag, str(qa_dir))
    assert rep["questions"] == 2
