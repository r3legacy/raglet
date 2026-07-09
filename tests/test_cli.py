"""Tests for CLI argument wiring and runtime-config override behaviour."""

import argparse
import os

from raglet import RAG, RAGConfig
from raglet.cli import _build_rag


def _ns(**kw):
    base = dict(
        chunk_size=500,
        overlap=50,
        embedder="hash",
        llm="extractive",
        store=str(os.path.join(os.path.dirname(__file__), "fixtures")),
        no_sparse=False,
        rerank=False,
        reranker="score",
        rerank_top_n=5,
        top_k=5,
        query_expansion="none",
        memory_size=0,
        rrf_k=60,
        dense_weight=1.0,
        sparse_weight=1.0,
        answer_threshold=0.15,
        chunking_strategy="flat",
        parent_size=1500,
        child_size=500,
        child_overlap=50,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_build_rag_wires_chunking_strategy():
    # Without loading an existing index, CLI chunking flags must be honoured.
    args = _ns(store="/tmp/raglet-nonexistent-store", chunking_strategy="parent_child")
    rag = _build_rag(args, load=False)
    assert rag.config.chunking_strategy == "parent_child"
    assert rag.config.child_size == 500


def test_build_rag_runtime_flags_override_persisted(tmp_path):
    store = tmp_path / "store"
    # Build and persist an index with the default runtime knobs.
    rag = RAG(RAGConfig(embedder="hash", store_path=str(store)))
    rag.save()

    # CLI should be able to override retrieval-time knobs on a subsequent
    # `ask`/`serve`/`eval`, even though the persisted config differs.
    args = _ns(
        store=str(store),
        query_expansion="multi",
        no_sparse=True,
        rerank=True,
        reranker="llm",
    )
    rag2 = _build_rag(args, load=True)
    assert rag2.config.query_expansion == "multi"
    assert rag2.config.use_sparse is False
    assert rag2.config.use_rerank is True
    assert rag2.config.reranker == "llm"
    # The dependent helpers must reflect the overridden values.
    assert rag2._expander is not None
    assert rag2._reranker is not None


def test_eval_default_data_is_bundled():
    from raglet import eval as eval_mod

    # The default QA file must ship with the package so `raglet eval` works
    # without an explicit --data, even when installed from a wheel.
    assert os.path.exists(eval_mod._DEFAULT_QA)


def test_cli_ls_reports_index(tmp_path, capsys):
    import shutil

    from raglet.cli import cmd_ls

    corpus = tmp_path / "c"
    corpus.mkdir()
    shutil.copy(
        os.path.join(os.path.dirname(__file__), "fixtures", "rag_intro.txt"),
        corpus / "rag_intro.txt",
    )
    args = _ns(store=str(tmp_path / "s"))
    rag = _build_rag(args, load=False)
    rag.ingest(str(corpus))

    cmd_ls(args)
    out = capsys.readouterr().out
    assert "chunks" in out
    assert "rag_intro.txt" in out


def test_cli_ls_empty_index(tmp_path, capsys):
    from raglet.cli import cmd_ls

    args = _ns(store=str(tmp_path / "empty"))
    cmd_ls(args)
    out = capsys.readouterr().out
    assert "no index" in out
