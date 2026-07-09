"""Unit tests for the advanced building blocks: chunking, expansion, memory."""

from raglet.chunking import chunk_parent_child, chunk_text
from raglet.expansion import QueryExpander
from raglet.memory import ConversationMemory


def test_chunk_parent_child_structure():
    text = " ".join(f"word{i}" for i in range(400))
    blocks = chunk_parent_child(text, parent_size=150, child_size=50, child_overlap=10)
    assert blocks
    for block in blocks:
        assert "parent" in block and "children" in block
        assert block["children"]
        # Children are slices of the parent text.
        for child in block["children"]:
            assert child in block["parent"]


def test_chunk_parent_child_empty():
    assert chunk_parent_child("") == []
    assert chunk_parent_child("   ") == []


def test_flat_chunk_text_unchanged():
    # The default API must keep returning a list of strings.
    chunks = chunk_text("one two three four five six seven eight nine ten", chunk_size=3, overlap=1)
    assert all(isinstance(c, str) for c in chunks)
    assert len(chunks) > 1


def test_query_expander_none_returns_original():
    exp = QueryExpander("none")
    assert exp.expand("hello world") == ["hello world"]


def test_query_expander_lexical_fallback():
    # Without a real LLM, "multi" uses a dependency-free lexical decomposition.
    exp = QueryExpander("multi")
    out = exp.expand("what is RAG and how do embeddings work")
    assert "what is RAG and how do embeddings work" in out
    assert any("embeddings" in q for q in out)
    # HyDE degrades to the original query offline.
    assert QueryExpander("hyde").expand("what is RAG") == ["what is RAG"]


def test_conversation_memory_trim_and_persist(tmp_path):
    path = str(tmp_path / "mem.json")
    mem = ConversationMemory(max_turns=2, path=path)
    for i in range(5):
        mem.add("s", f"q{i}", f"a{i}")
    # Only the last 2 turns are kept.
    assert len(mem.history("s")) == 2
    assert mem.history("s")[0][0] == "q3"

    # Reloading restores the persisted sessions.
    mem2 = ConversationMemory(max_turns=2, path=path)
    assert len(mem2.history("s")) == 2

    # Disabled memory stores nothing.
    disabled = ConversationMemory(max_turns=0)
    disabled.add("x", "q", "a")
    assert disabled.history("x") == []
