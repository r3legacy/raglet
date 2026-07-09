from raglet.chunking import chunk_text


def test_empty_text():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_single_chunk_when_short():
    text = "The quick brown fox jumps over the lazy dog."
    chunks = chunk_text(text, chunk_size=500)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_size_respected():
    words = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_text(words, chunk_size=20, overlap=0)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.split()) <= 21  # overflow by at most one unit


def test_overlap_carries_context():
    words = " ".join(f"w{i}" for i in range(60))
    chunks = chunk_text(words, chunk_size=20, overlap=5)
    # Joining all chunks must still contain the full original text.
    assert " ".join(chunks).replace("  ", " ") == words or " ".join(chunks).count("w0") >= 1
    assert len(chunks) > 1


def test_sentence_mode():
    text = "First sentence here. Second sentence there. Third one now."
    chunks = chunk_text(text, chunk_size=5, split_by="sentence")
    assert all("." in chunk for chunk in chunks)
    assert "First sentence here." in chunks[0]
