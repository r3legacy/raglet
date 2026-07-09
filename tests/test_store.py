
from raglet.store import VectorStore


def test_add_and_search():
    store = VectorStore()
    chunks = [
        {"text": "alpha", "source": "a.txt"},
        {"text": "beta", "source": "b.txt"},
        {"text": "gamma", "source": "c.txt"},
    ]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    store.add(chunks, embeddings)
    assert len(store.chunks) == 3
    assert store.dim == 3

    results = store.search([1.0, 0.0, 0.0], k=2)
    assert len(results) == 2
    assert results[0]["source"] == "a.txt"
    assert results[0]["score"] > 0.99


def test_remove_preserves_embedding_alignment():
    # Removing a chunk must keep each surviving chunk's *text* aligned with its
    # *embedding row*. A previous bug used the post-renumber id instead of the
    # original embedding row, shifting every later chunk's vector.
    dim = 4
    store = VectorStore()
    chunks = [{"text": f"t{i}", "source": f"s{i}.txt"} for i in range(dim)]
    embeddings = [[float(i == j) for j in range(dim)] for i in range(dim)]
    store.add(chunks, embeddings)

    store.remove([1])  # drop t1

    # Each surviving chunk must be retrievable by its own one-hot vector.
    for i in (0, 2, 3):
        vec = [float(i == j) for j in range(dim)]
        results = store.search(vec, k=1)
        assert results, f"no result for chunk t{i}"
        assert results[0]["source"] == f"s{i}.txt", f"misalignment at chunk t{i}"


def test_save_and_load(tmp_path):
    store = VectorStore()
    chunks = [{"text": "x", "source": "x.txt"}]
    store.add(chunks, [[0.5, 0.5, 0.0]])
    path = str(tmp_path / "store")
    store.save(path)

    loaded = VectorStore.load(path)
    assert len(loaded.chunks) == 1
    assert loaded.chunks[0]["source"] == "x.txt"
    results = loaded.search([0.5, 0.5, 0.0], k=1)
    assert results[0]["source"] == "x.txt"


def test_hash_embedding_ngrams_capture_morphology():
    from raglet.embeddings import HashEmbedding

    emb = HashEmbedding(dim=256, ngrams=True)
    a = emb.embed(["embedding vector"])[0]
    b = emb.embed(["embeddings vectors"])[0]
    # Related surface forms should be closer than unrelated text.
    import math

    def cos(x, y):
        dot = sum(p * q for p, q in zip(x, y))
        nx = math.sqrt(sum(v * v for v in x))
        ny = math.sqrt(sum(v * v for v in y))
        return dot / (nx * ny) if nx and ny else 0.0

    related = cos(a, b)
    unrelated = cos(a, emb.embed(["banana airplane"])[0])
    assert related > unrelated


def test_hash_embedding_is_deterministic():
    from raglet.embeddings import HashEmbedding

    emb = HashEmbedding(dim=128, ngrams=False)
    assert emb.embed(["same text"])[0] == emb.embed(["same text"])[0]
