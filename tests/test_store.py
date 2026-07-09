
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
