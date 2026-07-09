from raglet.sparse import BM25


def test_bm25_ranks_relevant_first():
    corpus = [
        "the cat sat on the mat",
        "dogs are great pets",
        "the dog and the cat played outside",
    ]
    bm25 = BM25()
    bm25.fit(corpus)
    results = bm25.search("cat", k=1)
    assert results[0][0] == 0

    results = bm25.search("dog", k=1)
    assert results[0][0] == 2

    results = bm25.search("dogs", k=1)
    assert results[0][0] == 1


def test_bm25_returns_requested_count():
    corpus = ["alpha beta", "beta gamma", "gamma delta", "delta epsilon"]
    bm25 = BM25()
    bm25.fit(corpus)
    results = bm25.search("beta", k=2)
    assert len(results) == 2
