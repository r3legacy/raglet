"""Pure-Python BM25 sparse retriever (no external dependencies)."""

import math
import re
from typing import Dict, List, Tuple

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    """A minimal BM25 implementation used for hybrid retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: List[List[str]] = []
        self.doc_lens: List[int] = []
        self.avgdl = 0.0
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_freqs: List[Dict[str, int]] = []

    def fit(self, corpus: List[str]) -> None:
        self.docs = [_tokenize(doc) for doc in corpus]
        self.doc_lens = [len(tokens) for tokens in self.docs]
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0.0
        self.df = {}
        self.doc_freqs = []
        for tokens in self.docs:
            freq: Dict[str, int] = {}
            for token in tokens:
                freq[token] = freq.get(token, 0) + 1
            self.doc_freqs.append(freq)
            for token in freq:
                self.df[token] = self.df.get(token, 0) + 1
        total = len(self.docs)
        self.idf = {
            token: math.log(1 + (total - df + 0.5) / (df + 0.5))
            for token, df in self.df.items()
        }

    def search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        query_tokens = _tokenize(query)
        scores = []
        for index, (tokens, freq) in enumerate(zip(self.docs, self.doc_freqs)):
            score = 0.0
            doc_len = self.doc_lens[index]
            for token in query_tokens:
                if token not in freq:
                    continue
                token_idf = self.idf.get(token, 0.0)
                term_freq = freq[token]
                denom = term_freq + self.k1 * (
                    1 - self.b + self.b * (doc_len / self.avgdl if self.avgdl else 1)
                )
                score += token_idf * (term_freq * (self.k1 + 1)) / denom
            scores.append(score)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(i, scores[i]) for i in order]
