# raglet

<p align="center">
  <em>A tiny, local-first RAG toolkit with hybrid retrieval and reranking.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/dependencies-numpy%20only%20(core)-orange.svg" alt="Core deps">
</p>

**raglet** is a small, readable Retrieval-Augmented Generation (RAG) library
that runs **entirely on your machine**. It combines dense (semantic) and
sparse (BM25) retrieval with Reciprocal Rank Fusion, optional reranking, and a
pluggable choice of embedders and LLMs — from a zero-download baseline to
local models and cloud APIs.

> No API key? No problem. The default embedder (`hash`) and LLM (`extractive`)
> need **no model downloads**, so `raglet` works the moment you install it.

---

## Features

- **Local-first & private** — runs offline; your documents never leave the machine.
- **Hybrid retrieval** — dense + BM25 fused with Reciprocal Rank Fusion (RRF).
- **Pluggable embedders** — `hash` (zero-dep), `local` (sentence-transformers), `openai`, `ollama`.
- **Pluggable LLMs** — `extractive` (offline baseline), `ollama`, `openai`, `anthropic`.
- **Optional reranking** — score-based or cross-encoder rerankers.
- **Tiny & readable** — every module is a few dozen lines; great for learning RAG.
- **CLI + Python API + Gradio UI** — use it however you like.
- **Built-in evaluation** — measure retrieval recall on a labeled dataset.

## Installation

```bash
pip install raglet
```

Install extras as needed:

```bash
pip install "raglet[local]"   # semantic embeddings via sentence-transformers + faiss
pip install "raglet[ui]"      # Gradio web UI
pip install "raglet[cloud]"   # OpenAI / Anthropic providers
pip install "raglet[all]"     # everything
```

## Quickstart (Python)

```python
from raglet import RAG, RAGConfig

rag = RAG(RAGConfig(embedder="hash", llm="extractive", store_path="./.store"))
rag.ingest("./docs")                 # .txt / .md / .pdf / .docx

result = rag.ask("What is our refund policy?")
print(result["answer"])
for src in result["sources"]:
    print("-", src["source"], f'({src["score"]:.3f})')
```

## Quickstart (CLI)

```bash
# 1. Index a folder of documents
raglet ingest ./docs --store ./.store

# 2. Ask a question
raglet ask "What is our refund policy?" --store ./.store

# 3. Launch a web UI
raglet serve --store ./.store

# 4. Evaluate retrieval quality
raglet eval --store ./.store --data tests/sample_qa.json
```

## Going semantic & local

Swap the zero-dep defaults for real local models:

```python
from raglet import RAG, RAGConfig

rag = RAG(RAGConfig(
    embedder="local",     # sentence-transformers (downloads a small model once)
    llm="ollama",         # needs `ollama run llama3.2` running locally
    use_rerank=True,      # enables cross-encoder reranking when available
))
rag.ingest("./docs")
print(rag.ask("Summarize the onboarding guide.")["answer"])
```

## Architecture

```
        ┌─────────────┐
docs ──▶ │  loaders   │  .txt / .md / .pdf / .docx
        └──────┬──────┘
               ▼
        ┌─────────────┐   ┌────────────────┐
        │  chunking   │──▶│  embeddings    │  hash | local | openai | ollama
        └─────────────┘   └───────┬────────┘
                                   ▼
        ┌─────────────┐    ┌──────────────┐   ┌────────────┐
        │   BM25      │    │  VectorStore │   │  fusion    │  Reciprocal Rank Fusion
        │  (sparse)   │───▶│   (dense)    │──▶│  (RRF)     │──▶ rerank ──▶ answer
        └─────────────┘    └──────────────┘   └────────────┘       (optional)
                                               │
                                               ▼
                                          ┌─────────┐
                                          │   LLM   │  extractive | ollama | openai | anthropic
                                          └─────────┘
```

## How hybrid retrieval works

1. **Dense**: embed the query and rank chunks by cosine similarity.
2. **Sparse**: rank the same chunks with BM25 over token matches.
3. **Fusion**: blend both rankings with Reciprocal Rank Fusion so a chunk
   that scores well in *either* signal rises to the top.
4. **Rerank** (optional): reorder the fused candidates with a cross-encoder
   for higher precision before generation.

## Comparison

| Feature            | **raglet** | LangChain | RAGFlow | LightRAG |
| ------------------ | ---------- | --------- | ------- | -------- |
| Local-first        | ✅         | ⚠️        | ✅      | ✅       |
| Zero-dependency run| ✅         | ❌        | ❌      | ❌       |
| Hybrid (BM25+dense)| ✅        | ⚠️        | ✅      | ⚠️       |
| Lines of code      | ~1k        | huge      | huge    | large    |
| Learning curve     | flat       | steep    | medium  | medium   |
| Gradio UI included | ✅         | ❌        | ✅      | ❌       |

*raglet trades breadth for simplicity: it is meant to be read, forked, and
extended — not to be an all-in-one platform.*

## Examples

See [`examples/`](examples/) for runnable notebooks:

- `quickstart.ipynb` — ingest + ask with zero model downloads.
- `local_only.ipynb` — fully offline pipeline (local embeddings + Ollama).
- `hybrid_vs_dense.ipynb` — hybrid retrieval vs. dense-only.

## Evaluation

raglet ships with a retrieval evaluation harness. Provide a JSON list of
`{"question": ..., "gold_source": ...}` objects:

```bash
raglet eval --store ./.store --data tests/sample_qa.json
# {"questions": 3, "answered": 3, "retrieval_recall@k": 1.0}
```

## Roadmap

- [ ] Metadata / filtering at retrieval time
- [ ] Streaming answers from the LLM
- [ ] Async ingest for large corpora
- [ ] More document formats (HTML, JSON, Notion)
- [ ] Pre-built Docker image for `raglet serve`

## Contributing

Pull requests are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md). Keep it
tiny, keep it readable, keep dependencies optional.

## License

[MIT](LICENSE)
