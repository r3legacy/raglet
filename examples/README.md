# raglet examples

Runnable notebooks for getting started with raglet.

| Notebook | What it shows |
| --- | --- |
| `quickstart.ipynb` | Minimal ingest + ask with zero model downloads. |
| `local_only.ipynb` | Fully offline pipeline (local embeddings + Ollama). |
| `hybrid_vs_dense.ipynb` | Comparing hybrid retrieval with dense-only. |

## Run them

```bash
pip install raglet notebook sentence-transformers gradio
jupyter notebook
```

Then open any `.ipynb` and run the cells. Replace `./docs` with a folder of
your own `.txt` / `.md` / `.pdf` / `.docx` files.
