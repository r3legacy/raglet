# Contributing to raglet

Thanks for your interest in improving raglet! This project aims to stay a
**tiny, local-first RAG toolkit** — small, readable, and dependency-light.

## Getting started

```bash
git clone https://github.com/r3legacy/raglet
cd raglet
pip install -e ".[dev]"
```

## Development workflow

1. Create a feature branch: `git checkout -b my-feature`.
2. Add or update tests under `tests/`. New behavior should be covered.
3. Run the suite: `pytest -q`.
4. Keep the public API small and the code readable.
5. Open a pull request describing the change and the motivation.

## Guidelines

- Prefer optional, lazily-imported dependencies over hard requirements.
- Add a short docstring and a type hint to every public function.
- Keep the README's feature list and comparison table up to date.

## Reporting issues

Open an issue with a minimal reproduction (input, command, expected vs actual
output). For bugs, include your Python version and which extras you installed.
