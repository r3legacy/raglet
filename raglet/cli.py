"""Command-line interface for raglet."""

import argparse
import json
import os
import sys
from typing import Any, Optional

from .core import RAG, RAGConfig
from . import eval as eval_mod


def _build_rag(args: argparse.Namespace) -> RAG:
    config = RAGConfig(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        embedder=args.embedder,
        llm=args.llm,
        store_path=args.store,
        use_sparse=not args.no_sparse,
        use_rerank=args.rerank,
        top_k=args.top_k,
    )
    rag = RAG(config)
    rag.load()
    return rag


def cmd_ingest(args: argparse.Namespace) -> None:
    rag = _build_rag(args)
    count = rag.ingest(args.path)
    print(f"[raglet] indexed {count} chunks -> {rag.config.store_path}")


def cmd_ask(args: argparse.Namespace) -> None:
    rag = _build_rag(args)
    result = rag.ask(args.question, k=args.top_k)
    print("\nANSWER:\n" + result["answer"])
    print("\nSOURCES:")
    for source in result["sources"]:
        score = source.get("score", 0.0)
        print(f"  - {source['source']}  (score {score:.3f})")


def cmd_serve(args: argparse.Namespace) -> None:
    try:
        import gradio as gr  # noqa: F401
    except ImportError:
        print("gradio is not installed. Run: pip install gradio")
        sys.exit(1)
    from .web_ui import build_ui

    rag = _build_rag(args)
    demo = build_ui(rag)
    demo.launch(server_port=args.port)


def cmd_eval(args: argparse.Namespace) -> None:
    rag = _build_rag(args)
    report = eval_mod.evaluate(rag, args.data)
    print(json.dumps(report, indent=2))


def cmd_version(args: argparse.Namespace) -> None:
    from . import __version__

    print("raglet", __version__)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--store",
        default=os.path.join(os.path.expanduser("~"), ".raglet", "store"),
    )
    parser.add_argument("--embedder", default="hash", choices=["hash", "local", "openai", "ollama"])
    parser.add_argument("--llm", default="extractive", choices=["extractive", "dummy", "ollama", "openai", "anthropic"])
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-sparse", action="store_true")
    parser.add_argument("--rerank", action="store_true")


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="raglet", description="A tiny, local-first RAG toolkit."
    )
    parser.add_argument("-v", "--version", action="store_true", help="Show version and exit.")
    subparsers = parser.add_subparsers(dest="command")

    ingest = subparsers.add_parser("ingest", parents=[_common()], help="Index files or a folder.")
    ingest.add_argument("path")
    ingest.set_defaults(func=cmd_ingest)

    ask = subparsers.add_parser("ask", parents=[_common()], help="Ask a question.")
    ask.add_argument("question")
    ask.set_defaults(func=cmd_ask)

    serve = subparsers.add_parser("serve", parents=[_common()], help="Launch the Gradio web UI.")
    serve.add_argument("--port", type=int, default=7860)
    serve.set_defaults(func=cmd_serve)

    evaluate = subparsers.add_parser("eval", parents=[_common()], help="Evaluate retrieval.")
    evaluate.add_argument("--data", default=None)
    evaluate.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)

    if args.version:
        cmd_version(args)
        return
    if not getattr(args, "command", None):
        parser.print_help()
        return
    args.func(args)


def _common() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    _add_common(common)
    return common


if __name__ == "__main__":
    main()
