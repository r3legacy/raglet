"""Command-line interface for raglet."""

import argparse
import json
import os
import sys
from typing import Optional

from . import eval as eval_mod
from .core import RAG, RAGConfig


def _build_rag(args: argparse.Namespace, load: bool = True) -> RAG:
    config = RAGConfig(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        embedder=args.embedder,
        llm=args.llm,
        store_path=args.store,
        use_sparse=not args.no_sparse,
        use_rerank=args.rerank,
        reranker=getattr(args, "reranker", "score"),
        rerank_top_n=getattr(args, "rerank_top_n", 5),
        top_k=args.top_k,
        query_expansion=getattr(args, "query_expansion", "none"),
        memory_size=getattr(args, "memory_size", 0),
    )
    rag = RAG(config)
    if load:
        rag.load()
    return rag


def cmd_ingest(args: argparse.Namespace) -> None:
    rag = _build_rag(args, load=not args.reset)
    count = rag.ingest(args.path, reset=args.reset)
    print(f"[raglet] indexed {count} chunks -> {rag.config.store_path}")


def cmd_ask(args: argparse.Namespace) -> None:
    rag = _build_rag(args)
    result = rag.ask(
        args.question,
        k=args.top_k,
        source=getattr(args, "source", None),
        session_id=getattr(args, "session", None),
    )
    if getattr(args, "json", False):
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
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
    if getattr(args, "answers", False):
        gen = eval_mod.evaluate_answers(
            rag, args.data, judge=getattr(args, "judge", "offline")
        )
        print(json.dumps(gen, indent=2))


def cmd_remove(args: argparse.Namespace) -> None:
    rag = _build_rag(args)
    removed = rag.remove(args.source)
    print(f"[raglet] removed {removed} chunks for source '{args.source}'")


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
    parser.add_argument(
        "--reranker",
        default="score",
        choices=["score", "cross-encoder", "llm"],
    )
    parser.add_argument("--rerank-top-n", type=int, default=5)
    parser.add_argument("--source", default=None, help="Limit retrieval to this source.")
    parser.add_argument(
        "--query-expansion",
        default="none",
        choices=["none", "multi", "hyde"],
        help="Expand the query into variants to improve recall (hyde needs an LLM).",
    )
    parser.add_argument(
        "--memory-size",
        type=int,
        default=0,
        help="Number of past Q/A turns to remember per session (0 disables).",
    )
    parser.add_argument(
        "--chunking-strategy",
        default="flat",
        choices=["flat", "parent_child"],
        help="parent_child retrieves on small chunks but answers from parent context.",
    )


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="raglet", description="A tiny, local-first RAG toolkit."
    )
    parser.add_argument("-v", "--version", action="store_true", help="Show version and exit.")
    subparsers = parser.add_subparsers(dest="command")

    ingest = subparsers.add_parser("ingest", parents=[_common()], help="Index files or a folder.")
    ingest.add_argument("path")
    ingest.add_argument(
        "--reset", action="store_true", help="Wipe the existing index before ingesting."
    )
    ingest.set_defaults(func=cmd_ingest)

    ask = subparsers.add_parser("ask", parents=[_common()], help="Ask a question.")
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true", help="Emit the result as JSON.")
    ask.add_argument("--session", default=None, help="Conversation session id (needs --memory-size).")
    ask.set_defaults(func=cmd_ask)

    serve = subparsers.add_parser("serve", parents=[_common()], help="Launch the Gradio web UI.")
    serve.add_argument("--port", type=int, default=7860)
    serve.set_defaults(func=cmd_serve)

    evaluate = subparsers.add_parser("eval", parents=[_common()], help="Evaluate retrieval.")
    evaluate.add_argument("--data", default=None)
    evaluate.add_argument(
        "--answers", action="store_true", help="Also evaluate generated answers (faithfulness)."
    )
    evaluate.add_argument(
        "--judge",
        default="offline",
        choices=["offline", "llm"],
        help="Groundedness judge for answer evaluation.",
    )
    evaluate.set_defaults(func=cmd_eval)

    remove = subparsers.add_parser("rm", parents=[_common()], help="Remove a source from the index.")
    remove.add_argument("source", help="Source basename to remove (e.g. doc.txt).")
    remove.set_defaults(func=cmd_remove)

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
