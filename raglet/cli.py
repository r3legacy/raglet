"""Command-line interface for raglet."""

import argparse
import json
import os
import sys
from typing import Optional

from . import eval as eval_mod
from .core import RAG, RAGConfig

# (CLI arg attribute, RAGConfig attribute, transform) for knobs that may be
# overridden at runtime. A flag is only honored when it was *explicitly*
# given on the command line (achieved via ``argparse.SUPPRESS`` defaults),
# so ``ls``/``eval``/``rm`` never clobber a persisted index config.
_RUNTIME_OVERRIDES = [
    ("query_expansion", "query_expansion", None),
    ("rerank", "use_rerank", None),
    ("reranker", "reranker", None),
    ("rerank_top_n", "rerank_top_n", None),
    ("no_sparse", "use_sparse", "negate"),
    ("rrf_k", "rrf_k", None),
    ("dense_weight", "dense_weight", None),
    ("sparse_weight", "sparse_weight", None),
    ("answer_threshold", "answer_threshold", None),
    ("memory_size", "memory_size", None),
    ("top_k", "top_k", None),
    ("summarize", "summarize", None),
    ("summarizer", "summarizer", None),
    ("retry_on_weak", "retry_on_weak", None),
    ("rewrite_queries", "rewrite_queries", None),
    ("context_budget", "context_budget", None),
]


def _build_rag(args: argparse.Namespace, load: bool = True) -> RAG:
    # Index-compatibility settings (embedder, llm, chunking, sizes) are taken
    # from the CLI at construction and then *restored* from the persisted
    # index config on load(); they are intentionally not runtime-overridable.
    config = RAGConfig(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        embedder=args.embedder,
        llm=args.llm,
        store_path=args.store,
        chunking_strategy=args.chunking_strategy,
        parent_size=args.parent_size,
        child_size=args.child_size,
        child_overlap=args.child_overlap,
        split_by=args.split_by,
    )
    # Only knobs the user explicitly passed on the CLI override the
    # persisted index config; everything else keeps its saved value.
    overrides = {}
    for arg_attr, cfg_attr, mode in _RUNTIME_OVERRIDES:
        if not hasattr(args, arg_attr):
            continue
        value = getattr(args, arg_attr)
        if mode == "negate":
            value = not value
        overrides[cfg_attr] = value
        setattr(config, cfg_attr, value)

    rag = RAG(config)
    if load:
        rag.load()
        for cfg_attr, value in overrides.items():
            setattr(rag.config, cfg_attr, value)
        rag._build_auxiliaries()
    return rag


def cmd_ingest(args: argparse.Namespace) -> None:
    rag = _build_rag(args, load=not args.reset)
    count = rag.ingest(args.path, reset=args.reset)
    print(f"[raglet] indexed {count} chunks -> {rag.config.store_path}")


def cmd_ask(args: argparse.Namespace) -> None:
    rag = _build_rag(args)
    filters = _parse_filters(getattr(args, "filter", None))

    if getattr(args, "stream", False):
        for chunk in rag.ask_stream(
            args.question,
            k=getattr(args, "top_k", None),
            source=getattr(args, "source", None),
            session_id=getattr(args, "session", None),
            filters=filters,
        ):
            print(chunk, end="", flush=True)
        print()
        return

    result = rag.ask(
        args.question,
        k=getattr(args, "top_k", None),
        source=getattr(args, "source", None),
        session_id=getattr(args, "session", None),
        filters=filters,
    )
    if getattr(args, "json", False):
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print("\nANSWER:\n" + result["answer"])
    print(f"\nCONFIDENCE: {result.get('confidence', 0.0):.3f}")
    print("\nSOURCES:")
    for source in result["sources"]:
        score = source.get("score", 0.0)
        print(f"  - {source['source']}  (score {score:.3f})")


def _parse_filters(pairs: Optional[list]) -> Optional[dict]:
    """Turn repeated ``key=value`` CLI flags into a metadata filter dict."""
    if not pairs:
        return None
    filters: dict = {}
    for pair in pairs:
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        filters[key.strip()] = value.strip()
    return filters or None


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


def cmd_ls(args: argparse.Namespace) -> None:
    """Print a summary of a built index (chunk/source counts, config)."""
    rag = _build_rag(args)
    if not rag.store.chunks:
        print(f"[raglet] no index found at {rag.config.store_path}")
        return

    sources = {}
    for chunk in rag.store.chunks:
        sources[chunk.get("source", "unknown")] = (
            sources.get(chunk.get("source", "unknown"), 0) + 1
        )
    print(f"[raglet] index: {rag.config.store_path}")
    print(f"  chunks : {len(rag.store.chunks)}")
    print(f"  sources: {len(sources)}")
    for source, count in sorted(sources.items()):
        print(f"    - {source} ({count})")
    print(f"  embedder      : {rag.config.embedder}")
    print(f"  llm           : {rag.config.llm}")
    print(f"  chunking     : {rag.config.chunking_strategy} (split_by={rag.config.split_by})")
    print(f"  sparse (BM25): {rag.config.use_sparse}")
    print(f"  summarize    : {rag.config.summarize} ({rag.config.summarizer})")
    print(f"  retry/rewrite: {rag.config.retry_on_weak}/{rag.config.rewrite_queries}")
    if rag.config.context_budget:
        print(f"  ctx budget  : {rag.config.context_budget} tokens")
    if rag.config.use_rerank:
        print(f"  reranker     : {rag.config.reranker}")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--store",
        default=os.path.join(os.path.expanduser("~"), ".raglet", "store"),
    )
    parser.add_argument("--embedder", default="hash", choices=["hash", "local", "openai", "ollama"])
    parser.add_argument("--llm", default="extractive", choices=["extractive", "dummy", "ollama", "openai", "anthropic"])
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=argparse.SUPPRESS,
                        help="Number of candidates to retrieve.")
    parser.add_argument("--no-sparse", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--rerank", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument(
        "--reranker",
        default=argparse.SUPPRESS,
        choices=["score", "cross-encoder", "llm"],
    )
    parser.add_argument("--rerank-top-n", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--rrf-k", type=int, default=argparse.SUPPRESS, help="RRF smoothing constant.")
    parser.add_argument("--dense-weight", type=float, default=argparse.SUPPRESS, help="Weight for the dense ranking in RRF.")
    parser.add_argument("--sparse-weight", type=float, default=argparse.SUPPRESS, help="Weight for the BM25 ranking in RRF.")
    parser.add_argument(
        "--answer-threshold",
        type=float,
        default=argparse.SUPPRESS,
        help="Minimum confidence; below this the pipeline abstains.",
    )
    parser.add_argument("--memory-size", type=int, default=argparse.SUPPRESS,
                        help="Number of past Q/A turns to remember per session (0 disables).")
    parser.add_argument("--parent-size", type=int, default=1500, help="Parent window size (parent_child chunking).")
    parser.add_argument("--child-size", type=int, default=500, help="Child chunk size (parent_child chunking).")
    parser.add_argument("--child-overlap", type=int, default=50, help="Child overlap (parent_child chunking).")
    parser.add_argument("--source", default=None, help="Limit retrieval to this source.")
    parser.add_argument(
        "--query-expansion",
        default=argparse.SUPPRESS,
        choices=["none", "multi", "hyde"],
        help="Expand the query into variants to improve recall (hyde needs an LLM).",
    )
    parser.add_argument(
        "--chunking-strategy",
        default="flat",
        choices=["flat", "parent_child"],
        help="parent_child retrieves on small chunks but answers from parent context.",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Index an offline summary node per parent (RAPTOR-lite) to boost "
        "high-level recall. Implies parent/child grouping.",
    )
    parser.add_argument(
        "--summarizer",
        default=argparse.SUPPRESS,
        choices=["extractive", "llm"],
        help="How to build summary nodes (llm needs a real LLM).",
    )
    parser.add_argument(
        "--retry-on-weak",
        action="store_true",
        default=argparse.SUPPRESS,
        help="If the first retrieval is weak, broaden the query once before abstaining.",
    )
    parser.add_argument(
        "--rewrite-queries",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Rewrite follow-up questions against conversation history before retrieval.",
    )
    parser.add_argument(
        "--context-budget",
        type=int,
        default=argparse.SUPPRESS,
        help="Max estimated tokens of context sent to the LLM (0 = no limit).",
    )
    parser.add_argument(
        "--split-by",
        default="token",
        choices=["token", "sentence"],
        help="Unit used to build chunks (sentence keeps sentences intact).",
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
    ask.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Metadata filter as key=value (repeatable), e.g. --filter tag=policy.",
    )
    ask.add_argument("--stream", action="store_true", help="Stream the answer token by token.")
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

    ls = subparsers.add_parser(
        "ls", parents=[_common()], help="List an index's chunks, sources and config."
    )
    ls.set_defaults(func=cmd_ls)

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
