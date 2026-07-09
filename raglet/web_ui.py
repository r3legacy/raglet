"""Optional Gradio web UI for raglet."""

from typing import Any, Tuple


def build_ui(rag: Any):
    """Build a Gradio Blocks app bound to a :class:`raglet.RAG` instance."""
    import gradio as gr

    def answer(query: str, source: str) -> Tuple[str, str]:
        source_filter = source.strip() or None
        result = rag.ask(query, source=source_filter)
        sources = "\n".join(
            f"- {src['source']} ({src.get('score', 0):.3f})" for src in result["sources"]
        )
        return result["answer"], sources or "(no sources)"

    with gr.Blocks(title="raglet") as demo:
        gr.Markdown("# raglet — local-first RAG")
        gr.Markdown("Ask questions over the documents you indexed with `raglet ingest`.")
        query = gr.Textbox(
            label="Question", placeholder="What does the documentation say about ...?"
        )
        source = gr.Textbox(
            label="Source filter (optional)",
            placeholder="Limit retrieval to a specific source filename",
        )
        session = gr.Textbox(
            label="Session id (optional)",
            placeholder="Enable multi-turn memory if --memory-size > 0",
        )
        button = gr.Button("Ask")
        out_answer = gr.Textbox(label="Answer", lines=10)
        out_sources = gr.Textbox(label="Sources", lines=4)
        out_history = gr.Textbox(label="Conversation history", lines=6)

        def handle(q, s, sess):
            source_filter = s.strip() or None
            session_id = sess.strip() or None
            # Resolve sources/history once (without persisting memory) so the
            # streamed answer can be shown progressively alongside them.
            base = rag.ask(q, source=source_filter, session_id=None)
            sources = "\n".join(
                f"- {src['source']} ({src.get('score', 0):.3f})" for src in base["sources"]
            )
            answer = ""
            for chunk in rag.ask_stream(
                q, source=source_filter, session_id=session_id
            ):
                answer += chunk
                yield answer, sources or "(no sources)", "(streaming…)"
            history = ""
            if session_id and rag.memory is not None:
                history = "\n".join(
                    f"Q: {q}\nA: {a}" for q, a in rag.memory.history(session_id)
                )
            yield answer, sources or "(no sources)", history or "(no history)"

        button.click(
            handle, inputs=[query, source, session], outputs=[out_answer, out_sources, out_history]
        )

        gr.Markdown("## Manage index")
        rm_source = gr.Textbox(
            label="Remove source",
            placeholder="Source basename to delete from the index",
        )
        rm_button = gr.Button("Remove source")
        out_remove = gr.Textbox(label="Result", lines=2)
        rm_button.click(
            lambda s: f"Removed {rag.remove(s.strip())} chunks"
            if s.strip()
            else "Enter a source name",
            inputs=[rm_source],
            outputs=[out_remove],
        )
    return demo
