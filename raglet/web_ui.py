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
        button = gr.Button("Ask")
        out_answer = gr.Textbox(label="Answer", lines=10)
        out_sources = gr.Textbox(label="Sources", lines=4)
        button.click(answer, inputs=[query, source], outputs=[out_answer, out_sources])

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
