"""Optional Gradio web UI for raglet."""

from typing import Any, List, Tuple


def build_ui(rag: Any):
    """Build a Gradio Blocks app bound to a :class:`raglet.RAG` instance."""
    import gradio as gr

    def answer(query: str) -> Tuple[str, str]:
        result = rag.ask(query)
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
        button = gr.Button("Ask")
        out_answer = gr.Textbox(label="Answer", lines=10)
        out_sources = gr.Textbox(label="Sources", lines=4)
        button.click(answer, inputs=query, outputs=[out_answer, out_sources])
    return demo
