"""Document loaders for common text formats.

Heavy parsers (PDF, DOCX) are imported lazily so the core package has no
mandatory third-party dependencies. HTML/JSON/CSV are handled with the
standard library so they work out of the box.
"""

import csv
import json
import os
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".pdf",
    ".docx",
    ".html",
    ".htm",
    ".json",
    ".csv",
}


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="ignore") as handle:
        return handle.read()


def load_text(path: str) -> str:
    return _read_text(path)


def load_markdown(path: str) -> str:
    return _read_text(path)


def load_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("PDF support requires pypdf: pip install pypdf") from exc
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def load_docx(path: str) -> str:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "DOCX support requires python-docx: pip install python-docx"
        ) from exc
    document = docx.Document(path)
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


class _HTMLTextExtractor(HTMLParser):
    """Minimal tag-stripping parser that keeps textual content."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: List[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
        if tag in ("p", "div", "li", "br", "h1", "h2", "h3", "h4", "tr"):
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    def text(self) -> str:
        return "\n".join(p.strip() for p in self._parts if p.strip())


def load_html(path: str) -> str:
    with open(path, encoding="utf-8", errors="ignore") as handle:
        raw = handle.read()
    parser = _HTMLTextExtractor()
    parser.feed(raw)
    return parser.text()


def _json_strings(node: Any) -> List[str]:
    """Recursively collect string values from a parsed JSON structure."""
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        return [v for val in node.values() for v in _json_strings(val)]
    if isinstance(node, list):
        return [v for val in node for v in _json_strings(val)]
    return []


def load_json(path: str) -> str:
    with open(path, encoding="utf-8", errors="ignore") as handle:
        data = json.load(handle)
    return "\n".join(_json_strings(data))


def load_csv(path: str) -> str:
    with open(path, encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        rows = ["\t".join(cell for cell in row if cell.strip()) for row in reader]
    return "\n".join(row for row in rows if row)


_LOADERS = {
    ".txt": load_text,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".html": load_html,
    ".htm": load_html,
    ".json": load_json,
    ".csv": load_csv,
}


def load_document(path: str) -> Dict[str, Any]:
    """Load a single file into a document dict."""
    ext = Path(path).suffix.lower()
    if ext not in _LOADERS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    text = _LOADERS[ext](path)
    stat = os.stat(path)
    return {
        "text": text,
        "source": os.path.basename(path),
        "path": path,
        "metadata": {"bytes": stat.st_size, "ext": ext},
    }


def load_documents(path: str, glob: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load a single file or every supported file under a directory."""
    location = Path(path)
    if location.is_file():
        return [load_document(str(location))]
    if location.is_dir():
        # Use a catch-all pattern so extensionless files (e.g. a dotfile-free
        # README) are also discovered; the extension filter below still drops
        # anything unsupported.
        patterns = [glob] if glob else ["*"]
        found = []
        for pattern in patterns:
            found.extend(location.rglob(pattern))
        docs: List[Dict[str, Any]] = []
        for file_path in found:
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    docs.append(load_document(str(file_path)))
                except Exception as exc:  # pragma: no cover - corrupt input
                    print(f"[raglet] skipping {file_path}: {exc}")
        return docs
    raise FileNotFoundError(f"No such file or directory: {path}")
