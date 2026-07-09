"""Document loaders for common text formats.

Heavy parsers (PDF, DOCX) are imported lazily so the core package has no
mandatory third-party dependencies.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf", ".docx"}


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


_LOADERS = {
    ".txt": load_text,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".pdf": load_pdf,
    ".docx": load_docx,
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
        patterns = [glob] if glob else ["*.*"]
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
