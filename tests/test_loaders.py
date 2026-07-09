from raglet.loaders import SUPPORTED_EXTENSIONS, load_document, load_documents


def test_load_text(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("hello world", encoding="utf-8")
    doc = load_document(str(path))
    assert doc["text"] == "hello world"
    assert doc["source"] == "note.txt"


def test_load_markdown(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Title\nbody", encoding="utf-8")
    doc = load_document(str(path))
    assert "Title" in doc["text"]


def test_unsupported_extension(tmp_path):
    path = tmp_path / "note.csv"
    path.write_text("a,b", encoding="utf-8")
    try:
        load_document(str(path))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_load_directory(tmp_path):
    (tmp_path / "a.txt").write_text("one", encoding="utf-8")
    (tmp_path / "b.md").write_text("two", encoding="utf-8")
    (tmp_path / "ignore.csv").write_text("x,y", encoding="utf-8")
    docs = load_documents(str(tmp_path))
    assert len(docs) == 2
    sources = {d["source"] for d in docs}
    assert sources == {"a.txt", "b.md"}


def test_supported_extensions_constant():
    assert ".pdf" in SUPPORTED_EXTENSIONS
    assert ".docx" in SUPPORTED_EXTENSIONS
