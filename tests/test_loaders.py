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
    path = tmp_path / "note.xyz"
    path.write_text("a,b", encoding="utf-8")
    try:
        load_document(str(path))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_load_directory(tmp_path):
    (tmp_path / "a.txt").write_text("one", encoding="utf-8")
    (tmp_path / "b.md").write_text("two", encoding="utf-8")
    (tmp_path / "c.csv").write_text("x,y", encoding="utf-8")
    docs = load_documents(str(tmp_path))
    # txt, md and csv are all supported now.
    assert len(docs) == 3
    sources = {d["source"] for d in docs}
    assert sources == {"a.txt", "b.md", "c.csv"}


def test_load_html_strips_tags(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(
        "<html><body><p>Hello <b>world</b></p><script>ignore()</script></body></html>",
        encoding="utf-8",
    )
    doc = load_document(str(path))
    assert "Hello" in doc["text"] and "world" in doc["text"]
    assert "ignore" not in doc["text"]


def test_load_json_flattens_strings(tmp_path):
    path = tmp_path / "data.json"
    path.write_text('{"a": "foo", "b": ["bar", {"c": "baz"}]}', encoding="utf-8")
    doc = load_document(str(path))
    assert "foo" in doc["text"] and "bar" in doc["text"] and "baz" in doc["text"]


def test_load_csv_joins_rows(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("name,value\nrag,42\nlet,7", encoding="utf-8")
    doc = load_document(str(path))
    assert "rag" in doc["text"] and "42" in doc["text"]


def test_supported_extensions_constant():
    assert ".pdf" in SUPPORTED_EXTENSIONS
    assert ".docx" in SUPPORTED_EXTENSIONS
    assert ".html" in SUPPORTED_EXTENSIONS
    assert ".json" in SUPPORTED_EXTENSIONS
    assert ".csv" in SUPPORTED_EXTENSIONS
