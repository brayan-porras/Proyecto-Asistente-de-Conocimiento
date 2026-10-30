from app.chunking import chunk_sections
from app.models import Section


def _words(n: int) -> str:
    return " ".join(f"palabra{i}" for i in range(n))


def test_long_section_respects_size_and_overlaps():
    chunks = chunk_sections([Section(_words(400), "Guía")], 300, 60, 20)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)
    for previous, current in zip(chunks, chunks[1:]):
        tail = previous.text.split()[-3:]
        assert all(word in current.text for word in tail)


def test_chunks_never_cross_sections_and_keep_metadata():
    sections = [
        Section("a" * 50, "Guía > A", None),
        Section("b" * 50, "Guía > B", None),
    ]
    chunks = chunk_sections(sections, 1000, 100, 20)
    assert [c.section for c in chunks] == ["Guía > A", "Guía > B"]
    assert [c.chunk_index for c in chunks] == [0, 1]


def test_pdf_page_is_preserved():
    chunks = chunk_sections([Section("x" * 60, "Página 4", 4)], 1000, 100, 20)
    assert chunks[0].page == 4


def test_tiny_chunks_are_dropped():
    assert chunk_sections([Section("ok", "S")], 1000, 100, 20) == []


def test_embedding_text_includes_section():
    chunk = chunk_sections([Section("contenido " * 5, "Guía > A")], 1000, 100, 20)[0]
    assert chunk.embedding_text.startswith("Guía > A\n\n")
