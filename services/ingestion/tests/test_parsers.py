import pytest

from app.parsers import ParseError, parse_document, parse_markdown, parse_pdf, parse_text
from tests.conftest import make_pdf


def test_markdown_sections_carry_heading_path():
    md = "# Guía\nintro\n## Instalación\npip install x\n### Windows\nusa py\n## Uso\nllama a run"
    sections = parse_markdown(md)
    assert [s.section for s in sections] == [
        "Guía",
        "Guía > Instalación",
        "Guía > Instalación > Windows",
        "Guía > Uso",
    ]
    assert sections[2].text == "usa py"


def test_markdown_ignores_headings_inside_code_fences():
    md = "# Título\n```bash\n# esto es un comentario\nls\n```\ntexto final"
    sections = parse_markdown(md)
    assert len(sections) == 1
    assert "# esto es un comentario" in sections[0].text


def test_markdown_strips_front_matter():
    md = "---\ntitle: algo\n---\n# Hola\ncontenido"
    sections = parse_markdown(md)
    assert len(sections) == 1
    assert "title" not in sections[0].text


def test_markdown_without_headings_has_no_section():
    sections = parse_markdown("solo texto plano")
    assert sections[0].section is None


def test_text_single_section_and_empty():
    assert len(parse_text("  hola  ")) == 1
    assert parse_text("   \n ") == []


def test_pdf_pages_become_sections_with_page_number():
    sections = parse_pdf(make_pdf(["Primera pagina", "Segunda pagina"]))
    assert [s.page for s in sections] == [1, 2]
    assert sections[1].section == "Página 2"
    assert "Segunda" in sections[1].text


def test_invalid_pdf_raises_parse_error():
    with pytest.raises(ParseError):
        parse_pdf(b"esto no es un pdf")


def test_unsupported_extension_raises():
    with pytest.raises(ParseError):
        parse_document("datos.csv", b"a,b")


def test_latin1_text_is_decoded():
    sections = parse_document("nota.txt", "canción".encode("latin-1"))
    assert sections[0].text == "canción"
