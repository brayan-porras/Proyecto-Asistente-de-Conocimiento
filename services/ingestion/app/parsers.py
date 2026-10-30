import io
import re
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from app.models import Section

SUPPORTED_TYPES = {".pdf": "pdf", ".txt": "txt", ".md": "md"}

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FRONT_MATTER = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)


class ParseError(ValueError):
    pass


def doc_type_of(filename: str) -> str | None:
    return SUPPORTED_TYPES.get(Path(filename).suffix.lower())


def parse_document(filename: str, data: bytes) -> list[Section]:
    doc_type = doc_type_of(filename)
    if doc_type == "pdf":
        return parse_pdf(data)
    if doc_type == "md":
        return parse_markdown(_decode(data))
    if doc_type == "txt":
        return parse_text(_decode(data))
    raise ParseError(f"Tipo de archivo no soportado: {filename}")


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def parse_text(text: str) -> list[Section]:
    text = text.strip()
    return [Section(text)] if text else []


def parse_markdown(text: str) -> list[Section]:
    """Divide por encabezados; cada seccion lleva su ruta (H1 > H2 > H3)."""
    text = _FRONT_MATTER.sub("", text, count=1)
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            path = " > ".join(title for _, title in stack) or None
            sections.append(Section(body, path))
        buffer.clear()

    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        match = None if in_fence else _HEADING.match(line)
        if match:
            flush()
            level, title = len(match.group(1)), match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buffer.append(line)
    flush()
    return sections


def parse_pdf(data: bytes) -> list[Section]:
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except (PyPdfError, ValueError, OSError) as exc:
        raise ParseError(f"No se pudo leer el PDF: {exc}") from exc
    return [
        Section(text, f"Página {number}", number)
        for number, text in enumerate(pages, start=1)
        if text
    ]
