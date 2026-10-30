from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models import Chunk, Section


def chunk_sections(
    sections: list[Section],
    chunk_size: int,
    chunk_overlap: int,
    min_chars: int,
) -> list[Chunk]:
    """Cada seccion se parte por separado, asi un chunk nunca cruza encabezados ni paginas."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks: list[Chunk] = []
    for section in sections:
        for piece in splitter.split_text(section.text):
            piece = piece.strip()
            if len(piece) >= min_chars:
                chunks.append(Chunk(piece, section.section, section.page, len(chunks)))
    return chunks
