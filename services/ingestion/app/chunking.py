from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models import Chunk, Section


def _merge_short_pieces(pieces: list[str], merge_below: int) -> list[str]:
    """Une cada trozo corto con el siguiente; el ultimo se deja como esta."""
    merged: list[str] = []
    carry = ""
    for index, piece in enumerate(pieces):
        if carry:
            piece = f"{carry}\n\n{piece}"
            carry = ""
        if len(piece) < merge_below and index < len(pieces) - 1:
            carry = piece
            continue
        merged.append(piece)
    return merged


def chunk_sections(
    sections: list[Section],
    chunk_size: int,
    chunk_overlap: int,
    min_chars: int,
    merge_below: int = 0,
) -> list[Chunk]:
    """Cada seccion se parte por separado, asi un chunk nunca cruza encabezados ni paginas."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks: list[Chunk] = []
    for section in sections:
        pieces = [p.strip() for p in splitter.split_text(section.text)]
        for piece in _merge_short_pieces([p for p in pieces if p], merge_below):
            if len(piece) >= min_chars:
                chunks.append(Chunk(piece, section.section, section.page, len(chunks)))
    return chunks
