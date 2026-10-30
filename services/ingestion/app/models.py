from dataclasses import dataclass

from pydantic import BaseModel


@dataclass(frozen=True)
class Section:
    """Texto contiguo de un documento con su ubicacion (encabezados o pagina)."""

    text: str
    section: str | None = None
    page: int | None = None


@dataclass(frozen=True)
class Chunk:
    text: str
    section: str | None
    page: int | None
    chunk_index: int

    @property
    def embedding_text(self) -> str:
        # El encabezado aporta contexto que el fragmento por si solo no tiene.
        return f"{self.section}\n\n{self.text}" if self.section else self.text


class IngestResponse(BaseModel):
    source: str
    doc_type: str
    chunks: int
    replaced: bool
    ingested_at: str


class DocumentInfo(BaseModel):
    source: str
    doc_type: str
    chunks: int
    ingested_at: str
