from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field


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


IngestStatus = Literal["created", "updated", "unchanged", "duplicate"]


class IngestResponse(BaseModel):
    source: str
    doc_type: str
    chunks: int
    status: IngestStatus
    ingested_at: str
    duplicate_of: str | None = None


class DocumentInfo(BaseModel):
    source: str
    doc_type: str
    chunks: int
    ingested_at: str
    content_hash: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, description="Consulta en lenguaje natural")
    top_k: int = Field(default=5, ge=1, le=20, description="Cantidad de chunks a devolver")
    sources: list[str] | None = Field(
        default=None, description="Limita la búsqueda a estos documentos"
    )
    doc_type: str | None = Field(
        default=None, description="Limita la búsqueda a un tipo: pdf, md o txt"
    )


class SearchHit(BaseModel):
    text: str
    source: str
    section: str | None
    page: int | None
    chunk_index: int
    score: float
