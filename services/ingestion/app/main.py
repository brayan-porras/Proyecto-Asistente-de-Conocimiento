import logging
import time
import unicodedata
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.chunking import chunk_sections
from app.config import Settings, get_settings
from app.embeddings import Embedder, SentenceTransformerEmbedder
from app.models import DocumentInfo, IngestResponse, SearchHit, SearchRequest
from app.parsers import ParseError, doc_type_of, parse_document
from app.store import VectorStore

logger = logging.getLogger("ingestion")

_STORE_ERRORS = (ResponseHandlingException, UnexpectedResponse)


def _normalize(name: str) -> str:
    # Windows y macOS pueden entregar los acentos descompuestos (NFD).
    return unicodedata.normalize("NFC", name)


@lru_cache
def get_embedder() -> Embedder:
    settings = get_settings()
    return SentenceTransformerEmbedder(
        settings.embedding_model,
        settings.embedding_query_prefix,
        settings.embedding_passage_prefix,
    )


@lru_cache
def get_store() -> VectorStore:
    settings = get_settings()
    return VectorStore(
        QdrantClient(url=settings.qdrant_url),
        settings.qdrant_collection,
        settings.hnsw_m,
        settings.hnsw_ef_construct,
        settings.search_ef,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    embedder, store = get_embedder(), get_store()
    for attempt in range(1, 11):
        try:
            store.ensure_collection(embedder.dim)
            break
        except _STORE_ERRORS:
            logger.warning("Qdrant no disponible (intento %s/10)", attempt)
            time.sleep(2)
    else:
        raise RuntimeError("No se pudo conectar con Qdrant")
    yield


app = FastAPI(
    title="Servicio de ingesta",
    description="Procesa documentos PDF, TXT y MD, los divide en chunks con metadatos "
    "y los indexa en la base vectorial.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", summary="Estado del servicio", tags=["Sistema"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=201,
    summary="Ingestar un documento",
    description="Sube un PDF, TXT o MD, lo divide en chunks y lo indexa. "
    "Si el documento ya existía, reemplaza sus chunks anteriores.",
    tags=["Documentos"],
)
def ingest(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    embedder: Embedder = Depends(get_embedder),
    store: VectorStore = Depends(get_store),
) -> IngestResponse:
    source = _normalize(Path(file.filename or "").name)
    doc_type = doc_type_of(source)
    if doc_type is None:
        raise HTTPException(415, "Solo se aceptan archivos PDF, TXT y MD")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"El archivo supera {settings.max_upload_mb} MB")

    try:
        sections = parse_document(source, data)
    except ParseError as exc:
        raise HTTPException(422, str(exc)) from exc

    chunks = chunk_sections(
        sections,
        settings.chunk_size,
        settings.chunk_overlap,
        settings.min_chunk_chars,
        settings.merge_below_chars,
    )
    if not chunks:
        raise HTTPException(422, "El documento no contiene texto extraíble")

    vectors = embedder.embed_documents([chunk.embedding_text for chunk in chunks])
    ingested_at = datetime.now(timezone.utc).isoformat()
    try:
        replaced = store.replace_document(source, doc_type, chunks, vectors, ingested_at)
    except _STORE_ERRORS as exc:
        logger.exception("Error al escribir en Qdrant")
        raise HTTPException(503, "Base vectorial no disponible") from exc

    return IngestResponse(
        source=source,
        doc_type=doc_type,
        chunks=len(chunks),
        replaced=replaced,
        ingested_at=ingested_at,
    )


@app.post(
    "/search",
    response_model=list[SearchHit],
    summary="Buscar chunks relevantes",
    description="Búsqueda vectorial por similitud. Se puede filtrar por documento "
    "(`sources`) o por tipo (`doc_type`). Devuelve los chunks con su cita "
    "(documento, sección y página).",
    tags=["Búsqueda"],
)
def search(
    request: SearchRequest,
    embedder: Embedder = Depends(get_embedder),
    store: VectorStore = Depends(get_store),
) -> list[SearchHit]:
    vector = embedder.embed_query(request.query)
    sources = [_normalize(s) for s in request.sources] if request.sources else None
    try:
        return store.search(vector, request.top_k, sources, request.doc_type)
    except _STORE_ERRORS as exc:
        logger.exception("Error al buscar en Qdrant")
        raise HTTPException(503, "Base vectorial no disponible") from exc


@app.get(
    "/documents",
    response_model=list[DocumentInfo],
    summary="Listar documentos indexados",
    tags=["Documentos"],
)
def list_documents(store: VectorStore = Depends(get_store)) -> list[DocumentInfo]:
    try:
        return store.list_documents()
    except _STORE_ERRORS as exc:
        raise HTTPException(503, "Base vectorial no disponible") from exc


@app.delete(
    "/documents/{source}",
    summary="Eliminar un documento del índice",
    tags=["Documentos"],
)
def delete_document(
    source: str, store: VectorStore = Depends(get_store)
) -> dict[str, int]:
    try:
        deleted = store.delete_document(_normalize(source))
    except _STORE_ERRORS as exc:
        raise HTTPException(503, "Base vectorial no disponible") from exc
    if not deleted:
        raise HTTPException(404, f"Documento no encontrado: {source}")
    return {"deleted_chunks": deleted}
