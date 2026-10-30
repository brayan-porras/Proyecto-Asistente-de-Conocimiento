import uuid
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from app.models import Chunk, DocumentInfo

_INDEXED_FIELDS = ("source", "doc_type")


def _source_filter(source: str) -> qm.Filter:
    return qm.Filter(
        must=[qm.FieldCondition(key="source", match=qm.MatchValue(value=source))]
    )


class VectorStore:
    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        hnsw_m: int,
        hnsw_ef_construct: int,
    ) -> None:
        self._client = client
        self._collection = collection
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construct = hnsw_ef_construct

    def ensure_collection(self, dim: int) -> None:
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qm.VectorParams(
                size=dim,
                distance=qm.Distance.COSINE,
                hnsw_config=qm.HnswConfigDiff(
                    m=self._hnsw_m, ef_construct=self._hnsw_ef_construct
                ),
            ),
        )
        for field in _INDEXED_FIELDS:
            self._client.create_payload_index(
                self._collection, field, qm.PayloadSchemaType.KEYWORD
            )

    def _count(self, source: str) -> int:
        return self._client.count(
            self._collection, count_filter=_source_filter(source), exact=True
        ).count

    def replace_document(
        self,
        source: str,
        doc_type: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
        ingested_at: str,
    ) -> bool:
        """Reemplaza los chunks previos del mismo documento; devuelve si existian."""
        existing = self._count(source)
        if existing:
            self._client.delete(self._collection, points_selector=_source_filter(source))
        points = [
            qm.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}#{chunk.chunk_index}")),
                vector=vector,
                payload={
                    "source": source,
                    "doc_type": doc_type,
                    "section": chunk.section,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index,
                    "ingested_at": ingested_at,
                    "text": chunk.text,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upload_points(self._collection, points, batch_size=128)
        return existing > 0

    def list_documents(self) -> list[DocumentInfo]:
        counts: dict[str, int] = defaultdict(int)
        meta: dict[str, tuple[str, str]] = {}
        offset = None
        while True:
            records, offset = self._client.scroll(
                self._collection,
                limit=256,
                offset=offset,
                with_payload=["source", "doc_type", "ingested_at"],
                with_vectors=False,
            )
            for record in records:
                payload = record.payload or {}
                source = payload["source"]
                counts[source] += 1
                meta[source] = (payload["doc_type"], payload["ingested_at"])
            if offset is None:
                break
        return [
            DocumentInfo(
                source=source,
                doc_type=meta[source][0],
                chunks=count,
                ingested_at=meta[source][1],
            )
            for source, count in sorted(counts.items())
        ]

    def delete_document(self, source: str) -> int:
        existing = self._count(source)
        if existing:
            self._client.delete(self._collection, points_selector=_source_filter(source))
        return existing
