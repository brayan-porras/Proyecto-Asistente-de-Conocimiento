import logging
import time

import httpx

from app.domain.errors import RetrievalError
from app.domain.models import RetrievedChunk

logger = logging.getLogger("agent.ingestion")


class IngestionClient:
    def __init__(
        self,
        base_url: str,
        timeout: float,
        catalog_ttl: float = 30,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)
        self._catalog_ttl = catalog_ttl
        self._catalog: tuple[float, list[str]] | None = None

    def search(
        self, query: str, top_k: int, sources: list[str] | None = None
    ) -> list[RetrievedChunk]:
        try:
            response = self._client.post(
                "/search", json={"query": query, "top_k": top_k, "sources": sources}
            )
            response.raise_for_status()
            return [RetrievedChunk(**hit) for hit in response.json()]
        except httpx.TimeoutException as exc:
            raise RetrievalError("Tiempo de espera agotado al buscar en la documentación") from exc
        except httpx.HTTPError as exc:
            raise RetrievalError(f"No se pudo consultar la documentación: {exc}") from exc
        except ValueError as exc:
            raise RetrievalError(f"Respuesta inválida del servicio de ingesta: {exc}") from exc

    def sources(self) -> list[str]:
        now = time.monotonic()
        if self._catalog and now - self._catalog[0] < self._catalog_ttl:
            return self._catalog[1]
        try:
            response = self._client.get("/documents")
            response.raise_for_status()
            names = sorted(doc["source"] for doc in response.json())
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            logger.warning("No se pudo listar los documentos: %s", exc)
            return []
        self._catalog = (now, names)
        return names
