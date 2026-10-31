import logging
from collections.abc import Callable

import httpx
from ddgs import DDGS
from ddgs.exceptions import DDGSException

from app.domain.errors import WebSearchError
from app.domain.models import WebResult, WebSearchRequest
from app.domain.ports import WebSearch
from app.infrastructure.config import Settings

logger = logging.getLogger("agent.web")

TAVILY_URL = "https://api.tavily.com"
_SNIPPET_CHARS = 600

TextSearchFn = Callable[[str, int], list[dict]]


class TavilySearch:
    def __init__(
        self,
        api_key: str,
        timeout: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=TAVILY_URL,
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
            transport=transport,
        )

    def search(self, request: WebSearchRequest) -> list[WebResult]:
        try:
            response = self._client.post(
                "/search",
                json={
                    "query": request.query,
                    "max_results": request.max_results,
                    "search_depth": "basic",
                },
            )
            response.raise_for_status()
            rows = response.json()["results"]
            return [
                WebResult(
                    title=row["title"], url=row["url"], snippet=row["content"][:_SNIPPET_CHARS]
                )
                for row in rows
            ]
        except httpx.TimeoutException as exc:
            raise WebSearchError("Tiempo de espera agotado en la búsqueda web") from exc
        except httpx.HTTPError as exc:
            raise WebSearchError(f"Falló la búsqueda web con Tavily: {exc}") from exc
        except (ValueError, KeyError, TypeError) as exc:
            raise WebSearchError(f"Respuesta inválida de Tavily: {exc}") from exc


def _ddgs_text(timeout: float) -> TextSearchFn:
    def text(query: str, max_results: int) -> list[dict]:
        return DDGS(timeout=int(timeout)).text(query, max_results=max_results)

    return text


class DuckDuckGoSearch:
    def __init__(self, timeout: float, text_search: TextSearchFn | None = None) -> None:
        self._text = text_search or _ddgs_text(timeout)

    def search(self, request: WebSearchRequest) -> list[WebResult]:
        try:
            rows = self._text(request.query, request.max_results)
        except Exception as exc:
            if isinstance(exc, DDGSException) and "no results" in str(exc).lower():
                return []
            raise WebSearchError(f"Falló la búsqueda web con DuckDuckGo: {exc}") from exc
        return [
            WebResult(
                title=row["title"], url=row["href"], snippet=row.get("body", "")[:_SNIPPET_CHARS]
            )
            for row in rows
            if row.get("href") and row.get("title")
        ]


def build_web_search(settings: Settings) -> WebSearch:
    key = settings.tavily_api_key.get_secret_value()
    if key:
        logger.info("Búsqueda web: Tavily")
        return TavilySearch(key, settings.web_timeout_s)
    logger.info("Búsqueda web: DuckDuckGo (sin TAVILY_API_KEY)")
    return DuckDuckGoSearch(settings.web_timeout_s)
