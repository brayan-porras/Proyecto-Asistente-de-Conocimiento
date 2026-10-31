import json
import os
from collections.abc import Iterator
from urllib.parse import quote

import httpx

AGENT_URL = os.environ.get("AGENT_URL", "http://localhost:8002")
INGESTION_URL = os.environ.get("INGESTION_URL", "http://localhost:8001")

QUERY_TIMEOUT_S = 180
INGEST_TIMEOUT_S = 300
LIST_TIMEOUT_S = 15


class ApiError(RuntimeError):
    pass


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json()["detail"]
    except (ValueError, KeyError, TypeError):
        return f"Error HTTP {response.status_code}"
    if isinstance(detail, str):
        return detail
    return f"Error HTTP {response.status_code}: solicitud inválida"


def _call(method: str, url: str, timeout: float, **kwargs):
    try:
        response = httpx.request(method, url, timeout=timeout, **kwargs)
    except httpx.HTTPError as exc:
        raise ApiError(f"No se pudo conectar con {url}: {exc}") from exc
    if response.is_error:
        raise ApiError(_detail(response))
    return response.json()


def ask_stream(question: str, thread_id: str, sources: list[str]) -> Iterator[tuple[str, dict]]:
    payload = {"question": question, "thread_id": thread_id}
    if sources:
        payload["sources"] = sources
    url = f"{AGENT_URL}/query/stream"
    try:
        with httpx.stream("POST", url, json=payload, timeout=QUERY_TIMEOUT_S) as response:
            if response.is_error:
                response.read()
                raise ApiError(_detail(response))
            event = ""
            for line in response.iter_lines():
                if line.startswith("event: "):
                    event = line.removeprefix("event: ")
                elif line.startswith("data: "):
                    yield event, json.loads(line.removeprefix("data: "))
    except httpx.HTTPError as exc:
        raise ApiError(f"No se pudo conectar con {url}: {exc}") from exc


def list_documents() -> list[dict]:
    return _call("GET", f"{INGESTION_URL}/documents", LIST_TIMEOUT_S)


def upload(filename: str, data: bytes) -> dict:
    return _call(
        "POST", f"{INGESTION_URL}/ingest", INGEST_TIMEOUT_S, files={"file": (filename, data)}
    )


def delete(source: str) -> dict:
    return _call("DELETE", f"{INGESTION_URL}/documents/{quote(source)}", LIST_TIMEOUT_S)
