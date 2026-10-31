import json
from collections.abc import Iterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.api.schemas import QueryRequest, QueryResponse
from app.application.supervisor import Assistant
from app.domain.errors import AgentError
from app.domain.models import Answer, Step

router = APIRouter()


def get_assistant(request: Request) -> Assistant:
    return request.app.state.assistant


@router.get("/health", summary="Estado del servicio")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse, summary="Consultar al asistente")
def query(body: QueryRequest, assistant: Assistant = Depends(get_assistant)) -> QueryResponse:
    answer = assistant.ask(body.question, body.thread_id, body.sources)
    return QueryResponse.from_answer(answer)


def _event(kind: str, data: dict) -> str:
    return f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sse(events: Iterator[tuple[str, Step | Answer | Exception]]) -> Iterator[str]:
    for kind, payload in events:
        if isinstance(payload, Step):
            yield _event(kind, payload.model_dump())
        elif isinstance(payload, Answer):
            yield _event(kind, QueryResponse.from_answer(payload).model_dump(mode="json"))
        else:
            detail = str(payload) if isinstance(payload, AgentError) else "Error interno"
            yield _event(kind, {"detail": detail})


@router.post(
    "/query/stream",
    summary="Consultar al asistente con streaming (SSE)",
    description="Emite un evento `step` por cada nodo del grafo apenas termina, y al final "
    "un evento `answer` con la misma respuesta que `/query`. Si algo falla emite `error`.",
    response_class=StreamingResponse,
)
def query_stream(
    body: QueryRequest, assistant: Assistant = Depends(get_assistant)
) -> StreamingResponse:
    events = assistant.stream(body.question, body.thread_id, body.sources)
    return StreamingResponse(
        _sse(events),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
