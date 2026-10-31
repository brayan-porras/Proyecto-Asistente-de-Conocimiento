from fastapi import APIRouter, Depends, Request

from app.api.schemas import QueryRequest, QueryResponse
from app.application.supervisor import Assistant

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
