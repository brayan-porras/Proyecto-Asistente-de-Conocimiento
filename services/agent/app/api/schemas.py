from typing import Self

from pydantic import BaseModel, Field

from app.domain.models import Answer, Citation, Route, Step


class QueryRequest(BaseModel):
    question: str = Field(
        min_length=1, max_length=2000, description="Pregunta del usuario, en cualquier idioma"
    )
    thread_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        description="Identificador de la conversación; si se omite se crea uno nuevo",
    )
    sources: list[str] | None = Field(
        default=None, description="Limita la consulta a estos documentos indexados"
    )


class QueryResponse(BaseModel):
    thread_id: str = Field(description="Se reenvía en la siguiente pregunta para mantener el contexto")
    answer: str
    agent: str = Field(description="Agente que respondió: docs, research o direct")
    route: Route = Field(description="Decisión del supervisor, antes de cualquier fallback")
    external: bool = Field(description="True si la respuesta proviene de fuentes externas (web)")
    citations: list[Citation]
    nodes_traversed: list[str] = Field(description="Nodos recorridos, como agente.nodo")
    steps: list[Step]
    total_ms: int
    total_tokens: int

    @classmethod
    def from_answer(cls, answer: Answer) -> Self:
        return cls(
            thread_id=answer.thread_id,
            answer=answer.answer,
            agent=answer.agent,
            route=answer.route,
            external=answer.external,
            citations=answer.citations,
            nodes_traversed=[f"{s.agent}.{s.node}" for s in answer.steps],
            steps=answer.steps,
            total_ms=answer.total_ms,
            total_tokens=answer.total_tokens,
        )
