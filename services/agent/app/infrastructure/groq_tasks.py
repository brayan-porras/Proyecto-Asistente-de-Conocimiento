import logging
from typing import TypeVar

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel

from app.domain.errors import LLMError
from app.domain.models import RetrievedChunk, RoutingDecision, Turn, WebResult
from app.domain.ports import Out
from app.infrastructure.config import Settings
from app.infrastructure.prompts import (
    DIRECT_SYSTEM,
    GENERATOR_SYSTEM,
    GRADER_SYSTEM,
    PLANNER_SYSTEM,
    REWRITER_SYSTEM,
    ROUTER_SYSTEM,
    SYNTHESIZER_SYSTEM,
    format_for_direct,
    format_for_generator,
    format_for_grader,
    format_for_planner,
    format_for_rewriter,
    format_for_router,
    format_for_synthesizer,
)

logger = logging.getLogger("agent.llm")

T = TypeVar("T")

_STRUCTURED_ATTEMPTS = 2


class ChunkGrade(BaseModel):
    id: int
    relevant: bool


class GradeResult(BaseModel):
    grades: list[ChunkGrade]


def _tokens(message: AIMessage) -> int:
    return int((message.usage_metadata or {}).get("total_tokens", 0))


def _clean_query(raw: str) -> str:
    lines = raw.strip().splitlines()
    return lines[0].strip().strip("\"'`").strip() if lines else ""


class GroqClient:
    def __init__(self, settings: Settings) -> None:
        def make(model: str) -> ChatGroq:
            return ChatGroq(
                model=model,
                api_key=settings.groq_api_key,
                temperature=0,
                reasoning_effort=settings.reasoning_effort,
                request_timeout=settings.llm_timeout_s,
                max_retries=5,
            )

        self.large = make(settings.llm_model)
        self.small = make(settings.grader_model)

    @staticmethod
    def structured(llm: ChatGroq, schema: type[T], system: str, human: str) -> Out[T]:
        runnable = llm.with_structured_output(schema, method="json_schema", include_raw=True)
        error = ""
        for _ in range(_STRUCTURED_ATTEMPTS):
            try:
                result = runnable.invoke([SystemMessage(system), HumanMessage(human)])
            except Exception as exc:
                error = f"Falló la llamada al LLM: {exc}"
                continue
            if result["parsed"] is None:
                error = f"Salida estructurada inválida: {result['parsing_error']}"
                continue
            return Out(result["parsed"], _tokens(result["raw"]))
        raise LLMError(error)

    @staticmethod
    def text(llm: ChatGroq, system: str, human: str) -> Out[str]:
        try:
            message = llm.invoke([SystemMessage(system), HumanMessage(human)])
        except Exception as exc:
            raise LLMError(f"Falló la llamada al LLM: {exc}") from exc
        return Out(str(message.content).strip(), _tokens(message))


class GroqDocTasks:
    def __init__(self, client: GroqClient) -> None:
        self._client = client

    def grade(self, question: str, chunks: list[RetrievedChunk]) -> Out[list[bool]]:
        out = self._client.structured(
            self._client.small, GradeResult, GRADER_SYSTEM, format_for_grader(question, chunks)
        )
        relevant_ids = {g.id for g in out.value.grades if g.relevant}
        return Out([i in relevant_ids for i in range(1, len(chunks) + 1)], out.tokens)

    def rewrite(self, question: str, previous_queries: list[str]) -> Out[str]:
        out = self._client.text(
            self._client.small, REWRITER_SYSTEM, format_for_rewriter(question, previous_queries)
        )
        return Out(_clean_query(out.value), out.tokens)

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> Out[str]:
        return self._client.text(
            self._client.large, GENERATOR_SYSTEM, format_for_generator(question, chunks)
        )


class GroqResearchTasks:
    def __init__(self, client: GroqClient) -> None:
        self._client = client

    def plan_query(self, question: str, previous_queries: list[str]) -> Out[str]:
        out = self._client.text(
            self._client.small, PLANNER_SYSTEM, format_for_planner(question, previous_queries)
        )
        return Out(_clean_query(out.value), out.tokens)

    def synthesize(self, question: str, results: list[WebResult]) -> Out[str]:
        return self._client.text(
            self._client.large, SYNTHESIZER_SYSTEM, format_for_synthesizer(question, results)
        )


class GroqSupervisorTasks:
    def __init__(self, client: GroqClient) -> None:
        self._client = client

    def route(
        self, question: str, history: list[Turn], sources: list[str]
    ) -> Out[RoutingDecision]:
        return self._client.structured(
            self._client.small,
            RoutingDecision,
            ROUTER_SYSTEM,
            format_for_router(question, history, sources),
        )

    def answer_directly(self, question: str, history: list[Turn]) -> Out[str]:
        return self._client.text(
            self._client.small, DIRECT_SYSTEM, format_for_direct(question, history)
        )
