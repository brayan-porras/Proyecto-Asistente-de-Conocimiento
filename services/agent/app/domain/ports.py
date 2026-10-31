from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from app.domain.models import (
    RetrievedChunk,
    RoutingDecision,
    Turn,
    WebResult,
    WebSearchRequest,
)

T = TypeVar("T")

NO_CONTEXT = "NO_CONTEXT"


@dataclass(frozen=True)
class Out(Generic[T]):
    value: T
    tokens: int = 0


SearchFn = Callable[[str, int, list[str] | None], list[RetrievedChunk]]


class WebSearch(Protocol):
    def search(self, request: WebSearchRequest) -> list[WebResult]: ...


class DocumentCatalog(Protocol):
    def sources(self) -> list[str]: ...


class DocTasks(Protocol):
    def grade(self, question: str, chunks: list[RetrievedChunk]) -> Out[list[bool]]: ...

    def rewrite(self, question: str, previous_queries: list[str]) -> Out[str]: ...

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> Out[str]: ...


class ResearchTasks(Protocol):
    def plan_query(self, question: str, previous_queries: list[str]) -> Out[str]: ...

    def synthesize(self, question: str, results: list[WebResult]) -> Out[str]: ...


class SupervisorTasks(Protocol):
    def route(
        self, question: str, history: list[Turn], sources: list[str]
    ) -> Out[RoutingDecision]: ...

    def answer_directly(self, question: str, history: list[Turn]) -> Out[str]: ...
