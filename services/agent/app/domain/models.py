from enum import StrEnum

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    text: str
    source: str
    section: str | None = None
    page: int | None = None
    chunk_index: int
    score: float


class WebResult(BaseModel):
    title: str
    url: str
    snippet: str


class WebSearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=300)
    max_results: int = Field(default=5, ge=1, le=10)


class Citation(BaseModel):
    index: int
    source: str
    section: str | None = None
    page: int | None = None
    url: str | None = None
    quote: str


class Step(BaseModel):
    agent: str
    node: str
    ms: int
    tokens: int = 0
    detail: str = ""


class Route(StrEnum):
    DOCS = "docs"
    RESEARCH = "research"
    DIRECT = "direct"


class RoutingDecision(BaseModel):
    route: Route = Field(description="Which agent should handle the question")
    standalone_question: str = Field(
        description="The question rewritten so it is understandable without the chat history"
    )


class Turn(BaseModel):
    question: str
    answer: str


class Answer(BaseModel):
    thread_id: str
    answer: str
    agent: str
    route: Route
    external: bool
    citations: list[Citation]
    steps: list[Step]

    @property
    def total_ms(self) -> int:
        return sum(step.ms for step in self.steps)

    @property
    def total_tokens(self) -> int:
        return sum(step.tokens for step in self.steps)
