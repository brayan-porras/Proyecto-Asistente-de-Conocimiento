import logging
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from app.application.citations import build_web_citations, normalize_markers
from app.application.tracing import traced
from app.domain.errors import LLMError, WebSearchError
from app.domain.models import Citation, Step, WebResult, WebSearchRequest
from app.domain.ports import NO_CONTEXT, ResearchTasks, WebSearch

logger = logging.getLogger("agent.research")

_MAX_QUERY_CHARS = 300


class ResearchState(TypedDict, total=False):
    question: str
    query: str
    queries: Annotated[list[str], operator.add]
    results: list[WebResult]
    search_error: str | None
    answer: str
    citations: list[Citation]
    has_context: bool
    steps: Annotated[list[Step], operator.add]


class ResearchAgent:
    def __init__(
        self,
        web: WebSearch,
        tasks: ResearchTasks,
        max_results: int = 5,
        max_searches: int = 2,
    ) -> None:
        self._web = web
        self._tasks = tasks
        self._max_results = max_results
        self._max_searches = max_searches
        self.graph = self._build()

    def _build(self) -> CompiledStateGraph:
        builder = StateGraph(ResearchState)
        builder.add_node("planner", self.planner)
        builder.add_node("search", self.search)
        builder.add_node("synthesizer", self.synthesizer)
        builder.add_node("no_results", self.no_results)

        builder.add_edge(START, "planner")
        builder.add_edge("planner", "search")
        builder.add_conditional_edges(
            "search",
            self._after_search,
            {"synthesizer": "synthesizer", "planner": "planner", "no_results": "no_results"},
        )
        builder.add_edge("synthesizer", END)
        builder.add_edge("no_results", END)
        return builder.compile()

    def run(self, question: str) -> ResearchState:
        return self.graph.invoke({"question": question, "queries": [], "steps": []})

    @traced("research", "planner")
    def planner(self, state: ResearchState) -> dict:
        question = state["question"]
        try:
            out = self._tasks.plan_query(question, state["queries"])
        except LLMError as exc:
            logger.warning("Falló el planificador, se busca con la pregunta: %s", exc)
            return {"query": question, "_detail": f"planner falló, se usa la pregunta ({exc})"}
        return {
            "query": out.value,
            "_tokens": out.tokens,
            "_detail": f'consulta web: "{out.value}"',
        }

    @traced("research", "search")
    def search(self, state: ResearchState) -> dict:
        query = state["query"]
        try:
            request = WebSearchRequest(
                query=query[:_MAX_QUERY_CHARS], max_results=self._max_results
            )
            results = self._web.search(request)
        except (ValidationError, WebSearchError) as exc:
            logger.warning("Falló la búsqueda externa: %s", exc)
            return {
                "results": [],
                "search_error": str(exc),
                "queries": [query],
                "_detail": f"error: {exc}",
            }
        return {
            "results": results,
            "search_error": None,
            "queries": [query],
            "_detail": f'"{query}" -> {len(results)} resultados',
        }

    @traced("research", "synthesizer")
    def synthesizer(self, state: ResearchState) -> dict:
        results = state["results"]
        out = self._tasks.synthesize(state["question"], results)
        if out.value.startswith(NO_CONTEXT):
            return {
                "answer": "",
                "citations": [],
                "has_context": False,
                "_tokens": out.tokens,
                "_detail": "los resultados web no responden la pregunta",
            }
        answer = normalize_markers(out.value)
        citations = build_web_citations(answer, results)
        return {
            "answer": answer,
            "citations": citations,
            "has_context": True,
            "_tokens": out.tokens,
            "_detail": f"{len(citations)} enlaces citados",
        }

    @traced("research", "no_results")
    def no_results(self, state: ResearchState) -> dict:
        reason = state.get("search_error") or "la búsqueda no devolvió resultados"
        return {"answer": "", "citations": [], "has_context": False, "_detail": reason}

    def _after_search(self, state: ResearchState) -> str:
        if state.get("search_error"):
            return "no_results"
        if state["results"]:
            return "synthesizer"
        if len(state["queries"]) < self._max_searches:
            return "planner"
        return "no_results"
