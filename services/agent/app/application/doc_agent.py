import logging
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.application.citations import build_citations, normalize_markers
from app.application.tracing import traced
from app.domain.errors import LLMError, RetrievalError
from app.domain.models import Citation, RetrievedChunk, Step
from app.domain.ports import NO_CONTEXT, DocTasks, SearchFn

logger = logging.getLogger("agent.docs")


class DocState(TypedDict, total=False):
    question: str
    sources: list[str] | None
    query: str
    queries: Annotated[list[str], operator.add]
    documents: list[RetrievedChunk]
    relevant: list[RetrievedChunk]
    rewrites: int
    retrieval_error: str | None
    answer: str
    citations: list[Citation]
    has_context: bool
    steps: Annotated[list[Step], operator.add]


class DocAgent:
    def __init__(
        self,
        search: SearchFn,
        tasks: DocTasks,
        top_k: int = 6,
        max_rewrites: int = 2,
    ) -> None:
        self._search = search
        self._tasks = tasks
        self._top_k = top_k
        self._max_rewrites = max_rewrites
        self.graph = self._build()

    def _build(self) -> CompiledStateGraph:
        builder = StateGraph(DocState)
        builder.add_node("retriever", self.retriever)
        builder.add_node("grader", self.grader)
        builder.add_node("rewriter", self.rewriter)
        builder.add_node("generator", self.generator)
        builder.add_node("no_context", self.no_context)

        builder.add_edge(START, "retriever")
        builder.add_conditional_edges(
            "retriever",
            self._after_retrieve,
            {"grader": "grader", "no_context": "no_context"},
        )
        builder.add_conditional_edges(
            "grader",
            self._after_grade,
            {"generator": "generator", "rewriter": "rewriter", "no_context": "no_context"},
        )
        builder.add_edge("rewriter", "retriever")
        builder.add_edge("generator", END)
        builder.add_edge("no_context", END)
        return builder.compile()

    def run(self, question: str, sources: list[str] | None = None) -> DocState:
        return self.graph.invoke(
            {
                "question": question,
                "query": question,
                "sources": sources,
                "queries": [],
                "rewrites": 0,
                "steps": [],
            }
        )

    @traced("docs", "retriever")
    def retriever(self, state: DocState) -> dict:
        query = state["query"]
        try:
            documents = self._search(query, self._top_k, state.get("sources"))
        except RetrievalError as exc:
            logger.warning("Falló la recuperación: %s", exc)
            return {
                "documents": [],
                "retrieval_error": str(exc),
                "queries": [query],
                "_detail": f"error: {exc}",
            }
        top = f", mejor score {documents[0].score:.2f}" if documents else ""
        return {
            "documents": documents,
            "retrieval_error": None,
            "queries": [query],
            "_detail": f'"{query}" -> {len(documents)} chunks{top}',
        }

    @traced("docs", "grader")
    def grader(self, state: DocState) -> dict:
        documents = state["documents"]
        if not documents:
            return {"relevant": [], "_detail": "sin chunks que evaluar"}
        try:
            out = self._tasks.grade(state["question"], documents)
        except LLMError as exc:
            logger.warning("Falló el grader, se usan todos los chunks: %s", exc)
            return {"relevant": documents, "_detail": f"grader falló, se usan todos ({exc})"}
        relevant = [doc for doc, keep in zip(documents, out.value, strict=True) if keep]
        return {
            "relevant": relevant,
            "_tokens": out.tokens,
            "_detail": f"{len(relevant)}/{len(documents)} chunks relevantes",
        }

    @traced("docs", "rewriter")
    def rewriter(self, state: DocState) -> dict:
        out = self._tasks.rewrite(state["question"], state["queries"])
        return {
            "query": out.value,
            "rewrites": state["rewrites"] + 1,
            "_tokens": out.tokens,
            "_detail": f'nueva consulta: "{out.value}"',
        }

    @traced("docs", "generator")
    def generator(self, state: DocState) -> dict:
        relevant = state["relevant"]
        out = self._tasks.generate(state["question"], relevant)
        if out.value.startswith(NO_CONTEXT):
            return {
                "answer": "",
                "citations": [],
                "has_context": False,
                "_tokens": out.tokens,
                "_detail": "el modelo indicó que el contexto no alcanza",
            }
        answer = normalize_markers(out.value)
        citations = build_citations(answer, relevant)
        return {
            "answer": answer,
            "citations": citations,
            "has_context": True,
            "_tokens": out.tokens,
            "_detail": f"{len(citations)} citas",
        }

    @traced("docs", "no_context")
    def no_context(self, state: DocState) -> dict:
        reason = state.get("retrieval_error") or (
            f"sin chunks relevantes tras {state['rewrites']} reescrituras"
        )
        return {"answer": "", "citations": [], "has_context": False, "_detail": reason}

    @staticmethod
    def _after_retrieve(state: DocState) -> str:
        return "no_context" if state.get("retrieval_error") else "grader"

    def _after_grade(self, state: DocState) -> str:
        if state["relevant"]:
            return "generator"
        if state["rewrites"] < self._max_rewrites:
            return "rewriter"
        return "no_context"
