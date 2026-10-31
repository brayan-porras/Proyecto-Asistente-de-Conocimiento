import logging
import operator
import uuid
from typing import Annotated, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.application.doc_agent import DocAgent
from app.application.research_agent import ResearchAgent
from app.application.tracing import traced
from app.domain.errors import LLMError
from app.domain.models import Answer, Citation, Route, Step, Turn
from app.domain.ports import DocumentCatalog, SupervisorTasks

logger = logging.getLogger("agent.supervisor")

NOT_FOUND_ANSWER = (
    "No encontré información suficiente ni en la documentación indexada ni en la web "
    "para responder esa pregunta. Prueba a reformularla o sube el documento que la contiene."
)


def _append_or_reset(current: list[Step], new: list[Step]) -> list[Step]:
    return [] if not new else current + new


class SupervisorState(TypedDict, total=False):
    question: str
    sources: list[str] | None
    history: Annotated[list[Turn], operator.add]
    standalone: str
    route: Route
    answer: str
    agent: str
    external: bool
    citations: list[Citation]
    steps: Annotated[list[Step], _append_or_reset]


class Assistant:
    def __init__(
        self,
        tasks: SupervisorTasks,
        docs: DocAgent,
        research: ResearchAgent,
        catalog: DocumentCatalog,
        checkpointer: BaseCheckpointSaver,
        history_turns: int = 6,
    ) -> None:
        self._tasks = tasks
        self._docs = docs
        self._research = research
        self._catalog = catalog
        self._history_turns = history_turns
        self.graph = self._build(checkpointer)

    @property
    def docs(self) -> DocAgent:
        return self._docs

    @property
    def research(self) -> ResearchAgent:
        return self._research

    def _build(self, checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
        builder = StateGraph(SupervisorState)
        builder.add_node("router", self.router)
        builder.add_node("docs_agent", self.docs_agent)
        builder.add_node("research_agent", self.research_agent)
        builder.add_node("direct", self.direct)
        builder.add_node("remember", self.remember)

        builder.add_edge(START, "router")
        builder.add_conditional_edges(
            "router",
            self._after_router,
            {"docs_agent": "docs_agent", "research_agent": "research_agent", "direct": "direct"},
        )
        builder.add_conditional_edges(
            "docs_agent",
            self._after_docs,
            {"remember": "remember", "research_agent": "research_agent"},
        )
        builder.add_edge("research_agent", "remember")
        builder.add_edge("direct", "remember")
        builder.add_edge("remember", END)
        return builder.compile(checkpointer=checkpointer)

    def ask(
        self, question: str, thread_id: str | None = None, sources: list[str] | None = None
    ) -> Answer:
        thread_id = thread_id or uuid.uuid4().hex
        state = self.graph.invoke(
            {
                "question": question,
                "sources": sources,
                "answer": "",
                "agent": "",
                "external": False,
                "citations": [],
                "steps": [],
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        return Answer(
            thread_id=thread_id,
            answer=state["answer"],
            agent=state["agent"],
            route=state["route"],
            external=state["external"],
            citations=state["citations"],
            steps=state["steps"],
        )

    def _recent_history(self, state: SupervisorState) -> list[Turn]:
        return state.get("history", [])[-self._history_turns :]

    @traced("supervisor", "router")
    def router(self, state: SupervisorState) -> dict:
        question = state["question"]
        sources = state.get("sources")
        try:
            documents = sources or self._catalog.sources()
            out = self._tasks.route(question, self._recent_history(state), documents)
        except LLMError as exc:
            logger.warning("Falló el router, se usa la documentación: %s", exc)
            return {
                "route": Route.DOCS,
                "standalone": question,
                "_detail": f"router falló, ruta por defecto docs ({exc})",
            }
        route = Route.DOCS if sources else out.value.route
        return {
            "route": route,
            "standalone": out.value.standalone_question,
            "_tokens": out.tokens,
            "_detail": f'ruta {route.value}: "{out.value.standalone_question}"',
        }

    def docs_agent(self, state: SupervisorState) -> dict:
        result = self._docs.run(state["standalone"], state.get("sources"))
        update: dict = {"steps": result["steps"]}
        if result["has_context"]:
            update.update(
                answer=result["answer"],
                citations=result["citations"],
                agent="docs",
                external=False,
            )
        return update

    def research_agent(self, state: SupervisorState) -> dict:
        result = self._research.run(state["standalone"])
        if result["has_context"]:
            return {
                "answer": result["answer"],
                "citations": result["citations"],
                "agent": "research",
                "external": True,
                "steps": result["steps"],
            }
        return {
            "answer": NOT_FOUND_ANSWER,
            "citations": [],
            "agent": "research",
            "external": False,
            "steps": result["steps"],
        }

    @traced("direct", "answer")
    def direct(self, state: SupervisorState) -> dict:
        out = self._tasks.answer_directly(state["question"], self._recent_history(state))
        return {
            "answer": out.value,
            "agent": "direct",
            "_tokens": out.tokens,
            "_detail": "respuesta sin fuentes",
        }

    @staticmethod
    def remember(state: SupervisorState) -> dict:
        return {"history": [Turn(question=state["question"], answer=state["answer"])]}

    @staticmethod
    def _after_router(state: SupervisorState) -> str:
        return {
            Route.DOCS: "docs_agent",
            Route.RESEARCH: "research_agent",
            Route.DIRECT: "direct",
        }[state["route"]]

    @staticmethod
    def _after_docs(state: SupervisorState) -> str:
        return "remember" if state["answer"] else "research_agent"
