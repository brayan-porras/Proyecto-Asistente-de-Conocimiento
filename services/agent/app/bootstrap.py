from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from app.application.doc_agent import DocAgent
from app.application.research_agent import ResearchAgent
from app.application.supervisor import Assistant
from app.infrastructure.config import Settings
from app.infrastructure.groq_tasks import (
    GroqClient,
    GroqDocTasks,
    GroqResearchTasks,
    GroqSupervisorTasks,
)
from app.infrastructure.ingestion_client import IngestionClient
from app.infrastructure.observability import build_callbacks
from app.infrastructure.web_search import build_web_search


_CHECKPOINT_TYPES = [
    ("app.domain.models", name) for name in ("Citation", "Step", "Route", "Turn")
]


def build_checkpointer() -> MemorySaver:
    return MemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_TYPES))


def build_assistant(settings: Settings) -> Assistant:
    if not settings.groq_api_key.get_secret_value():
        raise RuntimeError("Falta GROQ_API_KEY: defínela en el .env o como variable de entorno")

    ingestion = IngestionClient(
        settings.ingestion_url, settings.retrieval_timeout_s, settings.catalog_ttl_s
    )
    groq = GroqClient(settings)
    docs = DocAgent(
        ingestion.search, GroqDocTasks(groq), settings.retrieval_top_k, settings.max_rewrites
    )
    research = ResearchAgent(
        build_web_search(settings),
        GroqResearchTasks(groq),
        settings.web_max_results,
        settings.max_web_searches,
    )
    return Assistant(
        GroqSupervisorTasks(groq),
        docs,
        research,
        ingestion,
        build_checkpointer(),
        settings.history_turns,
        build_callbacks(settings),
    )
