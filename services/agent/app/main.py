import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.application.supervisor import Assistant
from app.bootstrap import build_assistant
from app.domain.errors import AgentError
from app.infrastructure.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("agent")


def create_app(assistant: Assistant | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not hasattr(app.state, "assistant"):
            app.state.assistant = build_assistant(get_settings())
        yield

    app = FastAPI(
        title="Asistente de Conocimiento - API de consulta",
        description="Supervisor LangGraph con subagentes de documentación e investigación.",
        lifespan=lifespan,
    )
    if assistant is not None:
        app.state.assistant = assistant
    app.include_router(router)

    @app.exception_handler(AgentError)
    async def agent_error_handler(_: Request, exc: AgentError) -> JSONResponse:
        logger.error("Error en el agente: %s", exc)
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    return app


app = create_app()
