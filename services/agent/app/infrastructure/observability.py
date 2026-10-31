import logging

from langchain_core.callbacks import BaseCallbackHandler

from app.infrastructure.config import Settings

logger = logging.getLogger("agent.observability")

_client = None


def build_callbacks(settings: Settings) -> list[BaseCallbackHandler]:
    global _client
    public_key = settings.langfuse_public_key
    secret_key = settings.langfuse_secret_key.get_secret_value()
    if not (public_key and secret_key):
        return []
    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler
    except ImportError:
        logger.warning("Langfuse está configurado pero el paquete no está instalado")
        return []
    _client = Langfuse(public_key=public_key, secret_key=secret_key, host=settings.langfuse_host)
    logger.info("Trazas de Langfuse activadas en %s", settings.langfuse_host)
    return [CallbackHandler(public_key=public_key)]


def flush() -> None:
    if _client is not None:
        _client.flush()
