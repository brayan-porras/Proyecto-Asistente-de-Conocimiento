from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=("../../.env", ".env"), extra="ignore")

    groq_api_key: SecretStr = SecretStr("")
    llm_model: str = "openai/gpt-oss-120b"
    grader_model: str = "openai/gpt-oss-20b"
    reasoning_effort: str = "low"
    llm_timeout_s: float = 60

    ingestion_url: str = "http://localhost:8001"
    retrieval_timeout_s: float = 30
    retrieval_top_k: int = 6
    max_rewrites: int = 2
    catalog_ttl_s: float = 30

    tavily_api_key: SecretStr = SecretStr("")
    web_timeout_s: float = 15
    web_max_results: int = 5
    max_web_searches: int = 2

    history_turns: int = 6

    langfuse_public_key: str = ""
    langfuse_secret_key: SecretStr = SecretStr("")
    langfuse_host: str = "https://cloud.langfuse.com"


@lru_cache
def get_settings() -> Settings:
    return Settings()
