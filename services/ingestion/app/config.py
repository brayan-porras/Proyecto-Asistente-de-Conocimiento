from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "docs"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # ~1000 caracteres son ~250 tokens, por debajo del limite de 512 de bge-small.
    # El solapamiento de 150 (15%) evita perder ideas cortadas en el limite.
    chunk_size: int = 1000
    chunk_overlap: int = 150
    min_chunk_chars: int = 20

    max_upload_mb: int = 20

    # HNSW: m=16 y ef_construct=128 dan buen recall con poco consumo de memoria.
    hnsw_m: int = 16
    hnsw_ef_construct: int = 128


@lru_cache
def get_settings() -> Settings:
    return Settings()
