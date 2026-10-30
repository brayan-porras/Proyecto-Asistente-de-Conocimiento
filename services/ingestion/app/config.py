from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "docs"
    # Multilingue: las preguntas llegan en español y la documentacion esta en ingles.
    embedding_model: str = "intfloat/multilingual-e5-small"
    # e5 exige estos prefijos; si se cambia de modelo, hay que ajustarlos y reindexar.
    embedding_query_prefix: str = "query: "
    embedding_passage_prefix: str = "passage: "

    # ~1000 caracteres son ~250 tokens, por debajo del limite de 512 de bge-small.
    # El solapamiento de 150 (15%) evita perder ideas cortadas en el limite.
    chunk_size: int = 1000
    chunk_overlap: int = 150
    min_chunk_chars: int = 20
    # Un trozo mas corto que esto se une al siguiente de su seccion (suele ser una
    # frase introductoria que sin su lista no dice nada).
    merge_below_chars: int = 120

    max_upload_mb: int = 20

    # HNSW: m=16 y ef_construct=128 dan buen recall con poco consumo de memoria.
    hnsw_m: int = 16
    hnsw_ef_construct: int = 128
    # ef en consulta: mas candidatos explorados = mas recall y mas latencia.
    search_ef: int = 64


@lru_cache
def get_settings() -> Settings:
    return Settings()
