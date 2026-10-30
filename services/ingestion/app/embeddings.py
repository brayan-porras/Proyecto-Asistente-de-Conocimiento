from typing import Protocol


class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, query_prefix: str, passage_prefix: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix

    @property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, batch_size=32, normalize_embeddings=True)
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode([self._passage_prefix + text for text in texts])

    def embed_query(self, query: str) -> list[float]:
        return self._encode([self._query_prefix + query])[0]
