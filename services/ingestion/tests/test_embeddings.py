import numpy as np

from app.embeddings import SentenceTransformerEmbedder


class RecordingModel:
    def __init__(self):
        self.seen: list[str] = []

    def encode(self, texts, **_):
        self.seen.extend(texts)
        return np.zeros((len(texts), 3))


def _embedder(model):
    embedder = object.__new__(SentenceTransformerEmbedder)
    embedder._model = model
    embedder._query_prefix = "query: "
    embedder._passage_prefix = "passage: "
    return embedder


def test_documents_and_queries_use_their_own_prefix():
    model = RecordingModel()
    embedder = _embedder(model)
    embedder.embed_documents(["texto a", "texto b"])
    embedder.embed_query("pregunta")
    assert model.seen == ["passage: texto a", "passage: texto b", "query: pregunta"]
