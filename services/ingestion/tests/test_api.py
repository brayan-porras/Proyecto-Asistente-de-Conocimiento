import pytest
from fastapi.testclient import TestClient

from app.main import app, get_embedder, get_store
from app.models import DocumentInfo, SearchHit
from tests.conftest import make_pdf


class FakeEmbedder:
    dim = 4

    def embed_documents(self, texts):
        return [[float(len(t) % 7), 1.0, 0.0, 0.5] for t in texts]

    def embed_query(self, query):
        return self.embed_documents([query])[0]


class FakeStore:
    def __init__(self):
        self.docs: dict[str, dict] = {}
        self.last_search: dict | None = None

    def search(self, vector, top_k, sources=None, doc_type=None):
        self.last_search = {
            "vector": vector,
            "top_k": top_k,
            "sources": sources,
            "doc_type": doc_type,
        }
        return [
            SearchHit(
                text="contenido",
                source="guia.md",
                section="Guia > Uso",
                page=None,
                chunk_index=1,
                score=0.9,
            )
        ]

    def replace_document(self, source, doc_type, chunks, vectors, ingested_at):
        replaced = source in self.docs
        self.docs[source] = {
            "doc_type": doc_type,
            "chunks": chunks,
            "vectors": vectors,
            "ingested_at": ingested_at,
        }
        return replaced

    def list_documents(self):
        return [
            DocumentInfo(
                source=s,
                doc_type=d["doc_type"],
                chunks=len(d["chunks"]),
                ingested_at=d["ingested_at"],
            )
            for s, d in sorted(self.docs.items())
        ]

    def delete_document(self, source):
        doc = self.docs.pop(source, None)
        return len(doc["chunks"]) if doc else 0


@pytest.fixture
def store():
    return FakeStore()


@pytest.fixture
def client(store):
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    app.dependency_overrides[get_store] = lambda: store
    yield TestClient(app)
    app.dependency_overrides.clear()


MD = b"# Guia\nintro del documento con texto\n## Uso\nllama a la funcion run para empezar"


def test_ingest_markdown_returns_chunk_count_and_stores_metadata(client, store):
    response = client.post("/ingest", files={"file": ("guia.md", MD)})
    assert response.status_code == 201
    body = response.json()
    assert body["source"] == "guia.md"
    assert body["doc_type"] == "md"
    assert body["chunks"] == 2
    assert body["replaced"] is False
    stored = store.docs["guia.md"]
    assert [c.section for c in stored["chunks"]] == ["Guia", "Guia > Uso"]
    assert len(stored["vectors"]) == 2


def test_ingest_pdf(client, store):
    pdf = make_pdf(["Contenido de la primera pagina del manual"])
    response = client.post("/ingest", files={"file": ("manual.pdf", pdf)})
    assert response.status_code == 201
    assert store.docs["manual.pdf"]["chunks"][0].page == 1


def test_reingest_marks_replaced(client):
    client.post("/ingest", files={"file": ("guia.md", MD)})
    response = client.post("/ingest", files={"file": ("guia.md", MD)})
    assert response.json()["replaced"] is True


def test_unsupported_type_is_415(client):
    response = client.post("/ingest", files={"file": ("datos.csv", b"a,b")})
    assert response.status_code == 415


def test_empty_document_is_422(client):
    response = client.post("/ingest", files={"file": ("vacio.txt", b"   ")})
    assert response.status_code == 422


def test_corrupt_pdf_is_422(client):
    response = client.post("/ingest", files={"file": ("roto.pdf", b"no es pdf")})
    assert response.status_code == 422


def test_decomposed_unicode_filename_is_stored_normalized(client, store):
    decomposed = "Técnica.md"
    client.post("/ingest", files={"file": (decomposed, MD)})
    assert "Técnica.md" in store.docs
    assert client.delete("/documents/" + decomposed).status_code == 200


def test_search_returns_hits_with_citation_fields(client, store):
    response = client.post("/search", json={"query": "como se usa run"})
    assert response.status_code == 200
    hit = response.json()[0]
    assert hit["source"] == "guia.md"
    assert hit["section"] == "Guia > Uso"
    assert store.last_search["top_k"] == 5


def test_search_forwards_filters(client, store):
    client.post(
        "/search",
        json={"query": "run", "top_k": 3, "sources": ["guia.md"], "doc_type": "md"},
    )
    assert store.last_search["sources"] == ["guia.md"]
    assert store.last_search["doc_type"] == "md"
    assert store.last_search["top_k"] == 3


@pytest.mark.parametrize("body", [{"query": ""}, {"query": "x", "top_k": 0}, {"query": "x", "top_k": 99}])
def test_search_validates_input(client, body):
    assert client.post("/search", json=body).status_code == 422


def test_list_and_delete_documents(client):
    client.post("/ingest", files={"file": ("guia.md", MD)})
    listed = client.get("/documents").json()
    assert [d["source"] for d in listed] == ["guia.md"]
    assert client.delete("/documents/guia.md").json() == {"deleted_chunks": 2}
    assert client.delete("/documents/guia.md").status_code == 404
