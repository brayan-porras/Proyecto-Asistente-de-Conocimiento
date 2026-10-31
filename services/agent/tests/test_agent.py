import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from ddgs.exceptions import DDGSException
from fastapi.testclient import TestClient

from app.application.citations import build_citations, build_web_citations, normalize_markers
from app.application.doc_agent import DocAgent
from app.application.research_agent import ResearchAgent
from app.application.supervisor import NOT_FOUND_ANSWER, Assistant
from app.bootstrap import build_assistant, build_checkpointer
from app.domain.errors import LLMError, RetrievalError, WebSearchError
from app.domain.models import RetrievedChunk, Route, RoutingDecision, WebResult, WebSearchRequest
from app.domain.ports import Out
from app.infrastructure.config import Settings
from app.infrastructure.groq_tasks import GroqClient
from app.infrastructure.ingestion_client import IngestionClient
from app.infrastructure.web_search import DuckDuckGoSearch, TavilySearch, build_web_search
from app.main import create_app


class Seq:
    def __init__(self, *items):
        self.items = list(items)
        self.args = []

    @property
    def calls(self):
        return len(self.args)

    def __call__(self, *args):
        item = self.items[min(len(self.args), len(self.items) - 1)]
        self.args.append(args)
        if isinstance(item, Exception):
            raise item
        return item


def chunk(index=0, source="guia.md", section="Guia > Uso", page=None):
    return RetrievedChunk(
        text=f"contenido {index}",
        source=source,
        section=section,
        page=page,
        chunk_index=index,
        score=0.8,
    )


def web_result(index=0):
    return WebResult(
        title=f"Resultado {index}", url=f"https://ejemplo.com/{index}", snippet=f"texto {index}"
    )


def doc_tasks(grades, answer="Respuesta [1].", rewrites=("nueva consulta", "otra consulta")):
    grader, rewriter = Seq(*grades), Seq(*rewrites)
    return SimpleNamespace(
        grader=grader,
        rewriter=rewriter,
        grade=lambda question, chunks: Out(grader(), 10),
        rewrite=lambda question, previous: Out(rewriter(), 5),
        generate=lambda question, chunks: Out(answer, 20),
    )


def research_tasks(answer="Según la web [1].", plans=("consulta web", "consulta amplia")):
    planner = Seq(*plans)
    return SimpleNamespace(
        plan_query=lambda question, previous: Out(planner(), 4),
        synthesize=lambda question, results: Out(answer, 15),
    )


def supervisor_tasks(*routes, standalone=None):
    next_route, calls = Seq(*(routes or (Route.DOCS,))), []

    def route(question, history, sources):
        calls.append((question, list(history), list(sources)))
        decision = RoutingDecision(route=next_route(), standalone_question=standalone or question)
        return Out(decision, 6)

    return SimpleNamespace(
        route=route, answer_directly=lambda question, history: Out("Hola", 8), calls=calls
    )


def fake_web(*results):
    return SimpleNamespace(search=Seq(*results))


def queries(search):
    return [args[0] for args in search.args]


def nodes(state):
    return [step.node for step in state["steps"]]


def path(answer):
    return [f"{step.agent}.{step.node}" for step in answer.steps]


def build(
    supervisor=None,
    grades=([True],),
    web_results=(web_result(0),),
    catalog=("guia.md",),
    history_turns=6,
):
    supervisor = supervisor or supervisor_tasks()
    search, web = Seq([chunk(0)]), fake_web(list(web_results))
    assistant = Assistant(
        supervisor,
        DocAgent(search, doc_tasks(grades, "Doc [1].")),
        ResearchAgent(web, research_tasks()),
        SimpleNamespace(sources=lambda: list(catalog)),
        build_checkpointer(),
        history_turns,
    )
    return assistant, supervisor, search, web


CHUNKS = [chunk(0, "a.md", "A > Uno"), chunk(1, "b.md", "B > Dos"), chunk(2, "c.pdf", None, page=7)]


def test_citations_keep_valid_markers_and_fill_fields():
    def cited(text):
        return [c.index for c in build_citations(text, CHUNKS)]

    assert cited("Primero [1] y luego [3].") == [1, 3]
    assert cited("Dato [2] y otro [9].") == [2]
    assert cited("Usa el segundo [2].\n```python\nx = items[1]\n```") == [2]
    assert cited("sin citas") == [1, 2, 3]
    normalized = normalize_markers("Guarda estado【3†L1-L4】 y usa hilos【1†L1-L3】【2】.")
    assert normalized == "Guarda estado[3] y usa hilos[1][2]."
    citations = build_citations(normalized, CHUNKS)
    assert citations[0].source == "a.md" and citations[2].page == 7
    long = chunk(0).model_copy(update={"text": "linea\n\n   uno " * 100})
    quote = build_citations("[1]", [long])[0].quote
    assert "\n" not in quote and len(quote) <= 200


def test_web_citations_carry_the_link_and_only_referenced_results():
    citations = build_web_citations("Dato [2].", [web_result(0), web_result(1)])
    assert [(c.index, c.url, c.source, c.quote) for c in citations] == [
        (2, "https://ejemplo.com/1", "Resultado 1", "texto 1")
    ]


def test_docs_happy_path_cites_and_forwards_the_source_filter():
    search = Seq([chunk(0), chunk(1)])
    state = DocAgent(search, doc_tasks([[True, False]])).run("como se usa?", ["guia.md"])
    assert nodes(state) == ["retriever", "grader", "generator"]
    assert state["has_context"] is True
    assert [c.index for c in state["citations"]] == [1]
    assert state["relevant"] == [chunk(0)]
    assert sum(s.tokens for s in state["steps"]) == 30
    assert {s.agent for s in state["steps"]} == {"docs"}
    assert search.args[0][2] == ["guia.md"]


def test_docs_rewriter_loops_back_and_is_bounded():
    search = Seq([chunk(0)], [chunk(1)])
    state = DocAgent(search, doc_tasks([[False], [True]])).run("pregunta vaga")
    assert nodes(state) == ["retriever", "grader", "rewriter", "retriever", "grader", "generator"]
    assert queries(search) == ["pregunta vaga", "nueva consulta"]

    search, tasks = Seq([chunk(0)]), doc_tasks([[False]])
    state = DocAgent(search, tasks, max_rewrites=2).run("fuera de la doc")
    assert search.calls == 3 and tasks.rewriter.calls == 2
    assert nodes(state)[-1] == "no_context" and state["has_context"] is False


def test_docs_degrades_on_retrieval_grader_and_generator_outcomes():
    tasks = doc_tasks([[True]])
    state = DocAgent(Seq(RetrievalError("ingesta caída")), tasks).run("x")
    assert nodes(state) == ["retriever", "no_context"] and tasks.rewriter.calls == 0
    assert "ingesta caída" in state["steps"][-1].detail

    state = DocAgent(Seq([chunk(0), chunk(1)]), doc_tasks([LLMError("rate limit")])).run("x")
    assert nodes(state) == ["retriever", "grader", "generator"] and len(state["relevant"]) == 2

    state = DocAgent(Seq([chunk(0)]), doc_tasks([[True]], answer="NO_CONTEXT")).run("x")
    assert state["has_context"] is False and state["answer"] == ""


def test_research_happy_path_plans_searches_and_cites_links():
    web = fake_web([web_result(0), web_result(1)])
    state = ResearchAgent(web, research_tasks(answer="Dato [2].")).run("¿última versión?")
    assert nodes(state) == ["planner", "search", "synthesizer"]
    assert [c.url for c in state["citations"]] == ["https://ejemplo.com/1"]
    assert web.search.args[0][0].query == "consulta web"
    assert {s.agent for s in state["steps"]} == {"research"}


def test_research_replans_once_then_gives_up():
    web = fake_web([], [web_result(0)])
    state = ResearchAgent(web, research_tasks()).run("tema raro")
    assert nodes(state) == ["planner", "search", "planner", "search", "synthesizer"]
    assert [a[0].query for a in web.search.args] == ["consulta web", "consulta amplia"]

    web = fake_web([])
    state = ResearchAgent(web, research_tasks(), max_searches=2).run("nada")
    assert web.search.calls == 2 and nodes(state)[-1] == "no_results"


def test_research_never_raises_and_respects_the_tool_schema():
    web = fake_web(WebSearchError("Tiempo de espera agotado en la búsqueda web"))
    state = ResearchAgent(web, research_tasks()).run("pregunta")
    assert nodes(state) == ["planner", "search", "no_results"] and web.search.calls == 1

    web = fake_web([web_result(0)])
    state = ResearchAgent(web, research_tasks(plans=(LLMError("caído"),))).run("pregunta original")
    assert web.search.args[0][0].query == "pregunta original"

    ResearchAgent(web, research_tasks(plans=("a" * 500,))).run("pregunta")
    assert len(web.search.args[1][0].query) == 300

    state = ResearchAgent(fake_web([web_result(0)]), research_tasks(answer="NO_CONTEXT")).run("x")
    assert state["has_context"] is False


REQUEST = WebSearchRequest(query="langgraph latest version")


def tavily(handler):
    return TavilySearch("tvly-test", 5, transport=httpx.MockTransport(handler))


def raising(exc):
    def handler(*args):
        raise exc

    return handler


def test_tavily_parses_results_and_sends_the_typed_request():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["body"] = request.read().replace(b" ", b"")
        return httpx.Response(
            200, json={"results": [{"title": "T", "url": "https://x.dev", "content": "c" * 900}]}
        )

    results = tavily(handler).search(REQUEST)
    assert results[0].url == "https://x.dev" and len(results[0].snippet) == 600
    assert seen["auth"] == "Bearer tvly-test" and b'"max_results":5' in seen["body"]


def test_tavily_empty_results_are_not_an_error():
    assert tavily(lambda r: httpx.Response(200, json={"results": []})).search(REQUEST) == []


def test_tavily_failures_become_web_search_error():
    cases = [
        (raising(httpx.ReadTimeout("lento")), "Tiempo de espera"),
        (lambda r: httpx.Response(429), "Tavily"),
        (lambda r: httpx.Response(200, json={"nope": 1}), "inválida"),
    ]
    for handler, message in cases:
        with pytest.raises(WebSearchError, match=message):
            tavily(handler).search(REQUEST)


def test_duckduckgo_maps_rows_and_skips_incomplete_ones():
    rows = [
        {"title": "A", "href": "https://a.dev", "body": "texto"},
        {"title": "sin enlace", "href": "", "body": "x"},
    ]
    results = DuckDuckGoSearch(5, text_search=lambda q, n: rows).search(REQUEST)
    assert [r.url for r in results] == ["https://a.dev"]


def test_duckduckgo_distinguishes_empty_searches_from_failures():
    empty = DuckDuckGoSearch(5, text_search=raising(DDGSException("No results found.")))
    assert empty.search(REQUEST) == []
    with pytest.raises(WebSearchError):
        DuckDuckGoSearch(5, text_search=raising(DDGSException("Ratelimit 202"))).search(REQUEST)


def test_tool_schema_and_factory():
    for fields in ({"query": "ab"}, {"query": "consulta válida", "max_results": 50}):
        with pytest.raises(ValueError):
            WebSearchRequest(**fields)
    assert isinstance(build_web_search(Settings(_env_file=None, tavily_api_key="k")), TavilySearch)
    assert isinstance(
        build_web_search(Settings(_env_file=None, tavily_api_key="")), DuckDuckGoSearch
    )


HIT = {
    "text": "contenido",
    "source": "guia.md",
    "section": "Guia > Uso",
    "page": None,
    "chunk_index": 1,
    "score": 0.9,
}


def ingestion(handler):
    return IngestionClient("http://ingestion", 5, transport=httpx.MockTransport(handler))


def test_ingestion_search_parses_hits_and_sends_filters():
    seen = {}

    def handler(request):
        seen["body"] = request.read().replace(b" ", b"")
        return httpx.Response(200, json=[HIT])

    hits = ingestion(handler).search("consulta", 3, ["guia.md"])
    assert hits[0].source == "guia.md" and b'"sources":["guia.md"]' in seen["body"]
    assert ingestion(lambda r: httpx.Response(200, json=[])).search("x", 3) == []


def test_ingestion_failures_become_retrieval_error():
    cases = [
        (raising(httpx.ReadTimeout("lento")), "Tiempo de espera"),
        (lambda r: httpx.Response(503), "No se pudo consultar"),
        (lambda r: httpx.Response(200, json=[{"nope": 1}]), "inválida"),
    ]
    for handler, message in cases:
        with pytest.raises(RetrievalError, match=message):
            ingestion(handler).search("x", 3)


def test_ingestion_sources_are_sorted_cached_and_degrade_to_empty():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json=[{"source": "b.pdf"}, {"source": "a.md"}])

    client = ingestion(handler)
    assert client.sources() == client.sources() == ["a.md", "b.pdf"]
    assert calls == ["/documents"]
    assert ingestion(raising(httpx.ConnectError("sin red"))).sources() == []
    assert ingestion(lambda r: httpx.Response(200, json=[{"nope": 1}])).sources() == []


def fake_llm(*outcomes):
    invoke = Seq(*outcomes)
    llm = SimpleNamespace(
        with_structured_output=lambda schema, **kwargs: SimpleNamespace(
            invoke=lambda messages: invoke()
        )
    )
    return llm, invoke


def test_structured_output_is_retried_once_and_then_raises():
    raw = SimpleNamespace(usage_metadata={"total_tokens": 7})
    llm, invoke = fake_llm(RuntimeError("tool call validation failed"), {"parsed": "ok", "raw": raw})
    out = GroqClient.structured(llm, str, "sistema", "usuario")
    assert (out.value, out.tokens, invoke.calls) == ("ok", 7, 2)

    llm, invoke = fake_llm(RuntimeError("caído"))
    with pytest.raises(LLMError, match="caído"):
        GroqClient.structured(llm, str, "sistema", "usuario")
    assert invoke.calls == 2

    llm, _ = fake_llm({"parsed": None, "parsing_error": "json roto", "raw": raw})
    with pytest.raises(LLMError, match="json roto"):
        GroqClient.structured(llm, str, "sistema", "usuario")


def test_docs_route_answers_from_documentation_without_touching_research():
    assistant, _, _, web = build()
    answer = assistant.ask("¿cómo funciona?")
    assert answer.agent == "docs" and answer.route == Route.DOCS and answer.external is False
    assert path(answer) == ["supervisor.router", "docs.retriever", "docs.grader", "docs.generator"]
    assert web.search.calls == 0


def test_docs_without_context_falls_back_to_research_and_flags_external():
    assistant, _, _, web = build(grades=([False],))
    answer = assistant.ask("tema que no está en los docs")
    assert answer.route == Route.DOCS and answer.agent == "research" and answer.external is True
    assert answer.citations[0].url == "https://ejemplo.com/0"
    assert path(answer)[-3:] == ["research.planner", "research.search", "research.synthesizer"]
    assert web.search.calls == 1


def test_research_and_direct_routes_skip_documentation():
    assistant, _, search, _ = build(supervisor_tasks(Route.RESEARCH))
    answer = assistant.ask("¿última versión de langgraph?")
    assert answer.agent == "research" and answer.external is True and search.calls == 0

    assistant, _, search, web = build(supervisor_tasks(Route.DIRECT))
    answer = assistant.ask("hola")
    assert answer.agent == "direct" and answer.citations == []
    assert path(answer) == ["supervisor.router", "direct.answer"]
    assert search.calls == 0 and web.search.calls == 0


def test_nothing_found_anywhere_returns_a_clear_message_not_an_error():
    assistant, *_ = build(grades=([False],), web_results=())
    answer = assistant.ask("pregunta imposible")
    assert answer.answer == NOT_FOUND_ANSWER
    assert answer.external is False and answer.citations == []


def test_sources_filter_and_router_failure_pick_the_docs_route():
    assistant, supervisor, search, _ = build(supervisor_tasks(Route.DIRECT))
    answer = assistant.ask("resume esto", sources=["guia.md"])
    assert answer.route == Route.DOCS and queries(search) == ["resume esto"]
    assert supervisor.calls[0][2] == ["guia.md"]

    broken = supervisor_tasks()
    broken.route = Seq(LLMError("caído"))
    answer = build(broken)[0].ask("pregunta")
    assert answer.route == Route.DOCS and "router falló" in answer.steps[0].detail


def test_router_gets_the_indexed_documents_and_search_uses_the_standalone_question():
    supervisor = supervisor_tasks(standalone="standalone question")
    assistant, _, search, _ = build(supervisor, catalog=("a.md", "b.pdf"))
    assistant.ask("y con async?")
    assert supervisor.calls[0][2] == ["a.md", "b.pdf"]
    assert queries(search) == ["standalone question"]


def test_threads_remember_their_own_history_and_do_not_leak_steps():
    assistant, supervisor, *_ = build()
    first = assistant.ask("primera", thread_id="hilo-1")
    second = assistant.ask("segunda", thread_id="hilo-1")
    assistant.ask("de otro hilo", thread_id="hilo-2")
    assert first.thread_id == "hilo-1"
    assert supervisor.calls[0][1] == []
    assert [t.question for t in supervisor.calls[1][1]] == ["primera"]
    assert supervisor.calls[2][1] == []
    assert len(second.steps) == len(first.steps) == 4
    assert len(assistant.ask("sin hilo").thread_id) == 32


def test_a_failed_turn_does_not_reuse_the_previous_answer_and_history_is_limited():
    assistant, *_ = build(grades=([True], [False]), web_results=())
    assistant.ask("sí hay docs", thread_id="t")
    assert assistant.ask("ya no hay", thread_id="t").answer == NOT_FOUND_ANSWER

    assistant, supervisor, *_ = build(history_turns=2)
    for i in range(4):
        assistant.ask(f"p{i}", thread_id="t")
    assert [t.question for t in supervisor.calls[-1][1]] == ["p1", "p2"]


def test_graphs_export_mermaid_diagrams():
    assistant, *_ = build()
    docs = assistant.docs.graph.get_graph().draw_mermaid()
    assert "rewriter --> retriever" in docs.replace("\t", "").replace("    ", "")
    expected = {
        docs: ("retriever", "grader", "rewriter", "generator", "no_context"),
        assistant.research.graph.get_graph().draw_mermaid(): ("planner", "search", "no_results"),
        assistant.graph.get_graph().draw_mermaid(): ("router", "docs_agent", "direct", "remember"),
    }
    for diagram, names in expected.items():
        assert all(name in diagram for name in names)


@pytest.fixture
def client():
    with TestClient(create_app(build()[0])) as test_client:
        yield test_client


def test_api_health_and_query_response(client):
    assert client.get("/health").json() == {"status": "ok"}
    body = client.post("/query", json={"question": "¿cómo funciona?"}).json()
    assert body["agent"] == "docs" and body["route"] == "docs" and body["external"] is False
    assert body["nodes_traversed"] == [
        "supervisor.router",
        "docs.retriever",
        "docs.grader",
        "docs.generator",
    ]
    assert body["total_tokens"] == 6 + 10 + 20
    assert body["citations"][0]["source"] == "guia.md" and len(body["thread_id"]) == 32


def test_api_reuses_the_thread_id_and_rejects_invalid_requests(client):
    first = client.post("/query", json={"question": "uno", "thread_id": "abc"}).json()
    second = client.post("/query", json={"question": "dos", "thread_id": "abc"}).json()
    assert first["thread_id"] == second["thread_id"] == "abc"
    for payload in ({}, {"question": ""}, {"question": "x" * 2001}):
        assert client.post("/query", json=payload).status_code == 422


def test_api_llm_failure_is_a_502_with_a_message():
    supervisor = supervisor_tasks(Route.DIRECT)
    supervisor.answer_directly = Seq(LLMError("Falló la llamada al LLM: sin cuota"))
    with TestClient(create_app(build(supervisor)[0])) as test_client:
        response = test_client.post("/query", json={"question": "hola"})
    assert response.status_code == 502 and "sin cuota" in response.json()["detail"]


def test_checkpoints_roundtrip_without_warnings_and_missing_key_fails_clearly(caplog):
    assistant, *_ = build(supervisor_tasks(Route.RESEARCH, Route.DOCS))
    with caplog.at_level(logging.WARNING):
        assistant.ask("primera", thread_id="t")
        assistant.ask("segunda", thread_id="t")
    assert "unregistered type" not in caplog.text
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        build_assistant(Settings(_env_file=None, groq_api_key=""))


APP = Path(__file__).resolve().parents[1] / "app"

FORBIDDEN = {
    "domain": {"app.application", "app.infrastructure", "app.api", "fastapi", "langgraph", "httpx"},
    "application": {"app.infrastructure", "app.api", "fastapi", "httpx", "langchain_groq"},
    "infrastructure": {"app.application", "app.api", "fastapi"},
    "api": {"app.infrastructure"},
}


def imports_of(file):
    modules = set()
    for node in ast.walk(ast.parse(file.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_layers_do_not_import_what_they_must_not():
    violations = [
        f"{file.relative_to(APP)} importa {module}"
        for layer, forbidden in FORBIDDEN.items()
        for file in (APP / layer).glob("*.py")
        for module in imports_of(file)
        if any(module == f or module.startswith(f + ".") for f in forbidden)
    ]
    assert not violations, "\n".join(violations)
