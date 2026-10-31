import time
import uuid
from pathlib import Path

import streamlit as st

import api_client as api

ASSETS = Path(__file__).parent
AGENTS = {
    "docs": "Documentación",
    "research": "Investigación web",
    "direct": "Respuesta directa",
}
UPLOAD_MESSAGES = {
    "created": "{source}: indexado ({chunks} fragmentos)",
    "updated": "{source}: actualizado, el contenido cambió ({chunks} fragmentos)",
    "unchanged": "{source}: sin cambios, ya estaba indexado",
    "duplicate": "{source}: es idéntico a {duplicate_of}, no se duplicó",
}
EXAMPLES = [
    "¿Cómo se agrega un subgrafo como nodo de un grafo padre?",
    "¿Cuál es la última versión estable de LangGraph?",
    "hola, ¿qué puedes hacer?",
]

st.set_page_config(
    page_title="Asistente de Conocimiento", page_icon=str(ASSETS / "icon.png"), layout="wide"
)


def start_conversation() -> None:
    st.session_state.thread_id = uuid.uuid4().hex
    st.session_state.messages = []


def upload_section() -> None:
    files = st.file_uploader(
        "Subir documentos", type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    if files and st.button("Indexar", type="primary", width="stretch"):
        for file in files:
            with st.spinner(f"Indexando {file.name}..."):
                try:
                    info = api.upload(file.name, file.getvalue())
                except api.ApiError as exc:
                    st.error(f"{file.name}: {exc}")
                    continue
            st.success(UPLOAD_MESSAGES[info["status"]].format(**info))


def documents_section() -> list[str]:
    try:
        documents = api.list_documents()
    except api.ApiError as exc:
        st.error(str(exc))
        return []
    names = [doc["source"] for doc in documents]
    selected = st.multiselect(
        f"Buscar solo en ({len(names)} indexados)",
        names,
        placeholder="Todos los documentos",
        help="Con un filtro activo se responde solo desde esos documentos.",
    )
    with st.expander("Gestionar documentos"):
        for doc in documents:
            name_col, button_col = st.columns([5, 1])
            name_col.caption(f"{doc['source']} · {doc['chunks']} fragmentos")
            if button_col.button("✕", key=f"delete-{doc['source']}", help="Eliminar"):
                try:
                    api.delete(doc["source"])
                except api.ApiError as exc:
                    st.error(str(exc))
                else:
                    st.rerun()
    return selected


def sidebar() -> list[str]:
    with st.sidebar:
        st.image(str(ASSETS / "logo.png"), width="stretch")
        st.subheader("Asistente de Conocimiento")
        upload_section()
        st.divider()
        selected = documents_section()
        st.divider()
        st.button("Nueva conversación", on_click=start_conversation, width="stretch")
        st.caption(f"Hilo de conversación: `{st.session_state.thread_id[:8]}`")
    return selected


def render_citations(citations: list[dict]) -> None:
    for citation in citations:
        title = citation["source"]
        if citation.get("url"):
            title = f"[{title}]({citation['url']})"
        place = citation.get("section") or ""
        if citation.get("page"):
            place = f"{place} · pág. {citation['page']}" if place else f"pág. {citation['page']}"
        st.markdown(f"**[{citation['index']}]** {title}" + (f" — {place}" if place else ""))
        st.caption(citation["quote"])


def render_steps(body: dict) -> None:
    rows = [
        {
            "Nodo": f"{step['agent']}.{step['node']}",
            "ms": step["ms"],
            "Tokens": step["tokens"],
            "Detalle": step["detail"],
        }
        for step in body["steps"]
    ]
    st.dataframe(rows, hide_index=True, width="stretch")


def typewriter(text: str):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.015)


def render_answer(body: dict, live: bool = False) -> None:
    if live:
        st.write_stream(typewriter(body["answer"]))
    else:
        st.markdown(body["answer"])
    if body["external"]:
        st.info("Esta respuesta viene de la web, no de tus documentos indexados.")
    st.caption(
        f"{AGENTS.get(body['agent'], body['agent'])} · ruta {body['route']} · "
        f"{body['total_ms'] / 1000:.1f} s · {body['total_tokens']} tokens"
    )
    if body["citations"]:
        with st.expander(f"Fuentes ({len(body['citations'])})"):
            render_citations(body["citations"])
    with st.expander(f"Trazabilidad ({len(body['steps'])} pasos)"):
        render_steps(body)


def render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            elif "error" in message:
                st.error(message["error"])
            else:
                render_answer(message["body"])


def run_stream(question: str, sources: list[str], status) -> dict:
    body = None
    for kind, data in api.ask_stream(question, st.session_state.thread_id, sources):
        if kind == "step":
            status.write(f"`{data['agent']}.{data['node']}` · {data['ms']} ms · {data['detail'][:90]}")
        elif kind == "answer":
            body = data
        else:
            raise api.ApiError(data["detail"])
    if body is None:
        raise api.ApiError("La conexión terminó sin respuesta")
    return body


def answer(question: str, sources: list[str]) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        status = st.status("Pensando...", expanded=True)
        try:
            body = run_stream(question, sources, status)
        except api.ApiError as exc:
            status.update(label="No se pudo responder", state="error")
            message = {"role": "assistant", "error": str(exc)}
            st.error(message["error"])
        else:
            status.update(label=f"Listo en {body['total_ms'] / 1000:.1f} s", state="complete", expanded=False)
            message = {"role": "assistant", "body": body}
            render_answer(body, live=True)
    st.session_state.messages.append(message)


def examples() -> str | None:
    st.markdown("#### Prueba con una de estas preguntas")
    columns = st.columns(len(EXAMPLES))
    for column, example in zip(columns, EXAMPLES):
        if column.button(example, width="stretch"):
            return example
    return None


def main() -> None:
    st.session_state.setdefault("thread_id", uuid.uuid4().hex)
    st.session_state.setdefault("messages", [])
    sources = sidebar()
    st.header("Pregunta sobre tu documentación")
    render_history()
    question = st.chat_input("Escribe tu pregunta...")
    if not question and not st.session_state.messages:
        question = examples()
    if question:
        answer(question, sources)


main()
