import argparse
import sys
import uuid

from app.application.supervisor import Assistant
from app.bootstrap import build_assistant
from app.infrastructure.observability import flush
from app.domain.errors import AgentError
from app.domain.models import Answer
from app.infrastructure.config import get_settings


def print_answer(result: Answer) -> None:
    print("\nPasos:")
    for step in result.steps:
        label = f"{step.agent}.{step.node}"
        print(f"  {label:<22} {step.ms:>6} ms {step.tokens:>6} tok  {step.detail}")
    print(f"  {'total':<22} {result.total_ms:>6} ms {result.total_tokens:>6} tok")

    origin = "web (externa)" if result.external else result.agent
    print(f"\nRespuesta [{origin}]:\n{result.answer}")
    for citation in result.citations:
        where = citation.url or citation.section or (f"página {citation.page}" if citation.page else "")
        print(f"  [{citation.index}] {citation.source} — {where}")


def print_mermaid(assistant: Assistant, which: str) -> None:
    graphs = {
        "supervisor": assistant.graph,
        "docs": assistant.docs.graph,
        "research": assistant.research.graph,
    }
    print(graphs[which].get_graph().draw_mermaid())


def ask(assistant: Assistant, question: str, thread_id: str, sources: list[str] | None) -> None:
    try:
        print_answer(assistant.ask(question, thread_id, sources))
    except AgentError as exc:
        print(f"Error: {exc}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--thread", default=uuid.uuid4().hex, help="id de la conversación")
    parser.add_argument("--source", action="append", help="limita la búsqueda a un documento")
    parser.add_argument("--chat", action="store_true", help="modo conversación")
    parser.add_argument("--mermaid", choices=["supervisor", "docs", "research"])
    args = parser.parse_args()

    assistant = build_assistant(get_settings())
    if args.mermaid:
        print_mermaid(assistant, args.mermaid)
        return 0
    if args.chat:
        print(f"Conversación {args.thread} (línea vacía para salir)")
        while question := input("\n> ").strip():
            ask(assistant, question, args.thread, args.source)
        return 0
    if not args.question:
        parser.error("falta la pregunta (o usa --chat)")
    ask(assistant, args.question, args.thread, args.source)
    return 0


if __name__ == "__main__":
    code = main()
    flush()
    sys.exit(code)
