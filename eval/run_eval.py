import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "agent"))

from app.application.citations import build_citations, normalize_markers  # noqa: E402
from app.application.supervisor import NOT_FOUND_ANSWER  # noqa: E402
from app.bootstrap import build_assistant  # noqa: E402
from app.domain.errors import AgentError  # noqa: E402
from app.domain.models import Turn  # noqa: E402
from app.domain.ports import NO_CONTEXT  # noqa: E402
from app.infrastructure.config import get_settings  # noqa: E402
from app.infrastructure.groq_tasks import GroqClient, GroqSupervisorTasks  # noqa: E402
from app.infrastructure.ingestion_client import IngestionClient  # noqa: E402
from app.infrastructure.observability import flush  # noqa: E402
from app.infrastructure.prompts import GENERATOR_SYSTEM, format_for_generator  # noqa: E402

JUDGE_SYSTEM = (
    "Eres un evaluador estricto de un asistente para desarrolladores. Recibes una pregunta, "
    "una referencia (la respuesta esperada o el criterio que debe cumplir) y la respuesta a "
    "evaluar. Juzga solo con la referencia, no con tu conocimiento propio.\n"
    "- score: de 1 (incorrecta o vacía) a 5 (correcta y completa).\n"
    "- correct: true si la respuesta contiene la información clave de la referencia o cumple "
    "su criterio, sin contradecirla. Un mensaje de 'no encontré información' es false.\n"
    "- hallucination: true solo si la respuesta contradice la referencia o afirma datos "
    "concretos claramente inventados. La referencia es un resumen: detalles adicionales "
    "plausibles y coherentes con ella NO son alucinación."
)


class Verdict(BaseModel):
    score: int
    correct: bool
    hallucination: bool
    reason: str


def judge(llm: GroqClient, item: dict, run: dict) -> dict:
    if run["error"]:
        return {"score": 1, "correct": False, "hallucination": False, "reason": run["error"]}
    cited = "\n".join(f"- {c}" for c in run["cited"]) or "ninguna"
    human = (
        f"Pregunta: {item['question']}\n\nReferencia: {item['reference']}\n\n"
        f"Respuesta a evaluar:\n{run['answer']}\n\nCitas incluidas (fuente y enlace):\n{cited}"
    )
    try:
        verdict = llm.structured(llm.large, Verdict, JUDGE_SYSTEM, human).value
    except AgentError as exc:
        return {"score": 1, "correct": False, "hallucination": False, "reason": f"juez falló: {exc}"}
    return {**verdict.model_dump(), "score": min(5, max(1, verdict.score))}


def failed(error: str, started: float) -> dict:
    return {
        "answer": "", "route": None, "agent": None, "external": False, "sources": [],
        "citations": 0, "cited": [], "ms": round((time.perf_counter() - started) * 1000),
        "tokens": 0, "error": error, "thread_id": None, "router_failed": False,
    }


def rate_limited(run: dict, verdict: dict) -> bool:
    text = f"{run['error'] or ''} {verdict['reason']}"
    return run["router_failed"] or "429" in text or "Rate limit" in text


def run_agentic(assistant, item: dict, threads: dict) -> dict:
    started = time.perf_counter()
    try:
        result = assistant.ask(item["question"], thread_id=threads.get(item.get("after")))
    except AgentError as exc:
        return failed(str(exc), started)
    threads[item["id"]] = result.thread_id
    return {
        "answer": result.answer, "route": result.route.value, "agent": result.agent,
        "external": result.external, "sources": [c.source for c in result.citations],
        "citations": len(result.citations),
        "cited": [f"{c.source} <{c.url}>" if c.url else c.source for c in result.citations],
        "ms": round((time.perf_counter() - started) * 1000),
        "tokens": result.total_tokens, "error": None, "thread_id": result.thread_id,
        "router_failed": any("router falló" in step.detail for step in result.steps),
    }


def run_linear(ingestion: IngestionClient, llm: GroqClient, item: dict, top_k: int) -> dict:
    started = time.perf_counter()
    try:
        chunks = ingestion.search(item["question"], top_k)
        out = llm.text(llm.large, GENERATOR_SYSTEM, format_for_generator(item["question"], chunks))
    except AgentError as exc:
        return failed(str(exc), started)
    if out.value.startswith(NO_CONTEXT):
        answer, citations = NOT_FOUND_ANSWER, []
    else:
        answer = normalize_markers(out.value)
        citations = build_citations(answer, chunks)
    return {
        "answer": answer, "route": None, "agent": "linear", "external": False,
        "sources": [c.source for c in citations], "citations": len(citations),
        "cited": [c.source for c in citations],
        "ms": round((time.perf_counter() - started) * 1000),
        "tokens": out.tokens, "error": None, "thread_id": None, "router_failed": False,
    }


def source_hit(item: dict, row: dict) -> bool | None:
    if item["category"] != "docs":
        return None
    return bool(set(row["sources"]) & set(item["expected_sources"]))


def summarize(rows: list[dict]) -> dict:
    hits = [r["source_hit"] for r in rows if r["source_hit"] is not None]
    return {
        "n": len(rows),
        "correct": mean(r["correct"] for r in rows),
        "score": mean(r["score"] for r in rows),
        "hallucination": mean(r["hallucination"] for r in rows),
        "source_hit": mean(hits) if hits else None,
        "ms": mean(r["ms"] for r in rows),
        "tokens": mean(r["tokens"] for r in rows),
    }


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def summary_table(groups: dict[str, list[dict]]) -> list[str]:
    lines = [
        "| Grupo | N | Correctas | Puntaje (1-5) | Alucinación | Fuente correcta | Latencia media | Tokens medios |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, rows in groups.items():
        if not rows:
            continue
        s = summarize(rows)
        lines.append(
            f"| {name} | {s['n']} | {pct(s['correct'])} | {s['score']:.2f} | "
            f"{pct(s['hallucination'])} | {pct(s['source_hit'])} | {s['ms'] / 1000:.1f} s | "
            f"{s['tokens']:.0f} |"
        )
    return lines


def build_report(results: list[dict]) -> str:
    agentic = [r for r in results if r["system"] == "agentic"]
    linear = [r for r in results if r["system"] == "linear"]
    lines = ["# Informe de evaluación", "", f"Preguntas: {len(agentic)} por sistema.", ""]

    lines += ["## Sistema agéntico contra RAG lineal", ""]
    lines += summary_table({"Agéntico": agentic, "RAG lineal": linear})

    lines += ["", "## Por categoría", ""]
    groups = {}
    for category in ("docs", "research", "direct"):
        groups[f"Agéntico · {category}"] = [r for r in agentic if r["category"] == category]
        groups[f"RAG lineal · {category}"] = [r for r in linear if r["category"] == category]
    lines += summary_table(groups)

    routed = sum(r["route"] == r["expected_route"] for r in agentic)
    lines += ["", "## Precisión del supervisor", "", f"Ruta correcta: {routed}/{len(agentic)} ({pct(routed / len(agentic))}).", ""]
    lines += ["| Esperada | Elegida | Preguntas |", "|---|---|---|"]
    confusion = defaultdict(list)
    for r in agentic:
        confusion[(r["expected_route"], r["route"])].append(r["id"])
    for (expected, got), ids in sorted(confusion.items(), key=lambda kv: str(kv[0])):
        lines.append(f"| {expected} | {got} | {', '.join(ids)} |")

    fallbacks = [r["id"] for r in agentic if r["route"] == "docs" and r["agent"] == "research"]
    lines += ["", f"Preguntas que pasaron de docs a la web por falta de contexto: {', '.join(fallbacks) or 'ninguna'}."]

    lines += ["", "## Casos fallidos del sistema agéntico", ""]
    bad = [r for r in agentic if not r["correct"] or r["route"] != r["expected_route"]]
    for r in bad:
        lines += [
            f"### {r['id']} · {r['question']}",
            f"- Ruta esperada/elegida: {r['expected_route']} / {r['route']} (agente: {r['agent']})",
            f"- Juez: {r['score']}/5, {r['reason']}",
            f"- Fuentes citadas: {', '.join(r['sources']) or 'ninguna'}; esperadas: {', '.join(r['expected_sources']) or 'n/a'}",
            f"- Respuesta: {r['answer'][:300].replace(chr(10), ' ')}",
            "",
        ]
    if not bad:
        lines += ["Ninguno.", ""]

    lines += ["## Detalle por pregunta", "", "| Id | Sistema | Ruta | Correcta | Puntaje | Fuente | ms | Tokens |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda r: (r["id"], r["system"])):
        hit = "n/a" if r["source_hit"] is None else ("sí" if r["source_hit"] else "no")
        lines.append(
            f"| {r['id']} | {r['system']} | {r['route'] or '-'} | {'sí' if r['correct'] else 'no'} | "
            f"{r['score']} | {hit} | {r['ms']} | {r['tokens']} |"
        )
    return "\n".join(lines) + "\n"


def load(output_dir: Path) -> list[dict]:
    path = output_dir / "results.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def save(results: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if results:
        (output_dir / "report.md").write_text(build_report(results), encoding="utf-8")


def routing_only(items: list[dict], settings) -> int:
    tasks = GroqSupervisorTasks(GroqClient(settings))
    sources = IngestionClient(settings.ingestion_url, settings.retrieval_timeout_s).sources()
    by_id = {item["id"]: item for item in items}
    rows = []
    for item in items:
        history = []
        if item.get("after"):
            previous = by_id[item["after"]]
            history = [Turn(question=previous["question"], answer=previous["reference"])]
        try:
            route = tasks.route(item["question"], history, sources).value.route.value
        except AgentError as exc:
            print(f"Falló el router en {item['id']}: {exc}")
            return 2
        rows.append((item["id"], item["expected_route"], route))
        print(f"{item['id']} esperada={item['expected_route']:8} elegida={route}")
    correct = sum(expected == got for _, expected, got in rows)
    print(f"\nRuta correcta: {correct}/{len(rows)} ({pct(correct / len(rows))})")
    wrong = [f"{i} ({e} -> {g})" for i, e, g in rows if e != g]
    print("Errores: " + (", ".join(wrong) or "ninguno"))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "eval" / "dataset.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "eval")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--routing-only", action="store_true")
    args = parser.parse_args()

    items = json.loads(args.dataset.read_text(encoding="utf-8"))[: args.limit]
    settings = get_settings()
    if args.routing_only:
        return routing_only(items, settings)
    top_k = args.top_k or settings.retrieval_top_k
    assistant = build_assistant(settings)
    llm = GroqClient(settings)
    ingestion = IngestionClient(settings.ingestion_url, settings.retrieval_timeout_s)

    results = load(args.output_dir) if args.resume else []
    done = {r["id"] for r in results}
    threads = {r["id"]: r["thread_id"] for r in results if r["thread_id"]}
    for index, item in enumerate(items, start=1):
        if item["id"] in done:
            continue
        runs = {
            "agentic": run_agentic(assistant, item, threads),
            "linear": run_linear(ingestion, llm, item, top_k),
        }
        rows = []
        for system, run in runs.items():
            verdict = judge(llm, item, run)
            if rate_limited(run, verdict):
                save(results, args.output_dir)
                cause = run["error"] or verdict["reason"] if not run["router_failed"] else "el router falló"
                print(f"Cuota de Groq agotada en {item['id']} ({system}): {cause}")
                print("Vuelve a ejecutar con --resume cuando pase el tiempo indicado")
                return 2
            row = {
                "id": item["id"], "system": system, "category": item["category"],
                "question": item["question"], "expected_route": item["expected_route"],
                "expected_sources": item["expected_sources"], **run, **verdict,
            }
            row["source_hit"] = source_hit(item, row)
            rows.append(row)
            print(f"[{index}/{len(items)}] {item['id']} {system:8} correcta={row['correct']} ruta={row['route']}")
        results.extend(rows)
        save(results, args.output_dir)

    flush()
    print(f"Informe en {args.output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
