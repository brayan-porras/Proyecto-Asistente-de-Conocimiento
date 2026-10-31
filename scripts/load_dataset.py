"""Descarga la documentación de LangGraph (más un PDF) y la ingesta en el servicio.

Uso, desde la raíz del repo y con el servicio de ingesta corriendo:
    python scripts/load_dataset.py
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

from mdx_cleaner import clean_mdx

DOCS_REPO = "langchain-ai/docs"
DOCS_PREFIX = "src/oss/langgraph/"
MIN_BYTES = 500
PDF_SOURCES = {
    "requests-docs.pdf": "https://requests.readthedocs.io/_/downloads/en/latest/pdf/",
}


def local_name(repo_path: str) -> str:
    relative = repo_path.removeprefix(DOCS_PREFIX).removesuffix(".mdx")
    return "langgraph-" + relative.replace("/", "-") + ".md"


def download(data_dir: Path) -> list[Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        tree_url = f"https://api.github.com/repos/{DOCS_REPO}/git/trees/main?recursive=1"
        response = client.get(tree_url)
        response.raise_for_status()
        entries = [
            e
            for e in response.json()["tree"]
            if e["path"].startswith(DOCS_PREFIX)
            and e["path"].endswith(".mdx")
            and e["size"] >= MIN_BYTES
        ]
        files: list[Path] = []
        for entry in entries:
            raw = client.get(
                f"https://raw.githubusercontent.com/{DOCS_REPO}/main/{entry['path']}"
            )
            raw.raise_for_status()
            target = data_dir / local_name(entry["path"])
            target.write_text(clean_mdx(raw.text), encoding="utf-8")
            files.append(target)
            print(f"descargado {target.name}")
        for name, url in PDF_SOURCES.items():
            pdf = client.get(url)
            pdf.raise_for_status()
            target = data_dir / name
            target.write_bytes(pdf.content)
            files.append(target)
            print(f"descargado {target.name}")
    return files


def ingest(files: list[Path], ingestion_url: str) -> int:
    failures = 0
    total_chunks = 0
    statuses: Counter[str] = Counter()
    started = time.perf_counter()
    with httpx.Client(timeout=900) as client:
        for path in files:
            with path.open("rb") as handle:
                response = client.post(
                    f"{ingestion_url}/ingest", files={"file": (path.name, handle)}
                )
            if response.is_success:
                body = response.json()
                total_chunks += body["chunks"]
                statuses[body["status"]] += 1
                print(f"{body['status']:9} {path.name}: {body['chunks']} chunks")
            else:
                failures += 1
                print(f"error {path.name}: {response.status_code} {response.text}")
    elapsed = time.perf_counter() - started
    summary = ", ".join(f"{count} {status}" for status, count in sorted(statuses.items()))
    print(f"\n{len(files) - failures}/{len(files)} documentos ({summary}), {total_chunks} chunks, {elapsed:.0f}s")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ingestion-url", default="http://localhost:8001")
    parser.add_argument("--data-dir", type=Path, default=Path("data/dataset"))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--only-download", action="store_true")
    args = parser.parse_args()

    if args.skip_download:
        files = sorted(
            p for p in args.data_dir.glob("*") if p.suffix in {".md", ".pdf", ".txt"}
        )
    else:
        files = download(args.data_dir)
    if args.only_download:
        return 0
    return 1 if ingest(files, args.ingestion_url) else 0


if __name__ == "__main__":
    sys.exit(main())
