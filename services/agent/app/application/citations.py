import re

from app.domain.models import Citation, RetrievedChunk, WebResult

_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_MARKER = re.compile(r"\[(\d+)\]")
_NATIVE_MARKER = re.compile(r"【(\d+)(?:†[^】]*)?】")
_QUOTE_CHARS = 200


def normalize_markers(answer: str) -> str:
    return _NATIVE_MARKER.sub(r"[\1]", answer)


def _referenced(answer: str, total: int) -> list[int]:
    prose = _CODE_FENCE.sub("", answer)
    used = sorted({int(n) for n in _MARKER.findall(prose) if 1 <= int(n) <= total})
    return used or list(range(1, total + 1))


def _quote(text: str) -> str:
    return " ".join(text.split())[:_QUOTE_CHARS]


def build_citations(answer: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(
            index=n,
            source=chunks[n - 1].source,
            section=chunks[n - 1].section,
            page=chunks[n - 1].page,
            quote=_quote(chunks[n - 1].text),
        )
        for n in _referenced(answer, len(chunks))
    ]


def build_web_citations(answer: str, results: list[WebResult]) -> list[Citation]:
    return [
        Citation(
            index=n,
            source=results[n - 1].title,
            url=results[n - 1].url,
            quote=_quote(results[n - 1].snippet),
        )
        for n in _referenced(answer, len(results))
    ]
