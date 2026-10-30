import re

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_IMPORT = re.compile(r"^import\s.+$")
_JSX_TAG_LINE = re.compile(r"^\s*</?[A-Za-z][\w.]*(\s[^>]*)?/?>\s*$")
_JSX_COMMENT_LINE = re.compile(r"^\s*\{/\*.*\*/\}\s*$")
_API_REF = re.compile(r"@\[`([^`]+)`\]")


def _front_matter_fields(text: str) -> tuple[dict[str, str], str]:
    match = _FRONT_MATTER.match(text)
    if not match:
        return {}, text
    fields = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        if value:
            fields[key.strip()] = value.strip().strip("\"'")
    return fields, text[match.end():]


def clean_mdx(text: str) -> str:
    """Convierte un .mdx de LangChain Docs en Markdown plano solo con contenido Python."""
    fields, body = _front_matter_fields(text)
    lines: list[str] = []
    in_fence = False
    in_js_block = False

    for line in body.splitlines():
        stripped = line.strip()
        if in_js_block:
            in_js_block = stripped != ":::"
            continue
        if not in_fence:
            if stripped == ":::js":
                in_js_block = True
                continue
            if stripped in (":::python", ":::"):
                continue
            if (
                _IMPORT.match(line)
                or _JSX_TAG_LINE.match(line)
                or _JSX_COMMENT_LINE.match(line)
            ):
                continue
            line = _API_REF.sub(r"`\1`", line)
        if stripped.startswith("```"):
            in_fence = not in_fence
        lines.append(line)

    header = [f"# {fields['title']}"] if "title" in fields else []
    if fields.get("description"):
        header.append(fields["description"])
    cleaned = "\n\n".join(header + ["\n".join(lines).strip()])
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip() + "\n"
