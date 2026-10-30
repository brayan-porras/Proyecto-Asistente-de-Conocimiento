from mdx_cleaner import clean_mdx

SAMPLE = """---
title: Persistence
description: Memoria con checkpointers.
---

import Snippet from '/snippets/x.mdx';

{/* comentario */}
<a id="threads" />

Texto con @[`StateGraph`] dentro.

:::python
```python
graph = builder.compile()
```
:::

:::js
```typescript
const graph = builder.compile();
```
:::

<Info>
Nota importante.
</Info>

## Uso
"""


def test_title_and_description_become_heading_and_paragraph():
    out = clean_mdx(SAMPLE)
    assert out.startswith("# Persistence\n\nMemoria con checkpointers.\n")


def test_removes_imports_comments_and_tag_only_lines():
    out = clean_mdx(SAMPLE)
    assert "import Snippet" not in out
    assert "comentario" not in out
    assert "<a id" not in out
    assert "<Info>" not in out
    assert "Nota importante." in out


def test_keeps_python_and_drops_js_blocks():
    out = clean_mdx(SAMPLE)
    assert "builder.compile()" in out
    assert "typescript" not in out
    assert ":::" not in out


def test_api_reference_is_unwrapped():
    assert "`StateGraph`" in clean_mdx(SAMPLE)
    assert "@[" not in clean_mdx(SAMPLE)


def test_code_fence_content_is_not_touched():
    mdx = "---\ntitle: T\n---\n\n```html\n<div>\n</div>\n```\n"
    assert "<div>" in clean_mdx(mdx)
