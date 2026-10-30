from app.store import build_filter


def test_no_filters_returns_none():
    assert build_filter() is None
    assert build_filter(sources=[], doc_type=None) is None


def test_sources_filter_uses_match_any():
    flt = build_filter(sources=["a.md", "b.md"])
    assert len(flt.must) == 1
    assert flt.must[0].key == "source"
    assert flt.must[0].match.any == ["a.md", "b.md"]


def test_sources_and_doc_type_are_combined():
    flt = build_filter(sources=["a.md"], doc_type="md")
    assert {c.key for c in flt.must} == {"source", "doc_type"}
