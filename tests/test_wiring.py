import json

import pytest

from backend import config, wiring
from backend.index.embedder import HashingEmbedder
from backend.index.store import PageIndex
from backend.models import Page


def write_data(tmp_path, files):
    data = tmp_path / "raw"
    data.mkdir()
    for f in files:
        (data / f).write_bytes(b"%PDF-1.4 fake")
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "company": "Infosys",
                "documents": [{"file": f, "doc_type": "results", "period": "Q2 FY26", "title": f} for f in files],
            }
        )
    )
    return data


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(wiring, "LocalEmbedder", HashingEmbedder)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "index.db")
    return tmp_path


def seed(db_path, doc_ids):
    index = PageIndex(db_path, HashingEmbedder())
    index.add_pages([Page(d, 1, "results", "Q2 FY26", f"Some text for {d} page one.") for d in doc_ids])


def test_build_components_rejects_index_docs_missing_from_manifest(env, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", write_data(env, ["q2-results.pdf"]))
    seed(config.DB_PATH, ["q2-results", "old-name"])
    with pytest.raises(RuntimeError, match=r"not in manifest\.json \(old-name\).*python -m backend\.ingest\.cli"):
        wiring.build_components()


def test_build_components_accepts_index_matching_manifest(env, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", write_data(env, ["q2-results.pdf"]))
    seed(config.DB_PATH, ["q2-results"])
    company, docs, index, agent = wiring.build_components()
    assert company == "Infosys" and [d.doc_id for d in docs] == ["q2-results"]


@pytest.mark.parametrize("value", [None, "", "   "])
def test_build_components_fails_fast_without_api_key(env, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    else:
        monkeypatch.setenv("ANTHROPIC_API_KEY", value)

    class MustNotLoad:
        def __init__(self, *args, **kwargs):
            raise AssertionError("the embedding model must not load without an API key")

    def no_manifest(*args, **kwargs):
        raise AssertionError("the manifest must not be read without an API key")

    monkeypatch.setattr(wiring, "LocalEmbedder", MustNotLoad)
    monkeypatch.setattr(wiring, "load_manifest", no_manifest)
    with pytest.raises(RuntimeError, match=r"^ANTHROPIC_API_KEY is not set\. Export it before starting the server\.$"):
        wiring.build_components()


def test_build_components_uses_short_timeout_and_no_sdk_retries(env, monkeypatch):
    import anthropic

    seen = {}

    class FakeAnthropic:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    monkeypatch.setattr(config, "DATA_DIR", write_data(env, ["q2-results.pdf"]))
    seed(config.DB_PATH, ["q2-results"])
    wiring.build_components()
    assert seen == {"timeout": 60.0, "max_retries": 0}
