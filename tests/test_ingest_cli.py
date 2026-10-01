import json

import fitz
import pytest

from backend.index.embedder import HashingEmbedder
from backend.index.store import PageIndex
from backend.ingest.cli import main, run_ingest
from backend.manifest import ManifestError


def make_pdf(path, texts):
    doc = fitz.open()
    for t in texts:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(72, 72, 520, 700), t, fontsize=11)
    doc.save(str(path))
    doc.close()


def test_run_ingest_indexes_every_manifest_document(tmp_path):
    make_pdf(tmp_path / "q2-results.pdf", ["Operating margin was 21.1% in the second quarter of the year.", "Revenue grew 3.1% in constant currency for the quarter."])
    make_pdf(tmp_path / "q2-concall.pdf", ["Management discussed wage hikes and visa costs at length today."])
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "company": "Infosys",
                "documents": [
                    {"file": "q2-results.pdf", "doc_type": "results", "period": "Q2 FY26", "title": "Q2 Results"},
                    {"file": "q2-concall.pdf", "doc_type": "concall", "period": "Q2 FY26", "title": "Q2 Concall"},
                ],
            }
        )
    )
    index = PageIndex(":memory:", HashingEmbedder())
    counts = run_ingest(tmp_path, index)
    assert counts == {"q2-results": 2, "q2-concall": 1}
    assert index.search("wage hikes")[0].doc_id == "q2-concall"
    assert index.get_page("q2-results", 2) is not None


def test_run_ingest_raises_manifest_error_on_bad_pdf(tmp_path):
    (tmp_path / "x.pdf").write_bytes(b"not a pdf")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "company": "Infosys",
                "documents": [
                    {"file": "x.pdf", "doc_type": "results", "period": "Q2 FY26", "title": "Bad PDF"},
                ],
            }
        )
    )
    index = PageIndex(":memory:", HashingEmbedder())
    with pytest.raises(ManifestError) as exc_info:
        run_ingest(tmp_path, index)
    assert "x.pdf" in str(exc_info.value)


def test_main_with_bad_pdf_prints_error_and_returns_1(tmp_path, monkeypatch, capsys):
    (tmp_path / "x.pdf").write_bytes(b"not a pdf")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "company": "Infosys",
                "documents": [
                    {"file": "x.pdf", "doc_type": "results", "period": "Q2 FY26", "title": "Bad PDF"},
                ],
            }
        )
    )
    monkeypatch.setattr("backend.index.embedder.LocalEmbedder", HashingEmbedder)
    db_file = tmp_path / "index.db"
    monkeypatch.setattr("backend.config.DB_PATH", str(db_file))

    result = main([str(tmp_path)])

    assert result == 1
    captured = capsys.readouterr()
    assert "Error:" in captured.out
    assert "x.pdf" in captured.out


def test_main_with_missing_manifest_returns_1_without_loading_embedder(tmp_path, monkeypatch, capsys):
    embedder_loaded = False

    class FailingEmbedder:
        def __init__(self):
            nonlocal embedder_loaded
            embedder_loaded = True
            raise AssertionError("LocalEmbedder should not be loaded for missing manifest")

    monkeypatch.setattr("backend.index.embedder.LocalEmbedder", FailingEmbedder)
    db_file = tmp_path / "index.db"
    monkeypatch.setattr("backend.config.DB_PATH", str(db_file))

    result = main([str(tmp_path)])

    assert result == 1
    assert not embedder_loaded
    captured = capsys.readouterr()
    assert "Error:" in captured.out


def test_run_ingest_is_idempotent(tmp_path):
    make_pdf(tmp_path / "q2-results.pdf", ["Operating margin was 21.1% in the second quarter of the year.", "Revenue grew 3.1% in constant currency for the quarter."])
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "company": "Infosys",
                "documents": [
                    {"file": "q2-results.pdf", "doc_type": "results", "period": "Q2 FY26", "title": "Q2 Results"},
                ],
            }
        )
    )
    index = PageIndex(":memory:", HashingEmbedder())

    counts1 = run_ingest(tmp_path, index)
    counts2 = run_ingest(tmp_path, index)

    assert counts1 == counts2
    assert index.page_counts() == counts1
