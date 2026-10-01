import json

import fitz

from backend.index.embedder import HashingEmbedder
from backend.index.store import PageIndex
from backend.ingest.cli import run_ingest


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
