import fitz

from backend.ingest.pdf import MIN_TEXT_CHARS, ingest_pdf
from backend.models import DocMeta

META = DocMeta("q2-results", "q2-results.pdf", "results", "Q2 FY26", "Q2 FY26 Results")

LONG_TEXT = "Operating margin was 21.1% in the second quarter, down 40 basis points from the prior quarter."


def make_pdf(path, page_texts):
    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page()
        if text:
            page.insert_textbox(fitz.Rect(72, 72, 520, 700), text, fontsize=11)
    doc.save(str(path))
    doc.close()


def test_extracts_text_with_one_indexed_pages_and_metadata(tmp_path):
    pdf = tmp_path / "q2-results.pdf"
    make_pdf(pdf, [LONG_TEXT, "Revenue grew 3.1% in constant currency during the quarter under review."])
    pages = ingest_pdf(pdf, META)
    assert [p.page_no for p in pages] == [1, 2]
    assert "Operating margin was 21.1%" in pages[0].text
    assert pages[0].doc_id == "q2-results"
    assert pages[0].doc_type == "results"
    assert pages[0].period == "Q2 FY26"
    assert pages[0].low_confidence is False


def test_blank_page_is_flagged_low_confidence_and_does_not_crash(tmp_path):
    pdf = tmp_path / "q2-results.pdf"
    make_pdf(pdf, [LONG_TEXT, ""])
    pages = ingest_pdf(pdf, META)
    assert len(pages) == 2
    assert pages[1].low_confidence is True
    assert pages[1].text == ""


def test_short_header_only_page_is_flagged(tmp_path):
    pdf = tmp_path / "q2-results.pdf"
    make_pdf(pdf, ["Page 3"])
    pages = ingest_pdf(pdf, META)
    assert len("Page 3") < MIN_TEXT_CHARS
    assert pages[0].low_confidence is True
    assert "Page 3" in pages[0].text
