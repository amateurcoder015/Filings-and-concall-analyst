from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from backend.models import DocMeta, Page

MIN_TEXT_CHARS = 40


def _try_ocr(page: "fitz.Page") -> str:
    """OCR needs Tesseract; without it we quietly return nothing."""
    try:
        textpage = page.get_textpage_ocr(full=True)
        return page.get_text("text", textpage=textpage).strip()
    except Exception:
        return ""


def _tables_as_markdown(page: "fitz.Page") -> list[str]:
    try:
        return [t.to_markdown() for t in page.find_tables().tables]
    except Exception:
        return []


def ingest_pdf(pdf_path: Path, meta: DocMeta) -> list[Page]:
    pages: list[Page] = []
    with fitz.open(str(pdf_path)) as doc:
        for number, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            low_confidence = False
            if len(text) < MIN_TEXT_CHARS:
                ocr_text = _try_ocr(page)
                if len(ocr_text) > len(text):
                    text = ocr_text
                low_confidence = len(text) < MIN_TEXT_CHARS
            tables = _tables_as_markdown(page)
            if tables:
                text = text + "\n\n" + "\n\n".join(tables)
            pages.append(
                Page(
                    doc_id=meta.doc_id,
                    page_no=number,
                    doc_type=meta.doc_type,
                    period=meta.period,
                    text=text,
                    low_confidence=low_confidence,
                )
            )
    return pages
