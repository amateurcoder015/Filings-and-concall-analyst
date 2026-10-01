# Filings & Concall Analyst v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A cited Q&A tool over one company's filings (Infosys as placeholder) where every claim carries `(doc, page, quote)` and each quote is machine-verified against the page text.

**Architecture:** PDFs are parsed once into pages and indexed in SQLite (FTS5 keyword + local embeddings, fused with reciprocal rank fusion). A Claude tool-loop agent (`search`, `read_page`, `submit_answer`) returns claims with verbatim quotes; a deterministic verifier outside the model checks each quote against the cited page. FastAPI serves the result to a three-pane React UI with a PDF source viewer that highlights the quote.

**Tech Stack:** Python 3.12, FastAPI, SQLite (FTS5), PyMuPDF, rapidfuzz, sentence-transformers (local), Anthropic SDK; React + TypeScript + Tailwind + Vite, react-pdf, vitest; pytest.

**Spec:** `docs/superpowers/specs/2026-10-01-filings-analyst-design.md`

## Global Constraints

- Python 3.10+ (dev machine has 3.12); backend is FastAPI; index is SQLite with a local embedding model; no hosted vector DB.
- Agent tool-call budget is capped at about 8 steps per question.
- Claims must carry a verbatim quote; claims without a quote are rejected and re-requested once.
- A claim is displayed as fact only when its quote passes verification; otherwise it is visibly marked unverified.
- When nothing relevant is found the answer is "not in the loaded filings"; the agent never guesses.
- Out-of-scope questions (prices, advice, other companies) are declined and pointed back to the loaded documents.
- UI carries an educational-use disclaimer; the tool gives no investment advice.
- `data/raw/` (source PDFs) and `data/index/` stay out of git; secrets stay out of git.
- v1 success: at least 90% of golden-set answers correct; 100% of displayed claims verified or visibly marked unverified; zero fabricated quotes displayed as verified.
- v1 out of scope: briefing and quarter diff, auto-fetching from BSE/NSE, multiple companies, auth.
- Every commit message ends with the trailer `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` (shown below as a second `-m`).
- Work inside `web/filings-analyst/` only. Do not touch sibling project folders.
- Milestone pushes go to `origin main` (https://github.com/amateurcoder015/Filings-and-concall-analyst) so the repo shows progression.

## Review Focus

Inputs and conditions the spec implies but does not spell out, most likely first. Each has a pinning test in the owning task.

1. **Search query full of FTS5 syntax** (`margin" AND (* OR NEAR`): must return results or an empty list, never a SQL error. (Task 3)
2. **Quote differs from page text only by curly quotes, non-breaking spaces, line breaks or hyphenation:** must still verify. (Task 4)
3. **Quote identical except one changed number** (21.1% vs 21.7%): must fail, never "weak". Wrong numbers are the worst failure for a finance tool. (Task 4)
4. **Claim cites a document or page that does not exist:** marked failed, no crash. (Task 4)
5. **Empty or whitespace-only question:** rejected with 422 and no model call. (Task 7)
6. **Manifest lists a file that is missing:** clear error naming the file, not a stack trace. (Task 1)

---

## File Structure

```
filings-analyst/
├── requirements.txt
├── pytest.ini
├── backend/
│   ├── __init__.py
│   ├── config.py              env-driven paths and model name
│   ├── models.py              dataclasses shared by every unit
│   ├── manifest.py            manifest.json loader + validation
│   ├── wiring.py              builds index + agent from config (server and eval share it)
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── pdf.py             PDF -> Page[]
│   │   └── cli.py             `python -m backend.ingest.cli`
│   ├── index/
│   │   ├── __init__.py
│   │   ├── embedder.py        HashingEmbedder, LocalEmbedder
│   │   └── store.py           PageIndex: SQLite + hybrid search
│   ├── verify/
│   │   ├── __init__.py
│   │   └── verifier.py        quote verification
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── tools.py           tool schemas + system prompt
│   │   └── agent.py           FilingsAgent tool loop
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py            create_app(...)
│   │   └── server.py          `app = ...` for uvicorn
│   └── eval/
│       ├── __init__.py
│       └── run_eval.py        golden-set harness
├── evals/golden.json
├── tests/
│   ├── conftest.py
│   ├── fakes.py
│   └── test_*.py
└── frontend/                  Vite React TS app
```

---

### Task 1: Project scaffold, models, manifest loader

**Files:**
- Create: `requirements.txt`, `pytest.ini`, `backend/__init__.py`, `backend/config.py`, `backend/models.py`, `backend/manifest.py`, `tests/conftest.py`, `tests/test_manifest.py`

**Interfaces:**
- Produces: `backend.models` dataclasses (`DocMeta`, `Page`, `PageHit`, `Claim`, `Answer`, `VerifiedClaim`, `VerifiedAnswer`, `DOC_TYPES`); `backend.manifest.load_manifest(directory: Path) -> tuple[str, list[DocMeta]]`; `backend.manifest.ManifestError`; `backend.config` constants `ROOT`, `DATA_DIR`, `DB_PATH`, `MODEL`.

- [ ] **Step 1: Create the environment and dependency files**

`requirements.txt`:
```
pymupdf>=1.24
fastapi>=0.110
uvicorn>=0.29
anthropic>=0.40
numpy>=1.26
rapidfuzz>=3.6
sentence-transformers>=2.7
httpx>=0.27
pytest>=8
```

`pytest.ini`:
```ini
[pytest]
pythonpath = .
testpaths = tests
```

Run:
```bash
cd /Users/TonyStark/Desktop/web/filings-analyst
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p backend tests && touch backend/__init__.py
```
Expected: installs succeed (sentence-transformers pulls torch, several minutes).

- [ ] **Step 2: Write `backend/config.py` and `backend/models.py`**

`backend/config.py`:
```python
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("FILINGS_DATA_DIR", ROOT / "data" / "raw" / "infosys"))
DB_PATH = Path(os.environ.get("FILINGS_DB", ROOT / "data" / "index" / "index.db"))
MODEL = os.environ.get("FILINGS_MODEL", "claude-sonnet-5-5")
```

`backend/models.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field

DOC_TYPES = ("annual_report", "results", "concall")


@dataclass(frozen=True)
class DocMeta:
    doc_id: str
    file: str
    doc_type: str
    period: str
    title: str


@dataclass(frozen=True)
class Page:
    doc_id: str
    page_no: int
    doc_type: str
    period: str
    text: str
    low_confidence: bool = False


@dataclass(frozen=True)
class PageHit:
    doc_id: str
    page_no: int
    doc_type: str
    period: str
    snippet: str
    score: float


@dataclass(frozen=True)
class Claim:
    text: str
    doc_id: str
    page_no: int
    quote: str


@dataclass
class Answer:
    summary: str
    claims: list[Claim] = field(default_factory=list)
    not_found: bool = False


@dataclass(frozen=True)
class VerifiedClaim:
    claim: Claim
    status: str  # "verified" | "weak" | "failed"


@dataclass
class VerifiedAnswer:
    summary: str
    claims: list[VerifiedClaim]
    not_found: bool

    @property
    def mostly_unverified(self) -> bool:
        if not self.claims:
            return False
        bad = sum(1 for c in self.claims if c.status != "verified")
        return bad * 2 > len(self.claims)
```

- [ ] **Step 3: Write the failing manifest tests**

`tests/test_manifest.py`:
```python
import json

import pytest

from backend.manifest import ManifestError, load_manifest


def write_manifest(directory, documents, company="Infosys"):
    (directory / "manifest.json").write_text(
        json.dumps({"company": company, "documents": documents})
    )


def entry(file="ar-fy25.pdf", doc_type="annual_report", period="FY25", title="Annual Report FY25"):
    return {"file": file, "doc_type": doc_type, "period": period, "title": title}


def test_loads_valid_manifest(tmp_path):
    (tmp_path / "ar-fy25.pdf").write_bytes(b"%PDF")
    write_manifest(tmp_path, [entry()])
    company, docs = load_manifest(tmp_path)
    assert company == "Infosys"
    assert docs[0].doc_id == "ar-fy25"
    assert docs[0].doc_type == "annual_report"
    assert docs[0].period == "FY25"


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(ManifestError, match="manifest.json not found"):
        load_manifest(tmp_path)


def test_listed_file_missing_names_the_file(tmp_path):
    write_manifest(tmp_path, [entry(file="nope.pdf")])
    with pytest.raises(ManifestError, match="nope.pdf"):
        load_manifest(tmp_path)


def test_unknown_doc_type_rejected(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    write_manifest(tmp_path, [entry(file="a.pdf", doc_type="tweet")])
    with pytest.raises(ManifestError, match="doc_type"):
        load_manifest(tmp_path)


def test_missing_field_rejected(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    bad = entry(file="a.pdf")
    del bad["period"]
    write_manifest(tmp_path, [bad])
    with pytest.raises(ManifestError, match="period"):
        load_manifest(tmp_path)


def test_duplicate_doc_ids_rejected(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    write_manifest(tmp_path, [entry(file="a.pdf"), entry(file="a.pdf")])
    with pytest.raises(ManifestError, match="duplicate"):
        load_manifest(tmp_path)


def test_invalid_json_rejected(tmp_path):
    (tmp_path / "manifest.json").write_text("{not json")
    with pytest.raises(ManifestError, match="not valid JSON"):
        load_manifest(tmp_path)
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.manifest'`

- [ ] **Step 5: Implement `backend/manifest.py`**

```python
from __future__ import annotations

import json
from pathlib import Path

from backend.models import DOC_TYPES, DocMeta

REQUIRED_FIELDS = ("file", "doc_type", "period", "title")


class ManifestError(Exception):
    pass


def load_manifest(directory: Path) -> tuple[str, list[DocMeta]]:
    directory = Path(directory)
    path = directory / "manifest.json"
    if not path.exists():
        raise ManifestError(f"manifest.json not found in {directory}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest.json is not valid JSON: {exc}") from exc

    company = data.get("company")
    if not company:
        raise ManifestError("manifest.json needs a 'company' field")

    docs: list[DocMeta] = []
    seen: set[str] = set()
    for entry in data.get("documents", []):
        for key in REQUIRED_FIELDS:
            if key not in entry:
                raise ManifestError(f"document entry is missing '{key}': {entry}")
        if entry["doc_type"] not in DOC_TYPES:
            raise ManifestError(
                f"doc_type '{entry['doc_type']}' is not one of {list(DOC_TYPES)}"
            )
        if not (directory / entry["file"]).exists():
            raise ManifestError(f"listed file not found: {entry['file']}")
        doc_id = Path(entry["file"]).stem
        if doc_id in seen:
            raise ManifestError(f"duplicate document id '{doc_id}'")
        seen.add(doc_id)
        docs.append(
            DocMeta(
                doc_id=doc_id,
                file=entry["file"],
                doc_type=entry["doc_type"],
                period=entry["period"],
                title=entry["title"],
            )
        )
    if not docs:
        raise ManifestError("manifest.json lists no documents")
    return company, docs
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_manifest.py -v`
Expected: 7 passed

- [ ] **Step 7: Commit**

```bash
git add requirements.txt pytest.ini backend tests
git commit -m "feat: project scaffold, shared models and manifest loader" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: PDF ingest

**Files:**
- Create: `backend/ingest/__init__.py`, `backend/ingest/pdf.py`, `tests/test_ingest_pdf.py`

**Interfaces:**
- Consumes: `DocMeta`, `Page` from `backend.models`.
- Produces: `ingest_pdf(pdf_path: Path, meta: DocMeta) -> list[Page]` (pages 1-indexed, tables appended as markdown, `low_confidence=True` for pages under 40 chars of text even after OCR attempt). `MIN_TEXT_CHARS = 40`.

- [ ] **Step 1: Write the failing tests**

`tests/test_ingest_pdf.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ingest_pdf.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.ingest'`

- [ ] **Step 3: Implement**

`backend/ingest/__init__.py`: empty file.

`backend/ingest/pdf.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ingest_pdf.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add backend/ingest tests/test_ingest_pdf.py
git commit -m "feat: PDF ingest with table extraction and OCR fallback" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Page index with hybrid search

**Files:**
- Create: `backend/index/__init__.py`, `backend/index/embedder.py`, `backend/index/store.py`, `tests/test_index.py`

**Interfaces:**
- Consumes: `Page`, `PageHit`.
- Produces: `HashingEmbedder(dim=256)` and `LocalEmbedder(model_name=...)`, both with `embed(texts: list[str]) -> np.ndarray` (normalized float32, shape `(n, dim)`); `PageIndex(db_path, embedder)` with `add_pages(pages)`, `search(query, top_k=8, doc_type=None, period=None) -> list[PageHit]`, `get_page(doc_id, page_no) -> Page | None`, `page_counts() -> dict[str, int]`. `":memory:"` is a valid `db_path`.

- [ ] **Step 1: Write the failing tests**

`tests/test_index.py`:
```python
import pytest

from backend.index.embedder import HashingEmbedder
from backend.index.store import PageIndex
from backend.models import Page


def make_pages():
    return [
        Page("q2-results", 1, "results", "Q2 FY26", "Operating margin was 21.1% in the second quarter, down 40 basis points."),
        Page("q2-results", 2, "results", "Q2 FY26", "Revenue grew 3.1% in constant currency driven by financial services."),
        Page("q2-concall", 1, "concall", "Q2 FY26", "Management said wage hikes and visa costs pressured margins this quarter."),
        Page("ar-fy25", 1, "annual_report", "FY25", "The board recommended a final dividend of 22 rupees per share."),
    ]


@pytest.fixture()
def index():
    idx = PageIndex(":memory:", HashingEmbedder())
    idx.add_pages(make_pages())
    return idx


def test_keyword_match_ranks_relevant_page_first(index):
    hits = index.search("operating margin second quarter")
    assert hits[0].doc_id == "q2-results" and hits[0].page_no == 1


def test_doc_type_filter(index):
    hits = index.search("margin", doc_type="concall")
    assert hits and all(h.doc_type == "concall" for h in hits)


def test_period_filter(index):
    hits = index.search("dividend", period="FY25")
    assert hits and all(h.period == "FY25" for h in hits)
    assert index.search("dividend", period="Q2 FY26") == [] or all(
        h.period == "Q2 FY26" for h in index.search("dividend", period="Q2 FY26")
    )


def test_fts_syntax_in_query_never_raises(index):
    for nasty in ['margin" AND (* OR NEAR', "'; DROP TABLE pages; --", "a* -b ^c", '"', "()"]:
        assert isinstance(index.search(nasty), list)


def test_empty_query_returns_nothing(index):
    assert index.search("   ") == []


def test_get_page_returns_full_text_or_none(index):
    page = index.get_page("q2-results", 2)
    assert page is not None and "constant currency" in page.text
    assert index.get_page("q2-results", 99) is None
    assert index.get_page("nope", 1) is None


def test_reingesting_a_doc_replaces_pages_instead_of_duplicating(index):
    index.add_pages(make_pages())
    assert index.page_counts() == {"q2-results": 2, "q2-concall": 1, "ar-fy25": 1}


def test_hit_snippet_is_single_line_and_bounded(index):
    hit = index.search("margin")[0]
    assert "\n" not in hit.snippet and len(hit.snippet) <= 300
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_index.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.index'`

- [ ] **Step 3: Implement the embedders**

`backend/index/__init__.py`: empty file.

`backend/index/embedder.py`:
```python
from __future__ import annotations

import re
import zlib

import numpy as np


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder. No model download; used in tests and as a fallback."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                out[i, zlib.crc32(token.encode()) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class LocalEmbedder:
    """Local sentence-transformers model; downloads weights on first use."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
        return np.asarray(vectors, dtype=np.float32)
```

- [ ] **Step 4: Implement the index**

`backend/index/store.py`:
```python
from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

from backend.models import Page, PageHit

RRF_K = 60
CANDIDATES = 50
SNIPPET_CHARS = 300

SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY,
    doc_id TEXT NOT NULL,
    page_no INTEGER NOT NULL,
    doc_type TEXT NOT NULL,
    period TEXT NOT NULL,
    text TEXT NOT NULL,
    low_confidence INTEGER NOT NULL DEFAULT 0,
    embedding BLOB NOT NULL,
    UNIQUE (doc_id, page_no)
);
CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(text);
"""


class PageIndex:
    def __init__(self, db_path, embedder):
        self._embedder = embedder
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # FastAPI serves requests from a thread pool, so the connection is shared across threads.
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.executescript(SCHEMA)

    def add_pages(self, pages: list[Page]) -> None:
        if not pages:
            return
        vectors = self._embedder.embed([p.text for p in pages])
        with self._db:
            for page, vector in zip(pages, vectors):
                row = self._db.execute(
                    "SELECT id FROM pages WHERE doc_id = ? AND page_no = ?",
                    (page.doc_id, page.page_no),
                ).fetchone()
                if row:
                    self._db.execute("DELETE FROM pages_fts WHERE rowid = ?", (row[0],))
                    self._db.execute("DELETE FROM pages WHERE id = ?", (row[0],))
                cursor = self._db.execute(
                    "INSERT INTO pages (doc_id, page_no, doc_type, period, text, low_confidence, embedding)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        page.doc_id,
                        page.page_no,
                        page.doc_type,
                        page.period,
                        page.text,
                        int(page.low_confidence),
                        np.asarray(vector, dtype=np.float32).tobytes(),
                    ),
                )
                self._db.execute(
                    "INSERT INTO pages_fts (rowid, text) VALUES (?, ?)",
                    (cursor.lastrowid, page.text),
                )

    def get_page(self, doc_id: str, page_no: int) -> Page | None:
        row = self._db.execute(
            "SELECT doc_id, page_no, doc_type, period, text, low_confidence"
            " FROM pages WHERE doc_id = ? AND page_no = ?",
            (doc_id, page_no),
        ).fetchone()
        if not row:
            return None
        return Page(row[0], row[1], row[2], row[3], row[4], bool(row[5]))

    def page_counts(self) -> dict[str, int]:
        rows = self._db.execute("SELECT doc_id, COUNT(*) FROM pages GROUP BY doc_id").fetchall()
        return {doc_id: count for doc_id, count in rows}

    def search(
        self,
        query: str,
        top_k: int = 8,
        doc_type: str | None = None,
        period: str | None = None,
    ) -> list[PageHit]:
        query = query.strip()
        if not query:
            return []
        keyword = self._keyword_rank(query, doc_type, period)
        semantic = self._semantic_rank(query, doc_type, period)

        scores: dict[int, float] = defaultdict(float)
        for ranking in (keyword, semantic):
            for rank, page_id in enumerate(ranking):
                scores[page_id] += 1.0 / (RRF_K + rank + 1)

        top = sorted(scores.items(), key=lambda item: -item[1])[:top_k]
        hits: list[PageHit] = []
        for page_id, score in top:
            row = self._db.execute(
                "SELECT doc_id, page_no, doc_type, period, text FROM pages WHERE id = ?",
                (page_id,),
            ).fetchone()
            snippet = re.sub(r"\s+", " ", row[4]).strip()[:SNIPPET_CHARS]
            hits.append(PageHit(row[0], row[1], row[2], row[3], snippet, score))
        return hits

    @staticmethod
    def _filter_sql(doc_type: str | None, period: str | None) -> tuple[str, list]:
        clauses, params = "", []
        if doc_type:
            clauses += " AND p.doc_type = ?"
            params.append(doc_type)
        if period:
            clauses += " AND p.period = ?"
            params.append(period)
        return clauses, params

    def _keyword_rank(self, query: str, doc_type, period) -> list[int]:
        tokens = re.findall(r"\w+", query)
        if not tokens:
            return []
        match = " OR ".join(f'"{token}"' for token in tokens)
        clauses, params = self._filter_sql(doc_type, period)
        rows = self._db.execute(
            "SELECT p.id FROM pages_fts JOIN pages p ON p.id = pages_fts.rowid"
            " WHERE pages_fts MATCH ?" + clauses + " ORDER BY bm25(pages_fts) LIMIT ?",
            [match, *params, CANDIDATES],
        ).fetchall()
        return [r[0] for r in rows]

    def _semantic_rank(self, query: str, doc_type, period) -> list[int]:
        clauses, params = self._filter_sql(doc_type, period)
        rows = self._db.execute(
            "SELECT p.id, p.embedding FROM pages p WHERE 1 = 1" + clauses, params
        ).fetchall()
        if not rows:
            return []
        matrix = np.vstack([np.frombuffer(blob, dtype=np.float32) for _, blob in rows])
        query_vector = np.asarray(self._embedder.embed([query])[0], dtype=np.float32)
        similarities = matrix @ query_vector
        order = np.argsort(-similarities)[:CANDIDATES]
        return [rows[i][0] for i in order if similarities[i] > 0]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_index.py -v`
Expected: 8 passed

- [ ] **Step 6: Commit and push the first milestone**

```bash
git add backend/index tests/test_index.py
git commit -m "feat: SQLite page index with hybrid keyword and embedding search" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
git push origin main
```

---

### Task 4: Quote verifier

> **Superseded:** the verifier code below was hardened over five review rounds; the authoritative implementation is `backend/verify/verifier.py` and its tests.

**Files:**
- Create: `backend/verify/__init__.py`, `backend/verify/verifier.py`, `tests/test_verifier.py`

**Interfaces:**
- Consumes: `Claim`, `Answer`, `VerifiedClaim`, `VerifiedAnswer`; any object with `get_page(doc_id, page_no) -> Page | None` (i.e. `PageIndex`).
- Produces: `verify_claim(claim, page_text: str | None) -> str` returning `"verified" | "weak" | "failed"`; `verify(answer, index) -> VerifiedAnswer`. Constants `MIN_QUOTE_CHARS = 12`, `WEAK_THRESHOLD = 90`.

- [ ] **Step 1: Write the failing tests**

`tests/test_verifier.py`:
```python
from backend.models import Answer, Claim, Page, VerifiedAnswer, VerifiedClaim
from backend.verify.verifier import MIN_QUOTE_CHARS, verify, verify_claim

PAGE = (
    "Operating margin for the quarter was 21.1%, down 40 basis\n"
    "points sequentially. Management said the under-\n"
    "lying demand environment remains “cautious” and that clients’ budgets are tight."
)


def claim(quote, doc_id="q2", page_no=1):
    return Claim(text="t", doc_id=doc_id, page_no=page_no, quote=quote)


def test_exact_quote_is_verified():
    assert verify_claim(claim("Operating margin for the quarter was 21.1%"), PAGE) == "verified"


def test_line_break_and_whitespace_differences_still_verify():
    assert verify_claim(claim("down 40 basis points sequentially"), PAGE) == "verified"


def test_curly_quotes_and_nbsp_still_verify():
    quote = "remains \"cautious\" and that clients' budgets are tight"
    assert verify_claim(claim(quote), PAGE) == "verified"


def test_hyphenation_across_line_break_still_verifies():
    assert verify_claim(claim("underlying demand environment remains"), PAGE) == "verified"


def test_minor_text_difference_is_weak_not_verified():
    page = "The operating margin for the quarter was 21.1 per cent, down sharply."
    quote = "The operating margin for the quarter was 21.1 percent, down sharply."
    assert verify_claim(claim(quote), page) == "weak"


def test_changed_number_fails_even_when_text_is_otherwise_identical():
    quote = "Operating margin for the quarter was 21.7%, down 40 basis points"
    assert verify_claim(claim(quote), PAGE) == "failed"


def test_sign_flip_fails():
    page = "Free cash flow conversion moved by -5% versus the previous quarter overall."
    assert verify_claim(claim("Free cash flow conversion moved by 5% versus the previous quarter overall."), page) == "failed"


def test_fabricated_quote_fails():
    assert verify_claim(claim("The company announced a large share buyback programme"), PAGE) == "failed"


def test_missing_page_fails():
    assert verify_claim(claim("Operating margin for the quarter was 21.1%"), None) == "failed"


def test_too_short_quote_fails():
    assert len("21.1%") < MIN_QUOTE_CHARS
    assert verify_claim(claim("21.1%"), PAGE) == "failed"


class FakeIndex:
    def __init__(self, pages):
        self._pages = {(p.doc_id, p.page_no): p for p in pages}

    def get_page(self, doc_id, page_no):
        return self._pages.get((doc_id, page_no))


def test_quote_on_a_different_page_than_cited_fails():
    index = FakeIndex([Page("q2", 1, "results", "Q2", "Nothing relevant here at all, just filler text."), Page("q2", 2, "results", "Q2", PAGE)])
    answer = Answer(summary="s", claims=[claim("Operating margin for the quarter was 21.1%", "q2", 1)])
    result = verify(answer, index)
    assert result.claims[0].status == "failed"


def test_claim_citing_nonexistent_doc_or_page_fails_without_crashing():
    index = FakeIndex([Page("q2", 1, "results", "Q2", PAGE)])
    answer = Answer(
        summary="s",
        claims=[claim("Operating margin for the quarter was 21.1%", "ghost", 1), claim("Operating margin for the quarter was 21.1%", "q2", 500)],
    )
    result = verify(answer, index)
    assert [c.status for c in result.claims] == ["failed", "failed"]


def test_verify_preserves_summary_and_not_found():
    index = FakeIndex([])
    result = verify(Answer(summary="Not in the loaded filings.", claims=[], not_found=True), index)
    assert result.not_found is True and result.summary == "Not in the loaded filings."
    assert result.claims == []


def test_mostly_unverified_flag():
    c = claim("x" * 20)
    mixed = VerifiedAnswer("s", [VerifiedClaim(c, "verified"), VerifiedClaim(c, "failed"), VerifiedClaim(c, "failed")], False)
    fine = VerifiedAnswer("s", [VerifiedClaim(c, "verified"), VerifiedClaim(c, "verified"), VerifiedClaim(c, "failed")], False)
    assert mixed.mostly_unverified is True
    assert fine.mostly_unverified is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_verifier.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.verify'`

- [ ] **Step 3: Implement**

`backend/verify/__init__.py`: empty file.

`backend/verify/verifier.py`:
```python
from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

from backend.models import Answer, Claim, VerifiedAnswer, VerifiedClaim

MIN_QUOTE_CHARS = 12
WEAK_THRESHOLD = 90

_PUNCTUATION = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "­": "",
    }
)
# A leading minus is part of the number: "-5%" and "5%" are different claims.
_NUMBER = re.compile(r"(?<![\w-])-?\d+(?:[.,]\d+)*")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_PUNCTUATION).lower()
    # A hyphen glued to a word, optionally followed by a line break, is joined up
    # ("under-\nlying" == "underlying"). A hyphen preceded by a space is a minus sign and is kept.
    text = re.sub(r"(?<=\w)-\s*(?=\w)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def verify_claim(claim: Claim, page_text: str | None) -> str:
    if page_text is None:
        return "failed"
    quote = _normalize(claim.quote)
    if len(quote) < MIN_QUOTE_CHARS:
        return "failed"
    page = _normalize(page_text)
    if quote in page:
        return "verified"
    # Near-miss (OCR noise, spelling): only acceptable if every number in the quote is on the page.
    page_numbers = set(_NUMBER.findall(page))
    if not all(number in page_numbers for number in _NUMBER.findall(quote)):
        return "failed"
    if fuzz.partial_ratio(quote, page) >= WEAK_THRESHOLD:
        return "weak"
    return "failed"


def verify(answer: Answer, index) -> VerifiedAnswer:
    verified: list[VerifiedClaim] = []
    for claim in answer.claims:
        page = index.get_page(claim.doc_id, claim.page_no)
        verified.append(VerifiedClaim(claim, verify_claim(claim, page.text if page else None)))
    return VerifiedAnswer(summary=answer.summary, claims=verified, not_found=answer.not_found)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_verifier.py -v`
Expected: 14 passed. If a number-related test fails, fix `_NUMBER` or the normalization, never the test: a changed figure or flipped sign must never reach "weak".

- [ ] **Step 5: Commit**

```bash
git add backend/verify tests/test_verifier.py
git commit -m "feat: deterministic quote verifier with number and sign guards" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Agent tool loop

**Files:**
- Create: `backend/agent/__init__.py`, `backend/agent/tools.py`, `backend/agent/agent.py`, `tests/fakes.py`, `tests/test_agent.py`
- Modify: `tests/conftest.py` (create)

**Interfaces:**
- Consumes: `PageIndex` (`search`, `get_page`), `Answer`, `Claim`.
- Produces: `FilingsAgent(client, index, model, company="Infosys")` with `ask(question: str, period: str | None = None) -> Answer`; `AgentError`; `NOT_FOUND_TEXT = "This is not in the loaded filings."`; `MAX_STEPS = 8`. `client` is anything with `client.messages.create(**kwargs)` matching the Anthropic SDK.

- [ ] **Step 1: Write the test helpers**

`tests/fakes.py`:
```python
from types import SimpleNamespace as NS


def tool_use(name, input, id="t1"):
    return NS(type="tool_use", id=id, name=name, input=input)


def text(value):
    return NS(type="text", text=value)


def response(*blocks):
    return NS(content=list(blocks), stop_reason="tool_use")


class ScriptedClient:
    """Stands in for anthropic.Anthropic: replays a script of responses or exceptions."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        kwargs["messages"] = list(kwargs["messages"])
        self.calls.append(kwargs)
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
```

`tests/conftest.py`:
```python
import pytest

from backend.index.embedder import HashingEmbedder
from backend.index.store import PageIndex
from backend.models import Page

MARGIN_PAGE = "Operating margin was 21.1% in the second quarter, down 40 basis points from the prior quarter."


@pytest.fixture()
def index():
    idx = PageIndex(":memory:", HashingEmbedder())
    idx.add_pages(
        [
            Page("q2-results", 1, "results", "Q2 FY26", MARGIN_PAGE),
            Page("q2-results", 2, "results", "Q2 FY26", "Revenue grew 3.1% in constant currency driven by financial services."),
            Page("q2-concall", 1, "concall", "Q2 FY26", "Management said wage hikes and visa costs pressured margins this quarter.", True),
        ]
    )
    return idx
```

- [ ] **Step 2: Write the failing tests**

`tests/test_agent.py`:
```python
import pytest

from backend.agent.agent import MAX_STEPS, NOT_FOUND_TEXT, AgentError, FilingsAgent
from tests.conftest import MARGIN_PAGE
from tests.fakes import ScriptedClient, response, text, tool_use

GOOD_CLAIM = {"text": "Margin was 21.1%.", "doc_id": "q2-results", "page_no": 1, "quote": MARGIN_PAGE}


def submit(claims=None, not_found=False, summary="Summary.", id="s1"):
    return response(tool_use("submit_answer", {"summary": summary, "claims": claims or [], "not_found": not_found}, id=id))


def make_agent(index, script):
    client = ScriptedClient(script)
    return FilingsAgent(client, index, model="test-model"), client


def test_search_then_read_then_submit_returns_answer(index):
    agent, client = make_agent(
        index,
        [
            response(tool_use("search", {"query": "operating margin"}, id="a")),
            response(tool_use("read_page", {"doc_id": "q2-results", "page_no": 1}, id="b")),
            submit([GOOD_CLAIM]),
        ],
    )
    answer = agent.ask("What was operating margin?")
    assert answer.not_found is False
    assert answer.claims[0].doc_id == "q2-results" and answer.claims[0].page_no == 1
    assert answer.claims[0].quote == MARGIN_PAGE
    # the search result was fed back to the model as a tool_result
    second_call_messages = client.calls[1]["messages"]
    tool_result = second_call_messages[-1]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "a"
    assert "q2-results p.1" in tool_result["content"]


def test_claim_without_quote_is_rejected_once_then_accepted(index):
    bad = dict(GOOD_CLAIM, quote="")
    agent, client = make_agent(index, [submit([bad], id="s1"), submit([GOOD_CLAIM], id="s2")])
    answer = agent.ask("q")
    assert answer.claims[0].quote == MARGIN_PAGE
    rejection = client.calls[1]["messages"][-1]["content"][0]
    assert rejection["is_error"] is True and rejection["tool_use_id"] == "s1"


def test_two_bad_submissions_fall_back_to_not_found(index):
    bad = dict(GOOD_CLAIM, quote="")
    agent, _ = make_agent(index, [submit([bad], id="s1"), submit([bad], id="s2")])
    answer = agent.ask("q")
    assert answer.not_found is True and answer.claims == []


def test_not_found_submission_passes_through(index):
    agent, _ = make_agent(index, [submit(not_found=True, summary="")])
    answer = agent.ask("What is the share price today?")
    assert answer.not_found is True
    assert answer.summary == NOT_FOUND_TEXT
    assert answer.claims == []


def test_answer_without_claims_and_not_found_false_is_rejected(index):
    agent, client = make_agent(index, [submit([], id="s1"), submit([GOOD_CLAIM], id="s2")])
    answer = agent.ask("q")
    assert len(answer.claims) == 1
    assert client.calls[1]["messages"][-1]["content"][0]["is_error"] is True


def test_plain_text_reply_is_nudged_to_submit(index):
    agent, client = make_agent(index, [response(text("Margin was 21%.")), submit([GOOD_CLAIM])])
    answer = agent.ask("q")
    assert len(answer.claims) == 1
    nudge = client.calls[1]["messages"][-1]
    assert nudge["role"] == "user" and "submit_answer" in nudge["content"]


def test_step_cap_stops_a_model_that_never_submits(index):
    script = [response(tool_use("search", {"query": "margin"}, id=f"s{i}")) for i in range(MAX_STEPS + 3)]
    agent, client = make_agent(index, script)
    answer = agent.ask("q")
    assert answer.not_found is True
    assert len(client.calls) == MAX_STEPS


def test_unknown_page_and_unknown_tool_return_errors_not_exceptions(index):
    agent, client = make_agent(
        index,
        [
            response(
                tool_use("read_page", {"doc_id": "ghost", "page_no": 9}, id="a"),
                tool_use("teleport", {}, id="b"),
            ),
            submit([GOOD_CLAIM]),
        ],
    )
    agent.ask("q")
    results = client.calls[1]["messages"][-1]["content"]
    assert all(r["is_error"] for r in results)


def test_low_confidence_page_is_flagged_to_the_model(index):
    agent, client = make_agent(
        index,
        [response(tool_use("read_page", {"doc_id": "q2-concall", "page_no": 1}, id="a")), submit(not_found=True)],
    )
    agent.ask("q")
    content = client.calls[1]["messages"][-1]["content"][0]["content"]
    assert "low-confidence" in content


def test_api_failure_is_retried_once(index):
    agent, _ = make_agent(index, [RuntimeError("boom"), submit([GOOD_CLAIM])])
    assert len(agent.ask("q").claims) == 1


def test_api_failing_twice_raises_agent_error(index):
    agent, _ = make_agent(index, [RuntimeError("boom"), RuntimeError("boom again")])
    with pytest.raises(AgentError):
        agent.ask("q")


def test_period_and_company_are_in_the_system_prompt(index):
    agent, client = make_agent(index, [submit(not_found=True)])
    agent.ask("q", period="Q2 FY26")
    system = client.calls[0]["system"]
    assert "Infosys" in system and "Q2 FY26" in system
    assert client.calls[0]["model"] == "test-model"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.agent'`

- [ ] **Step 4: Implement tools and prompt**

`backend/agent/__init__.py`: empty file.

`backend/agent/tools.py`:
```python
from __future__ import annotations

TOOLS = [
    {
        "name": "search",
        "description": (
            "Search the loaded filings for pages relevant to a query. Returns up to 8 pages with "
            "doc_id, page number, document type, period and a snippet. Use short keyword-style queries."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "doc_type": {"type": "string", "enum": ["annual_report", "results", "concall"]},
                "period": {"type": "string", "description": "e.g. 'Q2 FY26' or 'FY25'"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_page",
        "description": "Read the full text of one page. Always read a page before quoting from it.",
        "input_schema": {
            "type": "object",
            "properties": {"doc_id": {"type": "string"}, "page_no": {"type": "integer"}},
            "required": ["doc_id", "page_no"],
        },
    },
    {
        "name": "submit_answer",
        "description": (
            "Submit the final answer. Every claim needs doc_id, page_no and a verbatim quote copied "
            "exactly from that page. If the filings do not contain the answer, set not_found=true."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "One or two sentence plain-language answer."},
                "not_found": {"type": "boolean"},
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "The claim in your own words."},
                            "doc_id": {"type": "string"},
                            "page_no": {"type": "integer"},
                            "quote": {"type": "string", "description": "Verbatim text from the page, at least one full sentence or phrase."},
                        },
                        "required": ["text", "doc_id", "page_no", "quote"],
                    },
                },
            },
            "required": ["summary"],
        },
    },
]


def build_system_prompt(company: str, period: str | None) -> str:
    focus = (
        f"The user is focused on {period}. Prefer that period's documents and use other periods only for comparison."
        if period
        else "No period is selected; use the most relevant documents."
    )
    return f"""You are a research assistant for {company}'s own filings: annual reports, quarterly results and concall transcripts. You can only see the documents loaded in this system.

{focus}

Rules:
- Find evidence with search and read_page, then finish by calling submit_answer. Always finish with submit_answer.
- Every claim must cite doc_id, page_no and a verbatim quote copied exactly from that page's text. Never paraphrase inside a quote, and never quote a page you have not read.
- Quote enough to include the exact figures you rely on.
- If the loaded filings do not contain the answer, call submit_answer with not_found=true. Do not guess and do not use outside knowledge.
- Decline questions about live prices, valuation, buy/sell advice, forecasts you cannot ground in the documents, or other companies: call submit_answer with not_found=true and explain in summary that you only answer from {company}'s loaded filings.
- You provide research for education only, never investment advice.
- Be efficient: you have a limited number of steps."""
```

- [ ] **Step 5: Implement the agent**

`backend/agent/agent.py`:
```python
from __future__ import annotations

from backend.agent.tools import TOOLS, build_system_prompt
from backend.models import Answer, Claim

MAX_STEPS = 8
MAX_PAGE_CHARS = 8000
NOT_FOUND_TEXT = "This is not in the loaded filings."
GIVE_UP_TEXT = "I could not produce a properly cited answer. Try rephrasing or narrowing the question."


class AgentError(Exception):
    pass


def parse_answer(data: dict) -> Answer | str:
    """Return an Answer, or a message telling the model what to fix."""
    summary = str(data.get("summary", "")).strip()
    if data.get("not_found"):
        return Answer(summary=summary or NOT_FOUND_TEXT, claims=[], not_found=True)
    raw_claims = data.get("claims") or []
    if not raw_claims:
        return (
            "No claims given. Every answer needs claims with doc_id, page_no and a verbatim quote, "
            "or set not_found=true."
        )
    claims: list[Claim] = []
    for position, raw in enumerate(raw_claims, start=1):
        try:
            claim = Claim(
                text=str(raw["text"]).strip(),
                doc_id=str(raw["doc_id"]).strip(),
                page_no=int(raw["page_no"]),
                quote=str(raw["quote"]).strip(),
            )
        except (KeyError, TypeError, ValueError):
            return f"Claim {position} is malformed: each needs text, doc_id, page_no (integer) and quote."
        if not claim.text or not claim.quote:
            return f"Claim {position} has an empty text or quote; every claim needs a verbatim quote."
        claims.append(claim)
    return Answer(summary=summary, claims=claims, not_found=False)


class FilingsAgent:
    def __init__(self, client, index, model: str, company: str = "Infosys"):
        self._client = client
        self._index = index
        self._model = model
        self._company = company

    def ask(self, question: str, period: str | None = None) -> Answer:
        system = build_system_prompt(self._company, period)
        messages: list[dict] = [{"role": "user", "content": question}]
        rejected = 0
        for _ in range(MAX_STEPS):
            response = self._create(system, messages)
            messages.append({"role": "assistant", "content": response.content})
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                messages.append(
                    {"role": "user", "content": "Finish by calling the submit_answer tool with your answer."}
                )
                continue
            results = []
            for block in tool_uses:
                if block.name == "submit_answer":
                    parsed = parse_answer(block.input)
                    if isinstance(parsed, Answer):
                        return parsed
                    rejected += 1
                    if rejected > 1:
                        return Answer(summary=GIVE_UP_TEXT, claims=[], not_found=True)
                    content, is_error = parsed, True
                else:
                    content, is_error = self._run_tool(block.name, block.input)
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": content, "is_error": is_error}
                )
            messages.append({"role": "user", "content": results})
        return Answer(summary=GIVE_UP_TEXT, claims=[], not_found=True)

    def _create(self, system: str, messages: list[dict]):
        last_error: Exception | None = None
        for _ in range(2):
            try:
                return self._client.messages.create(
                    model=self._model,
                    max_tokens=2048,
                    system=system,
                    tools=TOOLS,
                    messages=messages,
                )
            except Exception as exc:  # network, rate limit, overload: retry once, then surface
                last_error = exc
        raise AgentError(f"Claude API call failed: {last_error}") from last_error

    def _run_tool(self, name: str, args: dict) -> tuple[str, bool]:
        if name == "search":
            hits = self._index.search(
                str(args.get("query", "")),
                top_k=8,
                doc_type=args.get("doc_type"),
                period=args.get("period"),
            )
            if not hits:
                return "No matching pages.", False
            lines = [f"{h.doc_id} p.{h.page_no} [{h.doc_type} {h.period}] {h.snippet}" for h in hits]
            return "\n".join(lines), False
        if name == "read_page":
            try:
                page = self._index.get_page(str(args["doc_id"]), int(args["page_no"]))
            except (KeyError, TypeError, ValueError):
                return "read_page needs doc_id and an integer page_no.", True
            if page is None:
                return "No such page.", True
            note = "[low-confidence page: text may be incomplete or OCR noise]\n" if page.low_confidence else ""
            return note + page.text[:MAX_PAGE_CHARS], False
        return f"Unknown tool '{name}'.", True
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_agent.py -v`
Expected: 12 passed

- [ ] **Step 7: Commit**

```bash
git add backend/agent tests/fakes.py tests/conftest.py tests/test_agent.py
git commit -m "feat: Claude tool-loop agent with search, read_page and structured submit" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Ingest CLI

**Files:**
- Create: `backend/ingest/cli.py`, `tests/test_ingest_cli.py`

**Interfaces:**
- Consumes: `load_manifest`, `ingest_pdf`, `PageIndex`, `LocalEmbedder`, `config.DATA_DIR`, `config.DB_PATH`.
- Produces: `run_ingest(directory: Path, index: PageIndex) -> dict[str, int]` (doc_id to page count); CLI `python -m backend.ingest.cli [directory]`.

- [ ] **Step 1: Write the failing test**

`tests/test_ingest_cli.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.ingest.cli'`

- [ ] **Step 3: Implement**

`backend/ingest/cli.py`:
```python
from __future__ import annotations

import argparse
from pathlib import Path

from backend import config
from backend.index.store import PageIndex
from backend.ingest.pdf import ingest_pdf
from backend.manifest import ManifestError, load_manifest


def run_ingest(directory: Path, index: PageIndex) -> dict[str, int]:
    _, docs = load_manifest(directory)
    counts: dict[str, int] = {}
    for meta in docs:
        pages = ingest_pdf(Path(directory) / meta.file, meta)
        index.add_pages(pages)
        counts[meta.doc_id] = len(pages)
        flagged = sum(1 for p in pages if p.low_confidence)
        note = f" ({flagged} low-confidence)" if flagged else ""
        print(f"  {meta.doc_id}: {len(pages)} pages{note}")
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse filings and build the page index.")
    parser.add_argument("directory", nargs="?", default=str(config.DATA_DIR))
    args = parser.parse_args(argv)

    from backend.index.embedder import LocalEmbedder

    try:
        index = PageIndex(config.DB_PATH, LocalEmbedder())
        print(f"Ingesting {args.directory}")
        counts = run_ingest(Path(args.directory), index)
    except ManifestError as exc:
        print(f"Error: {exc}")
        return 1
    print(f"Done: {sum(counts.values())} pages indexed into {config.DB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingest_cli.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add backend/ingest/cli.py tests/test_ingest_cli.py
git commit -m "feat: ingest CLI that builds the index from a manifest" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 7: API and wiring

**Files:**
- Create: `backend/wiring.py`, `backend/api/__init__.py`, `backend/api/main.py`, `backend/api/server.py`, `tests/test_api.py`

**Interfaces:**
- Consumes: `FilingsAgent` (`ask`), `AgentError`, `verify`, `PageIndex` (`get_page`, `page_counts`), `DocMeta`, `load_manifest`.
- Produces: `create_app(index, agent, docs: list[DocMeta], company: str, pdf_dir: Path) -> FastAPI` with endpoints `GET /health`, `GET /documents`, `POST /ask`, `GET /page/{doc_id}/{page_no}`, `GET /pdf/{doc_id}`; `backend.wiring.build_components() -> tuple[str, list[DocMeta], PageIndex, FilingsAgent]`; `backend.api.server.app`.
- `/ask` request: `{"question": str, "period": str | null}`. Response: `{"summary", "not_found", "mostly_unverified", "disclaimer", "claims": [{"text","doc_id","page_no","quote","status"}]}`.

- [ ] **Step 1: Write the failing tests**

`tests/test_api.py`:
```python
import pytest
from fastapi.testclient import TestClient

from backend.agent.agent import AgentError
from backend.api.main import create_app
from backend.models import Answer, Claim, DocMeta
from tests.conftest import MARGIN_PAGE

DOCS = [DocMeta("q2-results", "q2-results.pdf", "results", "Q2 FY26", "Q2 FY26 Results")]


class FakeAgent:
    def __init__(self, answer=None, error=None):
        self.answer, self.error, self.calls = answer, error, []

    def ask(self, question, period=None):
        self.calls.append((question, period))
        if self.error:
            raise self.error
        return self.answer


@pytest.fixture()
def pdf_dir(tmp_path):
    (tmp_path / "q2-results.pdf").write_bytes(b"%PDF-1.4 fake")
    return tmp_path


def client_for(index, pdf_dir, agent):
    return TestClient(create_app(index, agent, DOCS, "Infosys", pdf_dir))


def good_answer():
    return Answer(
        summary="Margin was 21.1%.",
        claims=[Claim("Margin was 21.1%.", "q2-results", 1, MARGIN_PAGE)],
    )


def test_health(index, pdf_dir):
    assert client_for(index, pdf_dir, FakeAgent()).get("/health").json() == {"status": "ok"}


def test_documents_lists_company_docs_with_page_counts(index, pdf_dir):
    body = client_for(index, pdf_dir, FakeAgent()).get("/documents").json()
    assert body["company"] == "Infosys"
    assert body["documents"] == [
        {"doc_id": "q2-results", "title": "Q2 FY26 Results", "doc_type": "results", "period": "Q2 FY26", "pages": 2}
    ]


def test_ask_returns_verified_claims_and_disclaimer(index, pdf_dir):
    agent = FakeAgent(answer=good_answer())
    body = client_for(index, pdf_dir, agent).post("/ask", json={"question": "Margin?", "period": "Q2 FY26"}).json()
    assert agent.calls == [("Margin?", "Q2 FY26")]
    assert body["claims"][0]["status"] == "verified"
    assert body["claims"][0]["page_no"] == 1
    assert body["mostly_unverified"] is False
    assert "investment advice" in body["disclaimer"]


def test_fabricated_quote_is_returned_marked_failed_never_verified(index, pdf_dir):
    answer = Answer(summary="s", claims=[Claim("x", "q2-results", 1, "The company announced a giant buyback programme")])
    body = client_for(index, pdf_dir, FakeAgent(answer=answer)).post("/ask", json={"question": "q"}).json()
    assert body["claims"][0]["status"] == "failed"
    assert body["mostly_unverified"] is True


def test_not_found_answer(index, pdf_dir):
    answer = Answer(summary="This is not in the loaded filings.", claims=[], not_found=True)
    body = client_for(index, pdf_dir, FakeAgent(answer=answer)).post("/ask", json={"question": "price?"}).json()
    assert body["not_found"] is True and body["claims"] == []


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_empty_question_is_rejected_without_calling_the_model(index, pdf_dir, question):
    agent = FakeAgent(answer=good_answer())
    response = client_for(index, pdf_dir, agent).post("/ask", json={"question": question})
    assert response.status_code == 422
    assert agent.calls == []


def test_overlong_question_is_rejected(index, pdf_dir):
    agent = FakeAgent(answer=good_answer())
    response = client_for(index, pdf_dir, agent).post("/ask", json={"question": "x" * 1001})
    assert response.status_code == 422 and agent.calls == []


def test_agent_failure_becomes_502(index, pdf_dir):
    agent = FakeAgent(error=AgentError("Claude API call failed: boom"))
    response = client_for(index, pdf_dir, agent).post("/ask", json={"question": "q"})
    assert response.status_code == 502
    assert "boom" in response.json()["detail"]


def test_page_endpoint(index, pdf_dir):
    c = client_for(index, pdf_dir, FakeAgent())
    assert "Operating margin" in c.get("/page/q2-results/1").json()["text"]
    assert c.get("/page/q2-results/99").status_code == 404


def test_pdf_endpoint_serves_the_file_and_404s_unknown_docs(index, pdf_dir):
    c = client_for(index, pdf_dir, FakeAgent())
    response = c.get("/pdf/q2-results")
    assert response.status_code == 200 and response.headers["content-type"] == "application/pdf"
    assert c.get("/pdf/ghost").status_code == 404
    assert c.get("/pdf/..%2Fmanifest").status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.api'`

- [ ] **Step 3: Implement the API**

`backend/api/__init__.py`: empty file.

`backend/api/main.py`:
```python
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.agent.agent import AgentError
from backend.models import DocMeta
from backend.verify.verifier import verify

DISCLAIMER = (
    "For research and education only. Nothing here is investment advice or a recommendation."
)


class AskRequest(BaseModel):
    question: str = Field(max_length=1000)
    period: Optional[str] = None


def create_app(index, agent, docs: list[DocMeta], company: str, pdf_dir: Path) -> FastAPI:
    app = FastAPI(title="Filings & Concall Analyst")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    by_id = {d.doc_id: d for d in docs}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/documents")
    def documents():
        counts = index.page_counts()
        return {
            "company": company,
            "documents": [
                {
                    "doc_id": d.doc_id,
                    "title": d.title,
                    "doc_type": d.doc_type,
                    "period": d.period,
                    "pages": counts.get(d.doc_id, 0),
                }
                for d in docs
            ],
        }

    @app.post("/ask")
    def ask(request: AskRequest):
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="Question must not be empty.")
        try:
            answer = agent.ask(question, period=request.period)
        except AgentError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        verified = verify(answer, index)
        return {
            "summary": verified.summary,
            "not_found": verified.not_found,
            "mostly_unverified": verified.mostly_unverified,
            "disclaimer": DISCLAIMER,
            "claims": [
                {
                    "text": v.claim.text,
                    "doc_id": v.claim.doc_id,
                    "page_no": v.claim.page_no,
                    "quote": v.claim.quote,
                    "status": v.status,
                }
                for v in verified.claims
            ],
        }

    @app.get("/page/{doc_id}/{page_no}")
    def page(doc_id: str, page_no: int):
        found = index.get_page(doc_id, page_no)
        if found is None:
            raise HTTPException(status_code=404, detail="No such page.")
        return {
            "doc_id": found.doc_id,
            "page_no": found.page_no,
            "text": found.text,
            "low_confidence": found.low_confidence,
        }

    @app.get("/pdf/{doc_id}")
    def pdf(doc_id: str):
        meta = by_id.get(doc_id)
        if meta is None:
            raise HTTPException(status_code=404, detail="No such document.")
        return FileResponse(Path(pdf_dir) / meta.file, media_type="application/pdf")

    return app
```

- [ ] **Step 4: Implement wiring and server entrypoint**

`backend/wiring.py`:
```python
from __future__ import annotations

from backend import config
from backend.agent.agent import FilingsAgent
from backend.index.embedder import LocalEmbedder
from backend.index.store import PageIndex
from backend.manifest import load_manifest


def build_components():
    """Builds the real index and agent. Needs ANTHROPIC_API_KEY and an ingested index."""
    import anthropic

    company, docs = load_manifest(config.DATA_DIR)
    index = PageIndex(config.DB_PATH, LocalEmbedder())
    if not index.page_counts():
        raise RuntimeError("The index is empty. Run: python -m backend.ingest.cli")
    agent = FilingsAgent(anthropic.Anthropic(), index, model=config.MODEL, company=company)
    return company, docs, index, agent
```

`backend/api/server.py`:
```python
from backend import config
from backend.api.main import create_app
from backend.wiring import build_components

_company, _docs, _index, _agent = build_components()
app = create_app(_index, _agent, _docs, _company, config.DATA_DIR)
```

- [ ] **Step 5: Run the full suite**

Run: `pytest -v`
Expected: all tests pass (manifest 7, ingest 3, index 8, verifier 14, agent 12, ingest CLI 1, API 13 with parametrized cases)

- [ ] **Step 6: Commit and push**

```bash
git add backend/wiring.py backend/api tests/test_api.py
git commit -m "feat: FastAPI endpoints with server-side claim verification" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
git push origin main
```

---

### Task 8: Golden-set eval harness

**Files:**
- Create: `backend/eval/__init__.py`, `backend/eval/run_eval.py`, `evals/golden.json`, `tests/test_eval.py`

**Interfaces:**
- Consumes: `VerifiedAnswer`; `backend.wiring.build_components`; `verify`.
- Produces: `evaluate(cases: list[dict], ask_fn) -> EvalResult` where `ask_fn(question: str) -> VerifiedAnswer`; `EvalResult(total, correct, citations_clean, failures: list[str])` with `correct_rate`; CLI `python -m backend.eval.run_eval [evals/golden.json]` exits 1 when `correct_rate < 0.9`.
- Golden case format: `{"question": str, "expect_contains": [str, ...], "expect_not_found": bool}`. A factual case passes when the answer is not "not found" and every `expect_contains` string appears (case-insensitive) in the summary plus claim texts plus quotes. `citations_clean` counts cases where no displayed claim has status `failed`.

- [ ] **Step 1: Write the failing tests**

`tests/test_eval.py`:
```python
from backend.eval.run_eval import evaluate
from backend.models import Claim, VerifiedAnswer, VerifiedClaim


def answer(summary, statuses=(), quote="Operating margin was 21.1% in the quarter", not_found=False):
    claims = [VerifiedClaim(Claim("Margin was 21.1%", "d", 1, quote), s) for s in statuses]
    return VerifiedAnswer(summary, claims, not_found)


def test_factual_case_passes_when_expected_strings_present():
    cases = [{"question": "margin?", "expect_contains": ["21.1%"], "expect_not_found": False}]
    result = evaluate(cases, lambda q: answer("Margin was 21.1%.", ["verified"]))
    assert result.total == 1 and result.correct == 1 and result.citations_clean == 1


def test_factual_case_fails_when_expected_string_missing_or_not_found():
    cases = [{"question": "margin?", "expect_contains": ["21.1%"], "expect_not_found": False}]
    assert evaluate(cases, lambda q: answer("Margin was 25%.", ["verified"], quote="Operating margin was 25% in the quarter")).correct == 0
    assert evaluate(cases, lambda q: answer("nope", [], not_found=True)).correct == 0


def test_expected_string_match_is_case_insensitive_and_searches_quotes():
    cases = [{"question": "q", "expect_contains": ["WAGE HIKES"], "expect_not_found": False}]
    result = evaluate(cases, lambda q: answer("Pressure on margins.", ["verified"], quote="Management cited wage hikes as the main driver"))
    assert result.correct == 1


def test_adversarial_case_requires_not_found():
    cases = [{"question": "share price?", "expect_contains": [], "expect_not_found": True}]
    assert evaluate(cases, lambda q: answer("not in filings", [], not_found=True)).correct == 1
    assert evaluate(cases, lambda q: answer("It is 1500.", ["verified"])).correct == 0


def test_failed_claims_reduce_citations_clean_not_correctness():
    cases = [{"question": "margin?", "expect_contains": ["21.1%"], "expect_not_found": False}]
    result = evaluate(cases, lambda q: answer("Margin was 21.1%.", ["verified", "failed"]))
    assert result.correct == 1 and result.citations_clean == 0


def test_failures_list_names_the_question():
    cases = [{"question": "What was margin?", "expect_contains": ["99%"], "expect_not_found": False}]
    result = evaluate(cases, lambda q: answer("x", ["verified"]))
    assert result.failures == ["What was margin?"]
    assert result.correct_rate == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.eval'`

- [ ] **Step 3: Implement**

`backend/eval/__init__.py`: empty file.

`backend/eval/run_eval.py`:
```python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from backend.models import VerifiedAnswer

PASS_RATE = 0.9


@dataclass
class EvalResult:
    total: int = 0
    correct: int = 0
    citations_clean: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def correct_rate(self) -> float:
        return self.correct / self.total if self.total else 0.0


def _haystack(answer: VerifiedAnswer) -> str:
    parts = [answer.summary]
    for verified in answer.claims:
        parts += [verified.claim.text, verified.claim.quote]
    return "\n".join(parts).casefold()


def _is_correct(case: dict, answer: VerifiedAnswer) -> bool:
    if case.get("expect_not_found"):
        return answer.not_found
    if answer.not_found:
        return False
    haystack = _haystack(answer)
    return all(expected.casefold() in haystack for expected in case.get("expect_contains", []))


def evaluate(cases: list[dict], ask_fn) -> EvalResult:
    result = EvalResult(total=len(cases))
    for case in cases:
        answer = ask_fn(case["question"])
        if _is_correct(case, answer):
            result.correct += 1
        else:
            result.failures.append(case["question"])
        if all(v.status != "failed" for v in answer.claims):
            result.citations_clean += 1
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the golden question set against the real agent.")
    parser.add_argument("golden", nargs="?", default="evals/golden.json")
    args = parser.parse_args(argv)

    from backend.verify.verifier import verify
    from backend.wiring import build_components

    _, _, index, agent = build_components()
    cases = json.loads(Path(args.golden).read_text())
    result = evaluate(cases, lambda q: verify(agent.ask(q), index))

    print(f"Correct:          {result.correct}/{result.total} ({result.correct_rate:.0%})")
    print(f"Citations clean:  {result.citations_clean}/{result.total}")
    for question in result.failures:
        print(f"  FAILED: {question}")
    return 0 if result.correct_rate >= PASS_RATE else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

`evals/golden.json` (adversarial cases need no filing knowledge; factual cases are added in Task 10 once the real PDFs are loaded):
```json
[
  {"question": "What is Infosys's share price today?", "expect_contains": [], "expect_not_found": true},
  {"question": "Should I buy Infosys stock now?", "expect_contains": [], "expect_not_found": true},
  {"question": "What was TCS's operating margin last quarter?", "expect_contains": [], "expect_not_found": true}
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval.py -v && pytest -q`
Expected: 6 passed, then the whole suite green

- [ ] **Step 5: Commit**

```bash
git add backend/eval evals tests/test_eval.py
git commit -m "feat: golden-set eval harness with adversarial not-found cases" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Frontend (three-pane workspace)

**Files:**
- Create: `frontend/` (Vite scaffold), `frontend/src/types.ts`, `frontend/src/api.ts`, `frontend/src/lib/citations.ts`, `frontend/src/lib/highlight.ts`, `frontend/src/lib/citations.test.ts`, `frontend/src/lib/highlight.test.ts`, `frontend/src/components/TopBar.tsx`, `frontend/src/components/BriefingPane.tsx`, `frontend/src/components/ChatPane.tsx`, `frontend/src/components/SourceViewer.tsx`
- Modify: `frontend/src/App.tsx`, `frontend/src/index.css`, `frontend/vite.config.ts`, `frontend/package.json`

**Interfaces:**
- Consumes: backend endpoints from Task 7 through the Vite proxy (`/api/*` maps to `http://localhost:8000/*`).
- Produces: `numberClaims`, `statusLabel` in `lib/citations.ts`; `normalize`, `shouldHighlight`, `escapeHtml` in `lib/highlight.ts`.

- [ ] **Step 1: Scaffold and install**

```bash
cd /Users/TonyStark/Desktop/web/filings-analyst
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
npm install react-pdf tailwindcss @tailwindcss/vite
npm install -D vitest
```

Add `"test": "vitest"` to the `scripts` in `frontend/package.json`.

`frontend/vite.config.ts`:
```ts
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
```

Replace `frontend/src/index.css` with:
```css
@import "tailwindcss";

:root {
  --accent: #0f766e;
  --ink: #1c1917;
  --muted: #78716c;
  --line: #e7e5e4;
  --paper: #fafaf9;
}

body {
  margin: 0;
  color: var(--ink);
  background: var(--paper);
  font-family: "Inter", system-ui, sans-serif;
  font-variant-numeric: tabular-nums;
}

.react-pdf__Page__textContent mark {
  background: rgba(250, 204, 21, 0.45);
  color: transparent;
}
```

- [ ] **Step 2: Write the failing library tests**

`frontend/src/lib/citations.test.ts`:
```ts
import { describe, expect, it } from 'vitest'
import { numberClaims, statusLabel } from './citations'
import type { Claim } from '../types'

const claim = (text: string, status: Claim['status']): Claim => ({
  text, doc_id: 'd', page_no: 1, quote: 'q', status,
})

describe('numberClaims', () => {
  it('numbers claims from 1 in order', () => {
    const numbered = numberClaims([claim('a', 'verified'), claim('b', 'failed')])
    expect(numbered.map((n) => n.n)).toEqual([1, 2])
    expect(numbered[1].claim.text).toBe('b')
  })
  it('handles no claims', () => {
    expect(numberClaims([])).toEqual([])
  })
})

describe('statusLabel', () => {
  it('never presents a failed claim as verified', () => {
    expect(statusLabel('verified')).toBe('Verified in source')
    expect(statusLabel('weak')).toBe('Close match, check the source')
    expect(statusLabel('failed')).toBe('Unverified')
  })
})
```

`frontend/src/lib/highlight.test.ts`:
```ts
import { describe, expect, it } from 'vitest'
import { escapeHtml, normalize, shouldHighlight } from './highlight'

describe('normalize', () => {
  it('lowercases, unifies curly quotes and collapses whitespace', () => {
    expect(normalize('  Clients’   BUDGETS \n tight ')).toBe("clients' budgets tight")
  })
})

describe('shouldHighlight', () => {
  const quote = 'Operating margin was 21.1% in the second quarter'
  it('highlights text items that are part of the quote', () => {
    expect(shouldHighlight('Operating margin was', quote)).toBe(true)
    expect(shouldHighlight('21.1%', quote)).toBe(true)
  })
  it('ignores items that are not in the quote', () => {
    expect(shouldHighlight('Revenue grew', quote)).toBe(false)
  })
  it('ignores tiny fragments that would highlight everywhere', () => {
    expect(shouldHighlight('a', quote)).toBe(false)
    expect(shouldHighlight(' ', quote)).toBe(false)
  })
})

describe('escapeHtml', () => {
  it('escapes markup so PDF text cannot inject HTML', () => {
    expect(escapeHtml('<b>"A&B"</b>')).toBe('&lt;b&gt;&quot;A&amp;B&quot;&lt;/b&gt;')
  })
})
```

- [ ] **Step 3: Run tests to verify they fail**

Run (in `frontend/`): `npx vitest --run`
Expected: FAIL, cannot resolve `./citations` and `./highlight`

- [ ] **Step 4: Implement types and libs**

`frontend/src/types.ts`:
```ts
export type ClaimStatus = 'verified' | 'weak' | 'failed'

export interface Claim {
  text: string
  doc_id: string
  page_no: number
  quote: string
  status: ClaimStatus
}

export interface AskResponse {
  summary: string
  not_found: boolean
  mostly_unverified: boolean
  disclaimer: string
  claims: Claim[]
}

export interface DocumentInfo {
  doc_id: string
  title: string
  doc_type: string
  period: string
  pages: number
}

export interface DocumentsResponse {
  company: string
  documents: DocumentInfo[]
}
```

`frontend/src/lib/citations.ts`:
```ts
import type { Claim, ClaimStatus } from '../types'

export interface NumberedClaim {
  n: number
  claim: Claim
}

export function numberClaims(claims: Claim[]): NumberedClaim[] {
  return claims.map((claim, i) => ({ n: i + 1, claim }))
}

export function statusLabel(status: ClaimStatus): string {
  switch (status) {
    case 'verified':
      return 'Verified in source'
    case 'weak':
      return 'Close match, check the source'
    case 'failed':
      return 'Unverified'
  }
}
```

`frontend/src/lib/highlight.ts`:
```ts
export function normalize(text: string): string {
  return text
    .toLowerCase()
    .replace(/[‘’]/g, "'")
    .replace(/[“”]/g, '"')
    .replace(/\s+/g, ' ')
    .trim()
}

// PDF text layers split a line into many small items, so an item is highlighted when it
// is a substring of the quote. Items under 4 characters are skipped to avoid noise.
export function shouldHighlight(item: string, quote: string): boolean {
  const text = normalize(item)
  return text.length >= 4 && normalize(quote).includes(text)
}

export function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run (in `frontend/`): `npx vitest --run`
Expected: 8 passed

- [ ] **Step 6: Implement the API client and components**

`frontend/src/api.ts`:
```ts
import type { AskResponse, DocumentsResponse } from './types'

export async function fetchDocuments(): Promise<DocumentsResponse> {
  const response = await fetch('/api/documents')
  if (!response.ok) throw new Error('Could not load documents. Is the backend running?')
  return response.json()
}

export async function ask(question: string, period: string | null): Promise<AskResponse> {
  const response = await fetch('/api/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, period }),
  })
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail ?? `Request failed (${response.status})`)
  }
  return response.json()
}

export const pdfUrl = (docId: string) => `/api/pdf/${encodeURIComponent(docId)}`
```

`frontend/src/components/TopBar.tsx`:
```tsx
interface Props {
  company: string
  periods: string[]
  period: string | null
  onPeriod: (period: string | null) => void
}

export function TopBar({ company, periods, period, onPeriod }: Props) {
  return (
    <header className="flex items-center gap-4 border-b border-[var(--line)] px-5 py-3">
      <h1 className="text-base font-semibold tracking-tight">{company}</h1>
      <label className="flex items-center gap-2 text-sm text-[var(--muted)]">
        Period
        <select
          className="rounded border border-[var(--line)] bg-white px-2 py-1 text-[var(--ink)]"
          value={period ?? ''}
          onChange={(e) => onPeriod(e.target.value || null)}
        >
          <option value="">All periods</option>
          {periods.map((p) => (
            <option key={p} value={p}>{p}</option>
          ))}
        </select>
      </label>
    </header>
  )
}
```

`frontend/src/components/BriefingPane.tsx`:
```tsx
import type { DocumentInfo } from '../types'

export function BriefingPane({ documents }: { documents: DocumentInfo[] }) {
  return (
    <aside className="flex flex-col gap-6 overflow-y-auto border-r border-[var(--line)] p-5 text-sm">
      <section>
        <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-[var(--muted)]">Loaded filings</h2>
        <ul className="flex flex-col gap-2">
          {documents.map((d) => (
            <li key={d.doc_id}>
              <div className="font-medium">{d.title}</div>
              <div className="text-[var(--muted)]">{d.period} · {d.pages} pages</div>
            </li>
          ))}
        </ul>
      </section>
      <section className="text-[var(--muted)]">
        <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide">Briefing</h2>
        <p>What changed, red flags and management tone will appear here in a later version.</p>
      </section>
    </aside>
  )
}
```

`frontend/src/components/ChatPane.tsx`:
```tsx
import { useState } from 'react'
import { ask } from '../api'
import { numberClaims, statusLabel } from '../lib/citations'
import type { AskResponse, Claim } from '../types'

interface Turn {
  question: string
  answer?: AskResponse
  error?: string
}

const SUGGESTIONS = [
  'What drove the change in operating margin this quarter?',
  'What did management say about the demand outlook?',
  'What is the guidance for the full year?',
]

interface Props {
  period: string | null
  onCite: (claim: Claim) => void
}

export function ChatPane({ period, onCite }: Props) {
  const [turns, setTurns] = useState<Turn[]>([])
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)

  async function submit(question: string) {
    const trimmed = question.trim()
    if (!trimmed || busy) return
    setDraft('')
    setBusy(true)
    setTurns((t) => [...t, { question: trimmed }])
    try {
      const answer = await ask(trimmed, period)
      setTurns((t) => t.map((turn, i) => (i === t.length - 1 ? { ...turn, answer } : turn)))
      const first = answer.claims[0]
      if (first) onCite(first)
    } catch (err) {
      const error = err instanceof Error ? err.message : 'Something went wrong.'
      setTurns((t) => t.map((turn, i) => (i === t.length - 1 ? { ...turn, error } : turn)))
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="flex min-h-0 flex-col border-r border-[var(--line)]">
      <div className="flex-1 space-y-6 overflow-y-auto p-5">
        {turns.length === 0 && (
          <div className="space-y-2 text-sm">
            <p className="text-[var(--muted)]">Ask about the loaded filings. Try:</p>
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                className="block text-left text-[var(--accent)] hover:underline"
                onClick={() => submit(s)}
              >
                {s}
              </button>
            ))}
          </div>
        )}
        {turns.map((turn, i) => (
          <div key={i} className="space-y-2">
            <p className="font-medium">{turn.question}</p>
            {!turn.answer && !turn.error && <p className="text-sm text-[var(--muted)]">Reading the filings…</p>}
            {turn.error && <p className="text-sm text-red-700">{turn.error}</p>}
            {turn.answer && <AnswerView answer={turn.answer} onCite={onCite} />}
          </div>
        ))}
      </div>
      <form
        className="border-t border-[var(--line)] p-4"
        onSubmit={(e) => {
          e.preventDefault()
          submit(draft)
        }}
      >
        <input
          className="w-full rounded border border-[var(--line)] bg-white px-3 py-2 text-sm outline-none focus:border-[var(--accent)]"
          placeholder="Ask about this filing…"
          value={draft}
          maxLength={1000}
          onChange={(e) => setDraft(e.target.value)}
          disabled={busy}
        />
        <p className="mt-2 text-xs text-[var(--muted)]">
          For research and education only. Not investment advice.
        </p>
      </form>
    </main>
  )
}

function AnswerView({ answer, onCite }: { answer: AskResponse; onCite: (claim: Claim) => void }) {
  if (answer.not_found) {
    return <p className="text-sm text-[var(--muted)]">{answer.summary}</p>
  }
  return (
    <div className="space-y-3 text-sm">
      {answer.mostly_unverified && (
        <p className="rounded border border-amber-300 bg-amber-50 px-3 py-2 text-amber-900">
          Most citations below could not be verified against the source. Treat this answer with caution.
        </p>
      )}
      <p>{answer.summary}</p>
      <ol className="space-y-2">
        {numberClaims(answer.claims).map(({ n, claim }) => (
          <li key={n} className={claim.status === 'failed' ? 'opacity-60' : ''}>
            <button
              className="mr-2 rounded bg-[var(--accent)] px-1.5 text-xs text-white"
              onClick={() => onCite(claim)}
              aria-label={`Open source for claim ${n}`}
            >
              {n}
            </button>
            {claim.text}
            <span className="ml-2 text-xs text-[var(--muted)]">{statusLabel(claim.status)}</span>
          </li>
        ))}
      </ol>
    </div>
  )
}
```

`frontend/src/components/SourceViewer.tsx`:
```tsx
import { useState } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import 'react-pdf/dist/Page/TextLayer.css'
import { pdfUrl } from '../api'
import { escapeHtml, shouldHighlight } from '../lib/highlight'
import { statusLabel } from '../lib/citations'
import type { Claim } from '../types'

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url,
).toString()

export function SourceViewer({ claim }: { claim: Claim | null }) {
  const [loadError, setLoadError] = useState(false)

  if (!claim) {
    return (
      <aside className="p-5 text-sm text-[var(--muted)]">
        Click a citation number to see the source page here, with the quoted passage highlighted.
      </aside>
    )
  }
  return (
    <aside className="flex min-h-0 flex-col">
      <div className="border-b border-[var(--line)] p-4 text-sm">
        <div className="mb-1 text-xs uppercase tracking-wide text-[var(--muted)]">
          {claim.doc_id} · page {claim.page_no} · {statusLabel(claim.status)}
        </div>
        <blockquote className="border-l-2 border-[var(--accent)] pl-3">{claim.quote}</blockquote>
      </div>
      <div className="min-h-0 flex-1 overflow-auto p-4">
        {loadError ? (
          <p className="text-sm text-red-700">Could not load the PDF. The quote above is the cited passage.</p>
        ) : (
          <Document
            key={claim.doc_id}
            file={pdfUrl(claim.doc_id)}
            onLoadError={() => setLoadError(true)}
            onLoadSuccess={() => setLoadError(false)}
          >
            <Page
              pageNumber={claim.page_no}
              width={520}
              customTextRenderer={({ str }) =>
                shouldHighlight(str, claim.quote) ? `<mark>${escapeHtml(str)}</mark>` : escapeHtml(str)
              }
            />
          </Document>
        )}
      </div>
    </aside>
  )
}
```

`frontend/src/App.tsx`:
```tsx
import { useEffect, useState } from 'react'
import { fetchDocuments } from './api'
import { BriefingPane } from './components/BriefingPane'
import { ChatPane } from './components/ChatPane'
import { SourceViewer } from './components/SourceViewer'
import { TopBar } from './components/TopBar'
import type { Claim, DocumentsResponse } from './types'

export default function App() {
  const [docs, setDocs] = useState<DocumentsResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [period, setPeriod] = useState<string | null>(null)
  const [active, setActive] = useState<Claim | null>(null)

  useEffect(() => {
    fetchDocuments().then(setDocs).catch((e: Error) => setError(e.message))
  }, [])

  if (error) return <p className="p-6 text-red-700">{error}</p>
  if (!docs) return <p className="p-6 text-[var(--muted)]">Loading…</p>

  const periods = [...new Set(docs.documents.map((d) => d.period))]
  return (
    <div className="flex h-screen flex-col">
      <TopBar company={docs.company} periods={periods} period={period} onPeriod={setPeriod} />
      <div className="grid min-h-0 flex-1 grid-cols-[260px_minmax(0,1fr)_minmax(0,1fr)]">
        <BriefingPane documents={docs.documents} />
        <ChatPane period={period} onCite={setActive} />
        <SourceViewer claim={active} />
      </div>
    </div>
  )
}
```

Delete the unused Vite starter files (`frontend/src/App.css`, `frontend/src/assets/react.svg`) and remove their imports from `frontend/src/main.tsx` if present.

- [ ] **Step 7: Verify the build and unit tests**

Run (in `frontend/`): `npx vitest --run && npm run build`
Expected: 8 tests pass; `tsc` and `vite build` finish with no errors. Fix any type errors before continuing (for example react-pdf's `customTextRenderer` parameter type).

- [ ] **Step 8: Smoke test the whole flow by hand**

With a real ingested index (see Task 10 Step 1) in two terminals:
```bash
# terminal 1, from filings-analyst/
source .venv/bin/activate && export ANTHROPIC_API_KEY=...   # your key
uvicorn backend.api.server:app --port 8000
# terminal 2
cd frontend && npm run dev
```
Open http://localhost:5173, ask a question, click a citation number. Expected: the answer shows numbered claims with status labels; the right pane shows the cited page with the quote highlighted. Ask "What is the share price today?". Expected: "not in the loaded filings" message. If the PDF pane is blank, check the browser console for a pdf.js worker error.

- [ ] **Step 9: Commit and push**

```bash
cd /Users/TonyStark/Desktop/web/filings-analyst
git add frontend
git commit -m "feat: three-pane React workspace with cited answers and source viewer" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
git push origin main
```

---

### Task 10: Real filings, golden set and v1 sign-off

**Files:**
- Create: `data/raw/infosys/manifest.json` (git-ignored with the PDFs), extend `evals/golden.json`
- Modify: `README.md`

This task needs the user's PDFs and an `ANTHROPIC_API_KEY`. If the PDFs are not in `data/raw/infosys/` yet, stop and ask for them.

- [ ] **Step 1: Load the filings**

Ask the user to download from Infosys investor relations (or BSE/NSE) into `data/raw/infosys/`: the latest annual report, 2-4 quarterly results PDFs and the matching concall transcripts. Write `data/raw/infosys/manifest.json` listing each file, for example:
```json
{
  "company": "Infosys",
  "documents": [
    {"file": "ar-fy25.pdf", "doc_type": "annual_report", "period": "FY25", "title": "Annual Report FY25"},
    {"file": "q2-fy26-results.pdf", "doc_type": "results", "period": "Q2 FY26", "title": "Q2 FY26 Results"},
    {"file": "q2-fy26-concall.pdf", "doc_type": "concall", "period": "Q2 FY26", "title": "Q2 FY26 Concall"}
  ]
}
```
Use the actual filenames and periods. Then run:
```bash
source .venv/bin/activate
python -m backend.ingest.cli
```
Expected: one line per document with page counts, then `Done: N pages indexed`. Note any low-confidence pages; if a whole document is flagged, it is scanned and needs Tesseract (`brew install tesseract`) before re-ingesting.

- [ ] **Step 2: Author the factual golden questions**

Read the ingested pages (for example `python -c "from backend.config import DB_PATH; ..."` or open the PDFs) and add 15-17 factual cases to `evals/golden.json` alongside the 3 adversarial ones, for a total of about 20. Each factual case uses this format, with `expect_contains` holding exact figures or phrases that appear in the source:
```json
{"question": "What was the operating margin in Q2 FY26?", "expect_contains": ["21."], "expect_not_found": false}
```
Mix question types: a single figure from results, a qualitative management statement from a concall, a number from the annual report, a question that needs two pages, a question about a different period than the one selected. Every `expect_contains` value must be confirmed against the source by a human before it goes in.

- [ ] **Step 3: Run the eval against the real agent**

Run: `python -m backend.eval.run_eval`
Expected: `Correct: N/20` with at least 90%, `Citations clean` equal to the total, and exit code 0.

If correctness is below 90%, read each failure before changing anything. Decide whether the question, the expected string or the agent is wrong. Fix agent failures through the system prompt in `backend/agent/tools.py` or search behavior in `backend/index/store.py`, re-run `pytest -q`, then re-run the eval. Do not lower the threshold or loosen expected strings to pass.

- [ ] **Step 4: Update the README**

Replace the "Planned v1" section of `README.md` with setup and usage:
```markdown
## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
```

## Load filings
Put PDFs and a `manifest.json` in `data/raw/infosys/` (fields: `file`, `doc_type` one of `annual_report | results | concall`, `period`, `title`), then:
```bash
python -m backend.ingest.cli
```

## Run
```bash
uvicorn backend.api.server:app --port 8000
cd frontend && npm install && npm run dev   # http://localhost:5173
```

## Test
```bash
pytest
python -m backend.eval.run_eval    # golden-set quality check, needs the API key and ingested filings
```

## How trust works
Every claim carries a document, page and verbatim quote. A deterministic verifier outside the model checks the quote against the cited page. Claims that fail are shown as unverified, never as fact.
```
Also record the latest eval numbers (correct rate, citations clean) in a short "Results" line.

- [ ] **Step 5: Final verification**

Run: `pytest -q && cd frontend && npx vitest --run && npm run build`
Expected: everything passes. Confirm `git status` shows no PDFs, index database or `.env` files staged.

- [ ] **Step 6: Commit and push**

```bash
cd /Users/TonyStark/Desktop/web/filings-analyst
git add README.md evals/golden.json
git commit -m "docs: v1 setup guide and golden set with real eval results" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
git push origin main
```
