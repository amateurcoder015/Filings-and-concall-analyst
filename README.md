# Filings & Concall Analyst

Ask questions about an Indian listed company's annual report, quarterly results and concall transcripts. Every claim in the answer is cited to a document, page and quote, and each quote is machine-checked against the stored page text. Infosys is the placeholder company; the company comes from `manifest.json`.

**Status:** v1 pipeline, API and UI are built and tested (375 backend / 20 frontend tests). Evaluation on real filings is pending: the agent needs a model API key (`ANTHROPIC_API_KEY`); no real-filings results exist yet.

Without a key you can still ingest PDFs, inspect the index, and run both test suites. `/ask` needs the key, and the server refuses to start without it.

Design: [docs/superpowers/specs/2026-10-01-filings-analyst-design.md](docs/superpowers/specs/2026-10-01-filings-analyst-design.md).

## How trust works

An agent searches and reads pages of the indexed filings, then returns a summary plus claims, each with `doc_id`, `page_no` and a verbatim `quote`. A deterministic verifier (no model involved) compares each quote with the page text and assigns a status:

| Status | Meaning |
|---|---|
| `verified` | The quote appears on the cited page (after normalising quotes, spaces, line breaks and hyphenation), and every figure in the claim text also appears in the quote. |
| `weak` | Near-miss, for example OCR noise or a cut word. Figures, signs, currency, units, direction and negation words are identical; only near-identical ordinary words differ. A best-effort labelled match, never counted as verified. |
| `unsupported` | The quote was found, but the claim text contains a figure that is not in the quote. |
| `failed` | Page or document does not exist, quote too short, or the quote does not match the page (including any changed figure or sign). |

The API response also carries `not_found` (nothing relevant in the loaded filings), `mostly_unverified` (more than half the claims are not verified) and `summary_supported` (every figure in the summary appears in the quote of a verified claim).

Accepted limits:
- Verification checks that the quote is on the page, not that the claim's reasoning from it is right.
- Table text is flattened, so attributing a number to the right column or row is not guaranteed.
- A quote that omits a nearby qualifier still verifies.
- Reformatted numbers (1200 vs 1,200) fail rather than match.
- Semantic flips between long ordinary words one edit apart can reach `weak`.

## Requirements

- Python 3.10+
- Node 18+
- Optional: Tesseract for scanned PDFs (`brew install tesseract`)
- The first ingest downloads the `BAAI/bge-small-en-v1.5` embedding model.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # needed to run the server and the eval, not for ingest or tests
cd frontend && npm install && cd ..
```

Optional environment variables: `FILINGS_DATA_DIR` (default `data/raw/infosys`), `FILINGS_DB` (default `data/index/index.db`), `FILINGS_MODEL` (default `claude-sonnet-5-5`). The Anthropic client uses a 60 second timeout and no automatic retries.

## Loading filings

Put the PDFs and a `manifest.json` in `data/raw/infosys` (the folder is git-ignored). `doc_type` is one of `annual_report`, `results`, `concall`. The document id is the PDF file name without `.pdf`.

```json
{
  "company": "Infosys",
  "documents": [
    {"file": "annual-report-fy25.pdf", "doc_type": "annual_report", "period": "FY25", "title": "Annual Report FY25"},
    {"file": "q2-fy26-results.pdf", "doc_type": "results", "period": "Q2 FY26", "title": "Q2 FY26 results"},
    {"file": "q2-fy26-concall.pdf", "doc_type": "concall", "period": "Q2 FY26", "title": "Q2 FY26 concall transcript"}
  ]
}
```

```bash
python -m backend.ingest.cli            # or pass a directory: python -m backend.ingest.cli path/to/dir
```

Re-ingesting is safe: each listed document is replaced, and documents no longer in the manifest are pruned from the index. Pages that needed OCR or look unreliable are reported as low-confidence.

## Running

```bash
uvicorn backend.api.server:app --host 127.0.0.1 --port 8000
cd frontend && npm run dev          # http://localhost:5173
```

The dev server proxies `/api/*` to `http://127.0.0.1:8000`; override with `VITE_API_TARGET=http://127.0.0.1:8001 npm run dev`. Startup fails with a clear message if the key is unset, the index is empty, or the index holds documents missing from the manifest.

Endpoints: `GET /health`, `GET /documents`, `POST /ask` (`question` up to 1000 chars, optional `period` that must match a period in the manifest, otherwise 422), `GET /page/{doc_id}/{page_no}`, `GET /pdf/{doc_id}`. Agent failures return 502; other errors return a generic 500 and are logged.

**Security:** the API has no authentication and spends your API key. Bind to `127.0.0.1`, never `0.0.0.0`.

## Testing

```bash
pytest -q                           # backend
cd frontend && npx vitest --run     # frontend
python -m backend.eval.run_eval     # golden questions against the real agent; needs the key and an ingested index
```

`evals/golden.json` is a list of cases:

```json
{"question": "...", "expect_contains": ["text expected in a verified claim"], "expect_pages": [{"doc_id": "annual-report-fy25", "page_no": 42}]}
{"question": "What is the share price today?", "expect_not_found": true}
```

A factual case needs `expect_contains` (case-insensitive substrings). Scoring counts only verified claims: the expected text must appear in a verified claim's text or quote, and every `expect_pages` entry must be cited by a verified claim. `expect_not_found` cases pass only if the answer is "not found". The run prints correct rate, clean-citation count and weak/unsupported/failed totals, and exits non-zero below 90% correct. The shipped file holds three not-found cases only; factual cases need real filings.

For research and education only. Nothing here is investment advice.
