# Filings & Concall Analyst: v1 Design

## Goal
A research tool for Indian listed companies. The user asks questions about a company's own documents (annual report, quarterly results, concall transcripts) and gets answers where every claim is cited to a document, page and quote, and the quote is machine-verified against the page text.

v1 scope: **one company (Infosys as placeholder), cited Q&A with a source viewer.**

Out of scope for v1: auto-generated briefing and quarter-over-quarter diff, auto-fetching from BSE/NSE, multiple companies, authentication.

## Approach
Agentic retrieval over a page index. PDFs are parsed once into pages. Claude gets `search` and `read_page` tools and must return claims as `{text, doc, page, quote}`. A verifier outside the model checks each quote against the cited page.

Rejected: classic chunk RAG (fragments miss multi-page and table questions, citations point at chunks); long-context stuffing (cost and latency per question, does not scale past one company).

## Structure
```
filings-analyst/
├── data/raw/infosys/   user-supplied PDFs + manifest.json
├── backend/
│   ├── ingest/         PDF -> pages -> index
│   ├── index/          page store + hybrid search
│   ├── agent/          Claude tool loop
│   ├── verify/         quote checker
│   └── api/            FastAPI
└── frontend/           React + TS + Tailwind + Vite
```

| Unit | Responsibility | Interface | Depends on |
|---|---|---|---|
| ingest | Extract page text and tables (as markdown), OCR fallback for scanned pages, tag doc type and period from manifest | `ingest(pdf, meta) -> Page[]` | PDF library |
| index | SQLite page store, keyword + local embedding hybrid search | `search(query, filters) -> PageHit[]`, `get_page(doc, n)` | ingest output |
| agent | Claude loop with `search` and `read_page`, max ~8 tool calls, returns claims | `ask(question, company, period) -> Answer` | index, Claude API |
| verify | Normalized fuzzy match of each quote against cited page text | `verify(Answer) -> VerifiedAnswer` | index |
| api | `POST /ask`, `GET /page/{doc}/{n}`, `GET /pdf/{doc}` | HTTP | agent, verify, index |
| frontend | Three panes: briefing placeholder, chat with `[n]` citations, PDF viewer highlighting the quote | HTTP | api |

Decisions: SQLite and a local embedding model (about 500-1000 pages, no hosted vector DB). The verifier is deterministic code, never the model.

## Data flow
**Ingest (once):** PDFs in `data/raw/infosys/` + `manifest.json` (file, doc type, period) -> per-page text and markdown tables -> OCR for near-empty pages (flagged low confidence) -> stored and indexed.

**Ask:** `POST /ask` -> agent loop -> claims with quotes (claims lacking a quote are rejected and re-requested once) -> verifier -> API returns answer with per-claim status (verified / weak / failed) -> UI renders `[n]` markers -> clicking opens the PDF at the page with the quote highlighted.

## Error handling
- Quote fails verification: shown as "unverified" in a muted style, never as fact. If most claims fail, the answer says so.
- Nothing relevant found: answers "not in the loaded filings", never guesses.
- Scanned or garbled page: OCR fallback, flagged low-confidence.
- Claude API error or timeout: one retry, then a clear chat message.
- Out-of-scope question (prices, advice, other companies): declines and points to the loaded documents.
- UI carries an educational-use disclaimer; the tool gives no investment advice.

## Testing
- Unit: verifier (exact, fuzzy, wrong-page, fabricated quotes), ingest on fixture pages, search ranking.
- Golden set of about 20 questions on the real filings with known answers and pages. Measures answer correctness, citation validity and not-found behavior.
- Adversarial set: questions whose answers are not in the documents must return "not found".
- Frontend smoke test: ask, then click a citation.

## v1 success criteria
- At least 90% of golden-set answers correct.
- 100% of displayed claims pass verification or are visibly marked unverified.
- Zero fabricated quotes displayed as verified.

## UI
Dense, calm research-terminal look: one accent colour, tabular numbers, no chat bubbles or gradients. Top bar with company and period pickers. Left pane reserved for the later briefing, middle pane chat, right pane source viewer.

## Project hygiene
Lives in `web/filings-analyst/`, separate from the other web projects. Source PDFs stay out of git (`data/raw/` ignored); the manifest format is documented in the README. Progress is pushed to https://github.com/amateurcoder015/Filings-and-concall-analyst as work proceeds.
