from __future__ import annotations

import re
import sqlite3
import threading
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
CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(text, tokenize = 'porter unicode61');
"""


class PageIndex:
    def __init__(self, db_path, embedder):
        self._embedder = embedder
        self._lock = threading.RLock()
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # FastAPI serves requests from a thread pool, so the connection is shared across threads.
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.executescript(SCHEMA)

    def add_pages(self, pages: list[Page]) -> None:
        if not pages:
            return
        # Compute embeddings outside the lock (this is slow and doesn't touch the DB)
        vectors = self._embedder.embed([p.text for p in pages])
        # Hold lock only around DB writes
        with self._lock:
            with self._db:
                for page, vector in zip(pages, vectors):
                    row = self._db.execute(
                        "SELECT id FROM pages WHERE doc_id = ? AND page_no = ?",
                        (page.doc_id, page.page_no),
                    ).fetchone()
                    if row:
                        self._db.execute("DELETE FROM pages_fts WHERE rowid = ?", (row[0],))
                        self._db.execute("DELETE FROM pages WHERE id = ?", (row[0],))
                    self._insert(page, vector)

    def replace_document(self, doc_id: str, pages: list[Page]) -> None:
        """Replace every page of `doc_id` with `pages`, so no page of an older version survives."""
        if any(page.doc_id != doc_id for page in pages):
            raise ValueError(f"replace_document({doc_id!r}) got pages of another document")
        vectors = self._embedder.embed([p.text for p in pages]) if pages else []
        with self._lock:
            with self._db:
                self._delete_doc(doc_id)
                for page, vector in zip(pages, vectors):
                    self._insert(page, vector)

    def prune_documents(self, keep_doc_ids: set[str]) -> None:
        """Delete every document whose doc_id is not in `keep_doc_ids`."""
        with self._lock:
            with self._db:
                stored = [r[0] for r in self._db.execute("SELECT DISTINCT doc_id FROM pages").fetchall()]
                for doc_id in stored:
                    if doc_id not in keep_doc_ids:
                        self._delete_doc(doc_id)

    def doc_ids(self) -> set[str]:
        with self._lock:
            return {r[0] for r in self._db.execute("SELECT DISTINCT doc_id FROM pages").fetchall()}

    def fts_row_count(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM pages_fts").fetchone()[0]

    def _delete_doc(self, doc_id: str) -> None:
        # Caller holds the lock and the transaction.
        self._db.execute(
            "DELETE FROM pages_fts WHERE rowid IN (SELECT id FROM pages WHERE doc_id = ?)", (doc_id,)
        )
        self._db.execute("DELETE FROM pages WHERE doc_id = ?", (doc_id,))

    def _insert(self, page: Page, vector) -> None:
        # Caller holds the lock and the transaction.
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
        with self._lock:
            row = self._db.execute(
                "SELECT doc_id, page_no, doc_type, period, text, low_confidence"
                " FROM pages WHERE doc_id = ? AND page_no = ?",
                (doc_id, page_no),
            ).fetchone()
            if not row:
                return None
            return Page(row[0], row[1], row[2], row[3], row[4], bool(row[5]))

    def page_counts(self) -> dict[str, int]:
        with self._lock:
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
        with self._lock:
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
                if row is None:
                    continue
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
