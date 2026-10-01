from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from backend.agent.agent import AgentError
from backend.models import DocMeta
from backend.verify.verifier import verify

DISCLAIMER = (
    "For research and education only. Nothing here is investment advice or a recommendation."
)


class AskRequest(BaseModel):
    question: str = Field(max_length=1000)
    period: Optional[str] = Field(default=None, max_length=32)


def create_app(index, agent, docs: list[DocMeta], company: str, pdf_dir: Path) -> FastAPI:
    app = FastAPI(title="Filings & Concall Analyst")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    by_id = {d.doc_id: d for d in docs}
    periods = {d.period for d in docs}

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception):
        logging.exception("Unhandled error while serving %s", request.url.path, exc_info=exc)
        return JSONResponse(
            status_code=500, content={"detail": "Internal error while answering. Check the server log."}
        )

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
        if request.period is not None and request.period not in periods:
            raise HTTPException(status_code=422, detail="Unknown period.")
        try:
            answer = agent.ask(question, period=request.period)
        except AgentError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        verified = verify(answer, index)
        return {
            "summary": verified.summary,
            "not_found": verified.not_found,
            "mostly_unverified": verified.mostly_unverified,
            "summary_supported": verified.summary_supported,
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
        path = Path(pdf_dir) / meta.file
        if not path.exists():
            raise HTTPException(status_code=404, detail="PDF file is missing on the server.")
        return FileResponse(path, media_type="application/pdf")

    return app
