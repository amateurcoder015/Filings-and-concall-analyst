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
    status: str  # "verified" | "weak" | "unsupported" | "failed"


@dataclass
class VerifiedAnswer:
    summary: str
    claims: list[VerifiedClaim]
    not_found: bool
    # Every figure in the summary appears in the quote of a verified claim (True for not_found answers).
    summary_supported: bool = True

    @property
    def mostly_unverified(self) -> bool:
        if not self.claims:
            return False
        bad = sum(1 for c in self.claims if c.status != "verified")
        return bad * 2 > len(self.claims)
