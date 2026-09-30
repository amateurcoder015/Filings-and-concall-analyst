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
