from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

from backend.models import Answer, Claim, VerifiedAnswer, VerifiedClaim

MIN_QUOTE_CHARS = 12
MIN_QUOTE_ALNUM = 8
WEAK_THRESHOLD = 90

_PUNCTUATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2012": "-",
        "\u2212": "-",
        "\ufe63": "-",
        "\uff0d": "-",
        "\u00ad": "",
    }
)

# Words whose presence or absence changes the meaning of a figure or a statement.
_PROTECTED = frozenset(
    "crore lakh thousand million billion trillion mn bn bps increase increased decrease decreased "
    "up down rose fell grew declined higher lower not no never without".split()
)
_WORD = re.compile(r"[^\W\d_]+")
_TRAILING = ".,;:)\"'!?]"
_LEADING = "\"'["


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_PUNCTUATION).lower()
    # A hyphen between LETTERS, optionally followed by a line break, is joined up
    # ("under-\nlying" == "underlying"). Hyphens next to digits (ranges, minus signs) are never joined.
    text = re.sub(r"(?<=[^\W\d_])-\s*(?=[^\W\d_])", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _numeric_tokens(text: str) -> list[str]:
    """Ordered whitespace tokens containing a digit (incl. FY26, Q2, 4G).

    A leading minus or opening parenthesis stays attached ("-5" != "5", "(5.2" != "5.2");
    trailing sentence punctuation is dropped so "21.1%", "21.1%," and "21.1%." agree.
    """
    tokens = []
    for raw in text.split():
        if not any(ch.isdigit() for ch in raw):
            continue
        token = raw.lstrip(_LEADING).rstrip(_TRAILING)
        if any(ch.isdigit() for ch in token):
            tokens.append(token)
    return tokens


def _protected_words(text: str) -> set[str]:
    return {word for word in _WORD.findall(text) if word in _PROTECTED}


def _has_clean_occurrence(quote: str, page: str) -> bool:
    """True if the quote occurs in the page on token boundaries (not inside a longer number or word)."""
    start = page.find(quote)
    while start != -1:
        end = start + len(quote)
        before = page[start - 1] if start > 0 else ""
        after = page[end] if end < len(page) else ""
        clean = True
        if quote[0].isalnum():
            if before.isalnum():
                clean = False
            elif quote[0].isdigit() and before in ("-", "(", ".", ","):
                clean = False
        if clean and (quote[-1].isalnum() or quote[-1] in ".,") and after.isalnum():
            clean = False
        if clean:
            return True
        start = page.find(quote, start + 1)
    return False


def _aligned_window(quote: str, page: str) -> str:
    """The part of the page best aligned with the quote, widened to whole whitespace-delimited tokens."""
    if len(quote) >= len(page):
        return page
    alignment = fuzz.partial_ratio_alignment(quote, page)
    start, end = alignment.dest_start, alignment.dest_end
    while start > 0 and not page[start - 1].isspace():
        start -= 1
    while end < len(page) and not page[end].isspace():
        end += 1
    return page[start:end]


def verify_claim(claim: Claim, page_text: str | None) -> str:
    if page_text is None:
        return "failed"
    quote = _normalize(claim.quote)
    if len(quote) < MIN_QUOTE_CHARS or sum(ch.isalnum() for ch in quote) < MIN_QUOTE_ALNUM:
        return "failed"
    page = _normalize(page_text)
    if _has_clean_occurrence(quote, page):
        return "verified"
    # Near-miss (OCR noise, spelling, or an occurrence glued into a longer number/word). Acceptable as
    # "weak" only if the aligned page text carries the same figures in the same order and the same
    # unit / direction / negation words.
    if fuzz.partial_ratio(quote, page) < WEAK_THRESHOLD:
        return "failed"
    window = _aligned_window(quote, page)
    if _numeric_tokens(window) != _numeric_tokens(quote):
        return "failed"
    if _protected_words(window) != _protected_words(quote):
        return "failed"
    return "weak"


def verify(answer: Answer, index) -> VerifiedAnswer:
    verified: list[VerifiedClaim] = []
    for claim in answer.claims:
        page = index.get_page(claim.doc_id, claim.page_no)
        verified.append(VerifiedClaim(claim, verify_claim(claim, page.text if page else None)))
    return VerifiedAnswer(summary=answer.summary, claims=verified, not_found=answer.not_found)
