from __future__ import annotations

import difflib
import re
import unicodedata

from rapidfuzz import fuzz

from backend.models import Answer, Claim, VerifiedAnswer, VerifiedClaim

MIN_QUOTE_CHARS = 12
MIN_QUOTE_ALNUM = 8
WEAK_THRESHOLD = 90
REPLACE_MIN_RATIO = 80
MAX_INSERT_DELETE = 1

_PUNCTUATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\ufe63": "-",
        "\uff0d": "-",
        "\u00ad": "",
    }
)

# Words whose change flips the meaning of a figure or a statement. They may never sit in a differing
# region of a "weak" match.
_PROTECTED = frozenset(
    """crore crores lakh lakhs thousand thousands million millions billion billions trillion mn bn bps
    increase increased increases decrease decreased decreases up down rose rise fell fall grew growth
    declined decline higher lower above below profit profits loss losses gain gains not no never without
    nor neither cannot positive negative from to""".split()
)
_PLAIN_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")
_CHUNK_TRAILING = ".,;:\"'"
# Characters that, glued to the start / end of a match, make it part of a longer figure or token.
_BAD_BEFORE = frozenset("-($\u20b9\u20ac\u00a3/%")
_BAD_AFTER = frozenset("-/")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_PUNCTUATION).lower()
    # A hyphen between LETTERS, optionally followed by a line break, is joined up
    # ("under-\nlying" == "underlying"). Hyphens next to digits (ranges, minus signs) are never joined.
    text = re.sub(r"(?<=[^\W\d_])-\s*(?=[^\W\d_])", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_clean(quote: str, page: str, start: int) -> bool:
    end = start + len(quote)
    if start > 0:
        before = page[start - 1]
        if before.isalnum() or before in _BAD_BEFORE:
            return False
        if before in ".," and quote[0].isdigit():
            return False
    if end < len(page):
        after = page[end]
        if after.isalnum() or after in _BAD_AFTER:
            return False
        if after in ".," and end + 1 < len(page) and page[end + 1].isdigit():
            return False
    return True


def _has_clean_occurrence(quote: str, page: str) -> bool:
    """True if the quote occurs in the page with clean boundaries on both sides."""
    start = page.find(quote)
    while start != -1:
        if _is_clean(quote, page, start):
            return True
        start = page.find(quote, start + 1)
    return False


def _aligned_window(quote: str, page: str) -> str:
    """The part of the page best aligned with the quote, widened to whole whitespace-delimited chunks."""
    if len(quote) >= len(page):
        return page
    alignment = fuzz.partial_ratio_alignment(quote, page)
    start, end = alignment.dest_start, alignment.dest_end
    while start > 0 and not page[start - 1].isspace():
        start -= 1
    while end < len(page) and not page[end].isspace():
        end += 1
    return page[start:end]


def _chunks(text: str) -> list[str]:
    return [chunk for chunk in (raw.rstrip(_CHUNK_TRAILING) for raw in text.split()) if chunk]


def _is_plain_word(chunk: str) -> bool:
    return bool(_PLAIN_WORD.fullmatch(chunk)) and chunk not in _PROTECTED and not chunk.endswith("n't")


def _same_skeleton(quote: str, window: str) -> bool:
    """Quote and page window agree chunk for chunk, except near-identical plain words."""
    q_chunks, w_chunks = _chunks(quote), _chunks(window)
    matcher = difflib.SequenceMatcher(None, q_chunks, w_chunks, autojunk=False)
    inserted_or_deleted = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        q_side, w_side = q_chunks[i1:i2], w_chunks[j1:j2]
        if not all(_is_plain_word(chunk) for chunk in q_side + w_side):
            return False
        if tag == "replace":
            if fuzz.ratio("".join(q_side), "".join(w_side)) < REPLACE_MIN_RATIO:
                return False
        else:
            inserted_or_deleted += len(q_side) + len(w_side)
            if inserted_or_deleted > MAX_INSERT_DELETE:
                return False
    return True


def verify_claim(claim: Claim, page_text: str | None) -> str:
    if page_text is None:
        return "failed"
    quote = _normalize(claim.quote)
    if len(quote) < MIN_QUOTE_CHARS or sum(ch.isalnum() for ch in quote) < MIN_QUOTE_ALNUM:
        return "failed"
    page = _normalize(page_text)
    if _has_clean_occurrence(quote, page):
        return "verified"
    # Near-miss (OCR noise, cut words, or a hit glued into a longer number/word). "weak" only if the
    # aligned page text has the same token skeleton: every figure, sign, currency mark, unit, direction
    # and negation word identical, with only near-identical ordinary words differing.
    if fuzz.partial_ratio(quote, page) < WEAK_THRESHOLD:
        return "failed"
    if not _same_skeleton(quote, _aligned_window(quote, page)):
        return "failed"
    return "weak"


def verify(answer: Answer, index) -> VerifiedAnswer:
    verified: list[VerifiedClaim] = []
    for claim in answer.claims:
        page = index.get_page(claim.doc_id, claim.page_no)
        verified.append(VerifiedClaim(claim, verify_claim(claim, page.text if page else None)))
    return VerifiedAnswer(summary=answer.summary, claims=verified, not_found=answer.not_found)
