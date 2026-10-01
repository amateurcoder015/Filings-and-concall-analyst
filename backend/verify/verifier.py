"""Deterministic quote verifier: checks each claim's quote against the stored page text.

Invariant: anything not provably identical in figures, signs, units, direction and negation goes to
"weak" or "failed"; "weak" never counts as verified; a claim whose figures are not in its quote is
"unsupported".

Accepted limits:
- Number reformatting such as 1200 vs 1,200 fails.
- Semantic flips among long plain words at edit distance 1 can reach "weak".
- A quote that omits a nearby qualifier still verifies.
"""
from __future__ import annotations

import difflib
import re
import unicodedata

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from backend.models import Answer, Claim, VerifiedAnswer, VerifiedClaim

MIN_QUOTE_CHARS = 12
MIN_QUOTE_ALNUM = 8
WEAK_THRESHOLD = 90
MAX_REPLACE_DISTANCE = 1
# Words shorter than this (after squashing) must match exactly: a one-letter edit on a short token is a
# different metric, unit, currency or ticker (PAT/PBT, MW/GW, INR/IDR, TCS/TVS).
MIN_EDIT_LETTERS = 6
MIN_CUT_LETTERS = 3
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
    nor neither cannot positive negative from to
    hardly barely scarcely seldom rarely nothing nobody nowhere
    non un none one two three four five six seven eight nine ten eleven twelve twenty thirty forty fifty
    sixty seventy eighty ninety hundred hundreds first second third half double triple zero nil
    over under about approximately nearly almost around roughly more less than only least most exceeding
    exceeds exceeded within between max maximum min minimum at
    ebit ebita ebitda ebitdar
    soared soar soars soured sour surged surge slumped slump plunged plunge jumped jump climbed climb
    dropped drop tumbled tumble slipped slip improved improve worsened worsen widened widen narrowed narrow
    expanded expand contracted contract strengthened weakened rallied slid sank dipped eased tightened
    loosened""".split()
)
_PLAIN_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")
_CHUNK_TRAILING = ".,;:\"'"
# Characters that, glued to the start / end of a match, make it part of a longer figure or token.
_BAD_BEFORE = frozenset("-($\u20b9\u20ac\u00a3/%")
# Comparison / sign / footnote marks: they only change the meaning of a figure, so they make the match
# unclean only when the quote starts with a digit or a currency sign.
_BAD_BEFORE_FIGURE = frozenset("<>~\u2248\u00b1+*\u2264\u2265")
_FIGURE_START = frozenset("$\u20b9\u20ac\u00a3")
_BAD_AFTER = frozenset("-/")
# Prefixes that negate, oppose or re-qualify the rest of the word (profitable / unprofitable,
# performing / nonperforming, typical / atypical, tax / pretax). Hyphens between letters are joined by
# _normalize, so "non-performing" is "nonperforming" here.
_NEGATING_PREFIXES = (
    "un", "non", "in", "im", "ir", "il", "dis", "de", "mis", "anti", "under", "over", "out",
    "a", "an", "ab", "mal", "counter", "contra", "sub", "pre", "post", "ex",
    "after", "before", "semi", "bi", "mega", "kilo", "giga", "milli", "micro",
)
# Suffixes that negate the start of the word (worth / worthless, debt / debtfree).
_NEGATING_SUFFIXES = ("less", "free")
# Words where a prefix above does not negate the rest ("underlying" is not the opposite of "lying"),
# so a quote boundary may still cut them.
_NON_NEGATING_COMPOUNDS = frozenset(
    "underlying undertaking undertakings understanding outstanding overall".split()
)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_PUNCTUATION).lower()
    # A hyphen between LETTERS, optionally followed by a line break, is joined up
    # ("under-\nlying" == "underlying"). Hyphens next to digits (ranges, minus signs) are never joined.
    text = re.sub(r"(?<=[^\W\d_])-\s*(?=[^\W\d_])", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _figure_led(quote: str) -> bool:
    """The quote opens with a figure: a digit or currency sign, or a digit within its first two chunks
    ("Rs 5", "USD 5", "INR 5", "Re 1", "Rs. 5")."""
    if quote[0].isdigit() or quote[0] in _FIGURE_START:
        return True
    return any(ch.isdigit() for chunk in quote.split()[:2] for ch in chunk)


def _is_clean(quote: str, page: str, start: int) -> bool:
    end = start + len(quote)
    if start > 0:
        before = page[start - 1]
        if before.isalnum() or before in _BAD_BEFORE:
            return False
        if before in _BAD_BEFORE_FIGURE and _figure_led(quote):
            return False
        if before in ".," and quote[0].isdigit():
            return False
    if end < len(page):
        after = page[end]
        if after.isalnum() or after in _BAD_AFTER:
            return False
        if after in ".,:" and end + 1 < len(page) and page[end + 1].isdigit():
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


def _squash(word: str) -> str:
    # "rn" -> "m" is a classic OCR confusion, applied to both sides before comparing.
    return word.replace("rn", "m")


def _negating_prefix(removed: str) -> bool:
    """`removed` (the part of a page word left out of the quote's first word) is or starts with a prefix
    that negates / re-qualifies the rest, or is itself a protected word ("down" + "graded")."""
    if removed in _PROTECTED or removed in _NEGATING_PREFIXES:
        return True
    return any(removed.startswith(prefix) for prefix in _NEGATING_PREFIXES if len(prefix) >= 2)


def _negating_suffix(removed: str) -> bool:
    return removed in _PROTECTED or removed.startswith(_NEGATING_SUFFIXES)


def _near_identical_word(x: str, y: str) -> bool:
    if x == y:
        return True
    if min(len(x), len(y)) < MIN_EDIT_LETTERS or Levenshtein.distance(x, y) > MAX_REPLACE_DISTANCE:
        return False
    # A one-letter negating prefix ("typical" / "atypical", "symmetric" / "asymmetric").
    for longer, shorter in ((x, y), (y, x)):
        if longer.endswith(shorter) and _negating_prefix(longer[: len(longer) - len(shorter)]):
            return False
    # The other party of a relationship ("employer" / "employee", "drawer" / "drawee").
    if x[:-1] == y[:-1] and {x[-1], y[-1]} == {"r", "e"} and (x.endswith("ee") or y.endswith("ee")):
        return False
    return True


def _near_identical(q_side: list[str], w_side: list[str]) -> bool:
    q_words, w_words = [_squash(w) for w in q_side], [_squash(w) for w in w_side]
    if len(q_words) != len(w_words):
        # Different word split ("per cent" / "percent"): the letters must be identical.
        return "".join(q_words) == "".join(w_words)
    # Word by word, so the single tolerated edit can never land in a short token.
    differing = [(x, y) for x, y in zip(q_words, w_words) if x != y]
    return len(differing) <= MAX_REPLACE_DISTANCE and all(_near_identical_word(x, y) for x, y in differing)


def _cut_of(cut: str, full: str, *, suffix: bool) -> bool:
    """`cut` is a proper suffix (or prefix) of `full`: the quote boundary sliced a plain word.

    Refused when the sliced-off part negates what is left ("un" + "profitable", "worth" + "less").
    """
    # Short page words are tokens (NPA / GNPA, GNP / GNPA): never sliced.
    if len(cut) < MIN_CUT_LETTERS or len(cut) >= len(full) or len(full) < MIN_EDIT_LETTERS:
        return False
    if not (_is_plain_word(cut) and _is_plain_word(full)):
        return False
    if suffix:
        if not full.endswith(cut):
            return False
        return full in _NON_NEGATING_COMPOUNDS or not _negating_prefix(full[: len(full) - len(cut)])
    if not full.startswith(cut):
        return False
    return not _negating_suffix(full[len(cut):])


def _same_skeleton(quote: str, window: str) -> bool:
    """Quote and page window agree chunk for chunk, except near-identical plain words."""
    q_chunks, w_chunks = _chunks(quote), _chunks(window)
    if not q_chunks or not w_chunks:
        return False
    # A quote boundary may cut the first / last word in half; that is not a difference.
    if _cut_of(q_chunks[0], w_chunks[0], suffix=True):
        q_chunks[0] = w_chunks[0]
    if _cut_of(q_chunks[-1], w_chunks[-1], suffix=False):
        q_chunks[-1] = w_chunks[-1]
    matcher = difflib.SequenceMatcher(None, q_chunks, w_chunks, autojunk=False)
    inserted_or_deleted = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        q_side, w_side = q_chunks[i1:i2], w_chunks[j1:j2]
        if not all(_is_plain_word(chunk) for chunk in q_side + w_side):
            return False
        if tag == "replace":
            if not _near_identical(q_side, w_side):
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


_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
# Characters that chain a number to its neighbours inside one token ("FY2025-26", "10/12", "1,234.5").
_NUMBER_CHAIN = frozenset("0123456789-/.,")


def _letter_glued(text: str, start: int, end: int) -> bool:
    """The number at text[start:end] is part of a token that contains a letter (FY26, Q2, 4G,
    FY2025-26, 5-year): walk outwards through digits and joiners and look for a letter."""
    i = start - 1
    while i >= 0 and text[i] in _NUMBER_CHAIN:
        i -= 1
    if i >= 0 and (text[i].isalpha() or text[i] == "_"):
        return True
    j = end
    while j < len(text) and text[j] in _NUMBER_CHAIN:
        j += 1
    if j < len(text) and text[j] == "%":
        return False
    return j < len(text) and (text[j].isalpha() or text[j] == "_")


def figure_tokens(text: str) -> set[str]:
    """Pure numeric figures in `text`, canonicalised for comparison.

    "1,234" == "1234", "21.1%" == "21.1", "-5" != "5", "(5.2)" == "-5.2". Letter-glued tokens (FY26, Q2,
    4G, H1) and spelled-out numbers are not figures.
    """
    text = _normalize(text)
    tokens: set[str] = set()
    for match in _NUMBER.finditer(text):
        start, end = match.span()
        if _letter_glued(text, start, end):
            continue
        value = match.group().rstrip(".,").replace(",", "")
        if not value:
            continue
        negative = False
        # A minus glued to the digit is a sign, unless it joins two figures ("2025-26", "10-12").
        if start > 0 and text[start - 1] == "-" and (start < 2 or not text[start - 2].isalnum()):
            negative = True
        # Accounting negative: "(5.2)" or "(5.2%)".
        tail = text[end:].lstrip(",")
        if start > 0 and text[start - 1] == "(" and (tail.startswith(")") or tail.startswith("%)")):
            negative = True
        if float(value) == 0:
            negative = False
        tokens.add(f"-{value}" if negative else value)
    return tokens


def verify(answer: Answer, index) -> VerifiedAnswer:
    verified: list[VerifiedClaim] = []
    for claim in answer.claims:
        page = index.get_page(claim.doc_id, claim.page_no)
        status = verify_claim(claim, page.text if page else None)
        if status == "verified" and not figure_tokens(claim.text) <= figure_tokens(claim.quote):
            # The quote is on the page, but the claim states a figure the quote does not contain.
            status = "unsupported"
        verified.append(VerifiedClaim(claim, status))
    summary_supported = True
    if not answer.not_found:
        backed: set[str] = set()
        for item in verified:
            if item.status == "verified":
                backed |= figure_tokens(item.claim.quote)
        summary_supported = figure_tokens(answer.summary) <= backed
    return VerifiedAnswer(
        summary=answer.summary,
        claims=verified,
        not_found=answer.not_found,
        summary_supported=summary_supported,
    )
