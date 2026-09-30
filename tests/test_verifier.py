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
    quote = "remains \"cautious\" and that clients' budgets are tight"
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
