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


# --- Fix round 1: hardening regressions -------------------------------------


def test_truncated_number_fails():
    assert verify_claim(claim("operating margin was 21.1"), "The operating margin was 21.17% for the quarter.") == "failed"


def test_truncated_year_fails():
    assert verify_claim(claim("revenue for fiscal 202"), "Revenue for fiscal 2025 rose in the quarter") == "failed"


def test_truncated_rupee_amount_fails():
    assert verify_claim(claim("net profit rose to Rs 4"), "Net profit rose to Rs 45 crore") == "failed"


def test_quote_starting_mid_number_fails():
    assert verify_claim(claim("1.1% for the quarter overall"), "Margin was 21.1% for the quarter overall") == "failed"


def test_dropped_hyphen_minus_fails():
    page = "Cash flow moved by -5% versus the previous quarter"
    assert verify_claim(claim("5% versus the previous quarter"), page) == "failed"


def test_dropped_unicode_minus_fails():
    page = "Cash flow moved by −5% versus the previous quarter"
    assert verify_claim(claim("5% versus the previous quarter"), page) == "failed"


def test_dropped_parentheses_negative_fails():
    assert verify_claim(claim("moved by 5.2 crore in the period"), "Cash moved by (5.2) crore in the period") == "failed"


def test_hyphen_range_not_joined_percent():
    assert verify_claim(claim("growth of 23% expected next year"), "Growth of 2-3% expected next year") == "failed"


def test_hyphen_range_not_joined_rupees():
    assert verify_claim(claim("EBITDA of Rs 1012 crore reported"), "EBITDA of Rs 10-12 crore reported") == "failed"


def test_hyphen_range_not_joined_fiscal_year():
    assert verify_claim(claim("fiscal 202526"), "Results for fiscal 2025-26 were strong") == "failed"


def test_swapped_figures_fail():
    page = "Revenue grew 12% in the quarter and attrition fell to 13% overall."
    quote = "Revenue grew 13% in the quarter and attrition fell to 12% overall."
    assert verify_claim(claim(quote), page) == "failed"


def test_changed_fiscal_year_label_fails():
    page = "Guidance for FY25 was revised upward after the strong quarter results."
    quote = "Guidance for FY26 was revised upward after the strong quarter results."
    assert verify_claim(claim(quote), page) == "failed"


def test_changed_quarter_label_fails():
    page = "Deal wins in Q2 were the highest in the last eight quarters overall."
    quote = "Deal wins in Q3 were the highest in the last eight quarters overall."
    assert verify_claim(claim(quote), page) == "failed"


def test_changed_unit_crore_lakh_fails():
    page = "Total order inflow for the period stood at Rs 1,200 crore across segments."
    quote = "Total order inflow for the period stood at Rs 1,200 lakh across segments."
    assert verify_claim(claim(quote), page) == "failed"


def test_changed_unit_million_billion_fails():
    page = "The company reported deal wins worth $4.2 million in the quarter just ended."
    quote = "The company reported deal wins worth $4.2 billion in the quarter just ended."
    assert verify_claim(claim(quote), page) == "failed"


def test_changed_direction_fails():
    page = "Attrition increased sharply during the quarter across all the business units."
    quote = "Attrition decreased sharply during the quarter across all the business units."
    assert verify_claim(claim(quote), page) == "failed"


def test_inserted_negation_fails():
    page = "Management expects demand to recover in the second half of the year ahead."
    quote = "Management does not expect demand to recover in the second half of the year ahead."
    assert verify_claim(claim(quote), page) == "failed"


def test_clean_boundary_quotes_still_verify():
    assert verify_claim(claim("Operating margin was 21.1%"), "Note: Operating margin was 21.1% for the quarter.") == "verified"
    assert verify_claim(claim("Operating margin was 21.1%"), "Operating margin was 21.1%, down sharply.") == "verified"
    assert verify_claim(claim("Operating margin was 21.1%"), "Operating margin was 21.1%. Next sentence.") == "verified"
    assert verify_claim(claim("Operating margin was 21.1%"), "Headline\nOperating margin was 21.1%\nnext") == "verified"
    assert verify_claim(claim("Operating margin was 21.1"), "Operating margin was 21.1. Next") == "verified"


def test_later_clean_occurrence_is_enough():
    page = "Margin was 21.17% in one place. Elsewhere operating margin was 21.1% overall."
    assert verify_claim(claim("operating margin was 21.1"), page) == "verified"


def test_year_on_year_hyphens_still_verify():
    assert verify_claim(claim("year-on-year growth was strong"), "Year-on-year growth was strong in Q2") == "verified"


def test_punctuation_only_quote_fails():
    page = "Results ............ shown here"
    assert verify_claim(claim("." * 12), page) == "failed"
    assert verify_claim(claim("-" * 12), "Results ------------ shown here") == "failed"
