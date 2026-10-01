import pytest

from backend.eval.run_eval import evaluate
from backend.models import Claim, VerifiedAnswer, VerifiedClaim


def answer(summary, statuses=(), quote="Operating margin was 21.1% in the quarter", claim_text=None, not_found=False):
    if claim_text is None:
        claim_text = quote
    claims = [VerifiedClaim(Claim(claim_text, "d", 1, quote), s) for s in statuses]
    return VerifiedAnswer(summary, claims, not_found)


def test_factual_case_passes_when_expected_strings_present():
    cases = [{"question": "margin?", "expect_contains": ["21.1%"], "expect_not_found": False}]
    result = evaluate(cases, lambda q: answer("Margin was 21.1%.", ["verified"]))
    assert result.total == 1 and result.correct == 1 and result.citations_clean == 1


def test_factual_case_fails_when_expected_string_missing_or_not_found():
    cases = [{"question": "margin?", "expect_contains": ["21.1%"], "expect_not_found": False}]
    assert evaluate(cases, lambda q: answer("Margin was 25%.", ["verified"], quote="Operating margin was 25% in the quarter")).correct == 0
    assert evaluate(cases, lambda q: answer("nope", [], not_found=True)).correct == 0


def test_expected_string_match_is_case_insensitive_and_searches_quotes():
    cases = [{"question": "q", "expect_contains": ["WAGE HIKES"], "expect_not_found": False}]
    result = evaluate(cases, lambda q: answer("Pressure on margins.", ["verified"], quote="Management cited wage hikes as the main driver"))
    assert result.correct == 1


def test_adversarial_case_requires_not_found():
    cases = [{"question": "share price?", "expect_contains": [], "expect_not_found": True}]
    assert evaluate(cases, lambda q: answer("not in filings", [], not_found=True)).correct == 1
    assert evaluate(cases, lambda q: answer("It is 1500.", ["verified"])).correct == 0


def test_failed_claims_reduce_citations_clean_not_correctness():
    cases = [{"question": "margin?", "expect_contains": ["21.1%"], "expect_not_found": False}]
    result = evaluate(cases, lambda q: answer("Margin was 21.1%.", ["verified", "failed"]))
    assert result.correct == 1 and result.citations_clean == 0


def test_failures_list_names_the_question():
    cases = [{"question": "What was margin?", "expect_contains": ["99%"], "expect_not_found": False}]
    result = evaluate(cases, lambda q: answer("x", ["verified"]))
    assert result.failures == ["What was margin?"]
    assert result.correct_rate == 0.0


def test_expected_string_in_claim_text_only():
    cases = [{"question": "q", "expect_contains": ["WAGE HIKES"], "expect_not_found": False}]
    result = evaluate(cases, lambda q: answer("Pressure on margins.", ["verified"], quote="Management cited the main driver", claim_text="wage hikes"))
    assert result.correct == 1


def test_factual_case_without_expect_contains_raises():
    cases = [{"question": "What is margin?", "expect_not_found": False}]
    with pytest.raises(ValueError, match="factual case needs expect_contains"):
        evaluate(cases, lambda q: answer("x", ["verified"]))
