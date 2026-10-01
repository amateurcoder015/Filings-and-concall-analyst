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


# --- Final fixes: verified-only scoring, status totals, expect_pages, validation ----

from backend.eval import run_eval  # noqa: E402
from backend.eval.run_eval import EvalResult  # noqa: E402


def mixed(summary, items, not_found=False):
    """items: (status, claim_text, quote, doc_id, page_no)"""
    claims = [VerifiedClaim(Claim(t, d, p, q), s) for s, t, q, d, p in items]
    return VerifiedAnswer(summary, claims, not_found)


FACT = {"question": "margin?", "expect_contains": ["21.1%"], "expect_not_found": False}


def test_summary_alone_no_longer_makes_a_case_correct():
    a = mixed("Margin was 21.1%.", [("verified", "Margin held", "Operating margin was strong in the quarter", "d", 1)])
    assert evaluate([FACT], lambda q: a).correct == 0


@pytest.mark.parametrize("status", ["weak", "unsupported", "failed"])
def test_only_verified_claims_count_for_correctness(status):
    a = mixed("x", [(status, "Margin was 21.1%", "Operating margin was 21.1% in the quarter", "d", 1)])
    assert evaluate([FACT], lambda q: a).correct == 0


def test_status_totals_and_citations_clean_need_every_claim_verified():
    good = mixed("x", [("verified", "Margin 21.1%", "Operating margin was 21.1% in the quarter", "d", 1)])
    noisy = mixed(
        "x",
        [
            ("verified", "Margin 21.1%", "Operating margin was 21.1% in the quarter", "d", 1),
            ("weak", "a", "b", "d", 1),
            ("unsupported", "a", "b", "d", 1),
            ("unsupported", "a", "b", "d", 1),
            ("failed", "a", "b", "d", 1),
        ],
    )
    answers = iter([good, noisy])
    result = evaluate([FACT, dict(FACT, question="again?")], lambda q: next(answers))
    assert result.correct == 2
    assert result.citations_clean == 1
    assert (result.weak_claims, result.unsupported_claims, result.failed_claims) == (1, 2, 1)


def test_weak_claim_makes_citations_unclean():
    a = mixed("x", [("verified", "Margin 21.1%", "Operating margin was 21.1% in the quarter", "d", 1), ("weak", "a", "b", "d", 1)])
    assert evaluate([FACT], lambda q: a).citations_clean == 0


def test_expect_pages_requires_every_page_among_verified_citations():
    case = dict(FACT, expect_pages=[{"doc_id": "q2", "page_no": 3}, {"doc_id": "ar", "page_no": 10}])
    both = mixed(
        "x",
        [
            ("verified", "Margin 21.1%", "Operating margin was 21.1% in the quarter", "q2", 3),
            ("verified", "Context", "Some context sentence", "ar", 10),
        ],
    )
    one_weak = mixed(
        "x",
        [
            ("verified", "Margin 21.1%", "Operating margin was 21.1% in the quarter", "q2", 3),
            ("weak", "Context", "Some context sentence", "ar", 10),
        ],
    )
    wrong_page = mixed("x", [("verified", "Margin 21.1%", "Operating margin was 21.1% in the quarter", "q2", 4)])
    assert evaluate([case], lambda q: both).correct == 1
    assert evaluate([case], lambda q: one_weak).correct == 0
    assert evaluate([case], lambda q: wrong_page).correct == 0


@pytest.mark.parametrize(
    "bad_case",
    [
        {"question": "q", "expect_not_found": False},
        {"question": "q", "expect_contains": ["x"], "expect_pages": {"doc_id": "d", "page_no": 1}},
        {"question": "q", "expect_contains": ["x"], "expect_pages": ["d:1"]},
        {"question": "q", "expect_contains": ["x"], "expect_pages": [{"doc_id": "d"}]},
        {"question": "q", "expect_contains": ["x"], "expect_pages": [{"doc_id": 5, "page_no": 1}]},
        {"question": "q", "expect_contains": ["x"], "expect_pages": [{"doc_id": "d", "page_no": "1"}]},
        {"question": "q", "expect_contains": ["x"], "expect_pages": [{"doc_id": "d", "page_no": True}]},
    ],
)
def test_all_cases_are_validated_before_the_first_ask(bad_case):
    calls = []

    def ask(q):
        calls.append(q)
        return answer("Margin was 21.1%.", ["verified"])

    with pytest.raises(ValueError):
        evaluate([FACT, bad_case], ask)
    assert calls == []


def test_main_prints_status_totals(tmp_path, monkeypatch, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text("[]")
    monkeypatch.setattr("backend.wiring.build_components", lambda: ("Infosys", [], object(), object()))
    monkeypatch.setattr(
        run_eval,
        "evaluate",
        lambda cases, ask_fn: EvalResult(total=2, correct=2, citations_clean=1, weak_claims=3, failed_claims=4, unsupported_claims=5),
    )
    assert run_eval.main([str(golden)]) == 0
    out = capsys.readouterr().out
    assert "Weak claims:" in out and "3" in out
    assert "Unsupported claims:" in out and "5" in out
    assert "Failed claims:" in out and "4" in out
