from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from backend.models import VerifiedAnswer

PASS_RATE = 0.9


@dataclass
class EvalResult:
    total: int = 0
    correct: int = 0
    citations_clean: int = 0
    weak_claims: int = 0
    unsupported_claims: int = 0
    failed_claims: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def correct_rate(self) -> float:
        return self.correct / self.total if self.total else 0.0


def _verified(answer: VerifiedAnswer):
    return [v for v in answer.claims if v.status == "verified"]


def _haystack(answer: VerifiedAnswer) -> str:
    # Only verified claims count: the model-written summary and unchecked claims prove nothing.
    parts: list[str] = []
    for verified in _verified(answer):
        parts += [verified.claim.text, verified.claim.quote]
    return "\n".join(parts).casefold()


def _is_correct(case: dict, answer: VerifiedAnswer) -> bool:
    if case.get("expect_not_found"):
        return answer.not_found
    if answer.not_found:
        return False
    haystack = _haystack(answer)
    if not all(expected.casefold() in haystack for expected in case.get("expect_contains", [])):
        return False
    cited = {(v.claim.doc_id, v.claim.page_no) for v in _verified(answer)}
    return all((page["doc_id"], page["page_no"]) in cited for page in case.get("expect_pages", []))


def _validate(case: dict) -> None:
    question = case.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"case needs a non-empty question: {case}")
    # Vacuous pass guard: factual cases must have expect_contains
    if not case.get("expect_not_found") and not case.get("expect_contains"):
        raise ValueError(f"factual case needs expect_contains: {question}")
    pages = case.get("expect_pages", [])
    if not isinstance(pages, list):
        raise ValueError(f"expect_pages must be a list: {question}")
    for page in pages:
        if not (
            isinstance(page, dict)
            and isinstance(page.get("doc_id"), str)
            and isinstance(page.get("page_no"), int)
            and not isinstance(page.get("page_no"), bool)
        ):
            raise ValueError(f'expect_pages entries need {{"doc_id": str, "page_no": int}}: {question}')


def evaluate(cases: list[dict], ask_fn) -> EvalResult:
    for case in cases:
        _validate(case)
    result = EvalResult(total=len(cases))
    for case in cases:
        answer = ask_fn(case["question"])
        if _is_correct(case, answer):
            result.correct += 1
        else:
            result.failures.append(case["question"])
        statuses = [v.status for v in answer.claims]
        result.weak_claims += statuses.count("weak")
        result.unsupported_claims += statuses.count("unsupported")
        result.failed_claims += statuses.count("failed")
        if all(status == "verified" for status in statuses):
            result.citations_clean += 1
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the golden question set against the real agent.")
    parser.add_argument("golden", nargs="?", default="evals/golden.json")
    args = parser.parse_args(argv)

    from backend.verify.verifier import verify
    from backend.wiring import build_components

    _, _, index, agent = build_components()
    cases = json.loads(Path(args.golden).read_text())
    result = evaluate(cases, lambda q: verify(agent.ask(q), index))

    print(f"Correct:          {result.correct}/{result.total} ({result.correct_rate:.0%})")
    print(f"Citations clean:  {result.citations_clean}/{result.total}")
    print(f"Weak claims:        {result.weak_claims}")
    print(f"Unsupported claims: {result.unsupported_claims}")
    print(f"Failed claims:      {result.failed_claims}")
    for question in result.failures:
        print(f"  FAILED: {question}")
    return 0 if result.correct_rate >= PASS_RATE else 1


if __name__ == "__main__":
    raise SystemExit(main())
