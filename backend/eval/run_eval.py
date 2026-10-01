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
    failures: list[str] = field(default_factory=list)

    @property
    def correct_rate(self) -> float:
        return self.correct / self.total if self.total else 0.0


def _haystack(answer: VerifiedAnswer) -> str:
    parts = [answer.summary]
    for verified in answer.claims:
        parts += [verified.claim.text, verified.claim.quote]
    return "\n".join(parts).casefold()


def _is_correct(case: dict, answer: VerifiedAnswer) -> bool:
    if case.get("expect_not_found"):
        return answer.not_found
    if answer.not_found:
        return False
    haystack = _haystack(answer)
    return all(expected.casefold() in haystack for expected in case.get("expect_contains", []))


def evaluate(cases: list[dict], ask_fn) -> EvalResult:
    result = EvalResult(total=len(cases))
    for case in cases:
        # Vacuous pass guard: factual cases must have expect_contains
        if not case.get("expect_not_found") and not case.get("expect_contains"):
            raise ValueError(f"factual case needs expect_contains: {case['question']}")
        answer = ask_fn(case["question"])
        if _is_correct(case, answer):
            result.correct += 1
        else:
            result.failures.append(case["question"])
        if all(v.status != "failed" for v in answer.claims):
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
    for question in result.failures:
        print(f"  FAILED: {question}")
    return 0 if result.correct_rate >= PASS_RATE else 1


if __name__ == "__main__":
    raise SystemExit(main())
