from __future__ import annotations

import re
import time

from backend.agent.tools import TOOLS, build_system_prompt
from backend.models import Answer, Claim

MAX_STEPS = 8
MAX_TOKENS = 4096
MAX_PAGE_CHARS = 12000
MAX_QUOTE_CHARS = 1000
ASK_DEADLINE_SECONDS = 180
# HTTP statuses worth one retry: timeout, conflict, rate limit, server errors, overload.
TRANSIENT_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
_PAGE_CLOSE = re.compile(r"</page>", re.IGNORECASE)
NOT_FOUND_TEXT = "This is not in the loaded filings."
NUDGE_TEXT = "Finish by calling the submit_answer tool with your answer."
CUT_OFF_TEXT = (
    "Your last reply was cut off. Be brief: use at most 3 claims with short verbatim quotes "
    "(one sentence each), then call submit_answer."
)
GIVE_UP_TEXT = "I could not produce a properly cited answer. Try rephrasing or narrowing the question."


class AgentError(Exception):
    pass


def parse_answer(data: dict) -> Answer | str:
    """Return an Answer, or a message telling the model what to fix."""
    summary = str(data.get("summary", "")).strip()
    if data.get("not_found"):
        return Answer(summary=summary or NOT_FOUND_TEXT, claims=[], not_found=True)
    raw_claims = data.get("claims") or []
    if not raw_claims:
        return (
            "No claims given. Every answer needs claims with doc_id, page_no and a verbatim quote, "
            "or set not_found=true."
        )
    claims: list[Claim] = []
    for position, raw in enumerate(raw_claims, start=1):
        try:
            claim = Claim(
                text=str(raw["text"]).strip(),
                doc_id=str(raw["doc_id"]).strip(),
                page_no=int(raw["page_no"]),
                quote=str(raw["quote"]).strip(),
            )
        except (KeyError, TypeError, ValueError):
            return f"Claim {position} is malformed: each needs text, doc_id, page_no (integer) and quote."
        if not claim.text or not claim.quote:
            return f"Claim {position} has an empty text or quote; every claim needs a verbatim quote."
        if len(claim.quote) > MAX_QUOTE_CHARS:
            return (
                f"Claim {position} has a quote longer than {MAX_QUOTE_CHARS} characters. Use a shorter verbatim "
                "quote: the one or two sentences that contain the figures you rely on."
            )
        claims.append(claim)
    return Answer(summary=summary, claims=claims, not_found=False)


class FilingsAgent:
    def __init__(self, client, index, model: str, company: str = "Infosys"):
        self._client = client
        self._index = index
        self._model = model
        self._company = company

    def ask(self, question: str, period: str | None = None) -> Answer:
        system = build_system_prompt(self._company, period)
        messages: list[dict] = [{"role": "user", "content": question}]
        rejected = 0
        deadline = time.monotonic() + ASK_DEADLINE_SECONDS
        for _ in range(MAX_STEPS):
            if time.monotonic() > deadline:
                return Answer(summary=GIVE_UP_TEXT, claims=[], not_found=True)
            response = self._create(system, messages)
            if getattr(response, "stop_reason", None) == "max_tokens":
                # cut off mid-reply: any tool_use block may be incomplete, so drop the turn entirely
                messages.append({"role": "user", "content": CUT_OFF_TEXT})
                continue
            if not response.content:
                messages.append({"role": "user", "content": NUDGE_TEXT})
                continue
            messages.append({"role": "assistant", "content": response.content})
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                messages.append(
                    {"role": "user", "content": NUDGE_TEXT}
                )
                continue
            results = []
            for block in tool_uses:
                if block.name == "submit_answer":
                    parsed = (
                        parse_answer(block.input)
                        if isinstance(block.input, dict)
                        else "submit_answer input must be an object."
                    )
                    if isinstance(parsed, Answer):
                        return parsed
                    rejected += 1
                    if rejected > 1:
                        return Answer(summary=GIVE_UP_TEXT, claims=[], not_found=True)
                    content, is_error = parsed, True
                else:
                    content, is_error = self._run_tool(block.name, block.input)
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": content, "is_error": is_error}
                )
            messages.append({"role": "user", "content": results})
        return Answer(summary=GIVE_UP_TEXT, claims=[], not_found=True)

    def _create(self, system: str, messages: list[dict]):
        for attempt in range(2):
            try:
                return self._client.messages.create(
                    model=self._model,
                    max_tokens=MAX_TOKENS,
                    system=system,
                    tools=TOOLS,
                    messages=messages,
                )
            except Exception as exc:
                # Only network trouble, rate limits and overload get one retry; a bad request or key never
                # succeeds on a second try.
                if attempt == 0 and _is_transient(exc):
                    continue
                raise AgentError(f"Claude API call failed: {exc}") from exc
        raise AssertionError("unreachable")

    def _run_tool(self, name: str, args: dict) -> tuple[str, bool]:
        if not isinstance(args, dict):
            return "Tool input must be an object.", True
        if name == "search":
            doc_type, period = args.get("doc_type"), args.get("period")
            if any(v is not None and not isinstance(v, str) for v in (doc_type, period)):
                return "search arguments doc_type and period must be strings.", True
            hits = self._index.search(
                str(args.get("query", "")),
                top_k=8,
                doc_type=doc_type,
                period=period,
            )
            if not hits:
                return "No matching pages.", False
            lines = [f"{h.doc_id} p.{h.page_no} [{h.doc_type} {h.period}] {h.snippet}" for h in hits]
            return "\n".join(lines), False
        if name == "read_page":
            return self._read_page(args)
        return f"Unknown tool '{name}'.", True

    def _read_page(self, args: dict) -> tuple[str, bool]:
        try:
            page = self._index.get_page(str(args["doc_id"]), int(args["page_no"]))
        except (KeyError, TypeError, ValueError):
            return "read_page needs doc_id and an integer page_no.", True
        offset = args.get("offset", 0)
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            return "read_page offset must be a non-negative integer.", True
        if page is None:
            return "No such page.", True
        note = "[low-confidence page: text may be incomplete or OCR noise]\n" if page.low_confidence else ""
        total = len(page.text)
        if offset > 0 and offset >= total:
            body = "[end of page]"
        else:
            end = min(offset + MAX_PAGE_CHARS, total)
            body = page.text[offset:end]
            if end < total:
                body += (
                    f"\n[page truncated: showing characters {offset}-{end} of {total}; "
                    f"call read_page with offset={end} to continue]"
                )
        # Page text must not be able to close the wrapper and pose as instructions outside it.
        body = _PAGE_CLOSE.sub(r"<\\/page>", body)
        return f'{note}<page doc_id="{page.doc_id}" page_no="{page.page_no}">\n{body}\n</page>', False


def _is_transient(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) in TRANSIENT_STATUS_CODES:
        return True
    try:
        import anthropic
    except ImportError:
        return False
    return isinstance(exc, (anthropic.APIConnectionError, anthropic.APITimeoutError))
