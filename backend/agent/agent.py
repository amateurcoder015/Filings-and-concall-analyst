from __future__ import annotations

from backend.agent.tools import TOOLS, build_system_prompt
from backend.models import Answer, Claim

MAX_STEPS = 8
MAX_TOKENS = 4096
MAX_PAGE_CHARS = 8000
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
        for _ in range(MAX_STEPS):
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
        last_error: Exception | None = None
        for _ in range(2):
            try:
                return self._client.messages.create(
                    model=self._model,
                    max_tokens=MAX_TOKENS,
                    system=system,
                    tools=TOOLS,
                    messages=messages,
                )
            except Exception as exc:  # network, rate limit, overload: retry once, then surface
                last_error = exc
        raise AgentError(f"Claude API call failed: {last_error}") from last_error

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
            try:
                page = self._index.get_page(str(args["doc_id"]), int(args["page_no"]))
            except (KeyError, TypeError, ValueError):
                return "read_page needs doc_id and an integer page_no.", True
            if page is None:
                return "No such page.", True
            note = "[low-confidence page: text may be incomplete or OCR noise]\n" if page.low_confidence else ""
            body = page.text[:MAX_PAGE_CHARS]
            return f'{note}<page doc_id="{page.doc_id}" page_no="{page.page_no}">\n{body}\n</page>', False
        return f"Unknown tool '{name}'.", True
