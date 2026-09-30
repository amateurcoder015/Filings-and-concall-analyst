from __future__ import annotations

TOOLS = [
    {
        "name": "search",
        "description": (
            "Search the loaded filings for pages relevant to a query. Returns up to 8 pages with "
            "doc_id, page number, document type, period and a snippet. Use short keyword-style queries."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "doc_type": {"type": "string", "enum": ["annual_report", "results", "concall"]},
                "period": {"type": "string", "description": "e.g. 'Q2 FY26' or 'FY25'"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_page",
        "description": "Read the full text of one page. Always read a page before quoting from it.",
        "input_schema": {
            "type": "object",
            "properties": {"doc_id": {"type": "string"}, "page_no": {"type": "integer"}},
            "required": ["doc_id", "page_no"],
        },
    },
    {
        "name": "submit_answer",
        "description": (
            "Submit the final answer. Every claim needs doc_id, page_no and a verbatim quote copied "
            "exactly from that page. If the filings do not contain the answer, set not_found=true."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "One or two sentence plain-language answer."},
                "not_found": {"type": "boolean"},
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "The claim in your own words."},
                            "doc_id": {"type": "string"},
                            "page_no": {"type": "integer"},
                            "quote": {"type": "string", "description": "Verbatim text from the page, at least one full sentence or phrase."},
                        },
                        "required": ["text", "doc_id", "page_no", "quote"],
                    },
                },
            },
            "required": ["summary"],
        },
    },
]


def build_system_prompt(company: str, period: str | None) -> str:
    focus = (
        f"The user is focused on {period}. Prefer that period's documents and use other periods only for comparison."
        if period
        else "No period is selected; use the most relevant documents."
    )
    return f"""You are a research assistant for {company}'s own filings: annual reports, quarterly results and concall transcripts. You can only see the documents loaded in this system.

{focus}

Rules:
- Find evidence with search and read_page, then finish by calling submit_answer. Always finish with submit_answer.
- Every claim must cite doc_id, page_no and a verbatim quote copied exactly from that page's text. Never paraphrase inside a quote, and never quote a page you have not read.
- Quote enough to include the exact figures you rely on.
- If the loaded filings do not contain the answer, call submit_answer with not_found=true. Do not guess and do not use outside knowledge.
- Decline questions about live prices, valuation, buy/sell advice, forecasts you cannot ground in the documents, or other companies: call submit_answer with not_found=true and explain in summary that you only answer from {company}'s loaded filings.
- You provide research for education only, never investment advice.
- Be efficient: you have a limited number of steps."""
