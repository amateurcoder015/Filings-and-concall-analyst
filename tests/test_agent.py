import pytest

from backend.agent.agent import MAX_STEPS, NOT_FOUND_TEXT, AgentError, FilingsAgent
from tests.conftest import MARGIN_PAGE
from tests.fakes import ScriptedClient, response, text, tool_use

GOOD_CLAIM = {"text": "Margin was 21.1%.", "doc_id": "q2-results", "page_no": 1, "quote": MARGIN_PAGE}


def submit(claims=None, not_found=False, summary="Summary.", id="s1"):
    return response(tool_use("submit_answer", {"summary": summary, "claims": claims or [], "not_found": not_found}, id=id))


def make_agent(index, script):
    client = ScriptedClient(script)
    return FilingsAgent(client, index, model="test-model"), client


def test_search_then_read_then_submit_returns_answer(index):
    agent, client = make_agent(
        index,
        [
            response(tool_use("search", {"query": "operating margin"}, id="a")),
            response(tool_use("read_page", {"doc_id": "q2-results", "page_no": 1}, id="b")),
            submit([GOOD_CLAIM]),
        ],
    )
    answer = agent.ask("What was operating margin?")
    assert answer.not_found is False
    assert answer.claims[0].doc_id == "q2-results" and answer.claims[0].page_no == 1
    assert answer.claims[0].quote == MARGIN_PAGE
    # the search result was fed back to the model as a tool_result
    second_call_messages = client.calls[1]["messages"]
    tool_result = second_call_messages[-1]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "a"
    assert "q2-results p.1" in tool_result["content"]


def test_claim_without_quote_is_rejected_once_then_accepted(index):
    bad = dict(GOOD_CLAIM, quote="")
    agent, client = make_agent(index, [submit([bad], id="s1"), submit([GOOD_CLAIM], id="s2")])
    answer = agent.ask("q")
    assert answer.claims[0].quote == MARGIN_PAGE
    rejection = client.calls[1]["messages"][-1]["content"][0]
    assert rejection["is_error"] is True and rejection["tool_use_id"] == "s1"


def test_two_bad_submissions_fall_back_to_not_found(index):
    bad = dict(GOOD_CLAIM, quote="")
    agent, _ = make_agent(index, [submit([bad], id="s1"), submit([bad], id="s2")])
    answer = agent.ask("q")
    assert answer.not_found is True and answer.claims == []


def test_not_found_submission_passes_through(index):
    agent, _ = make_agent(index, [submit(not_found=True, summary="")])
    answer = agent.ask("What is the share price today?")
    assert answer.not_found is True
    assert answer.summary == NOT_FOUND_TEXT
    assert answer.claims == []


def test_answer_without_claims_and_not_found_false_is_rejected(index):
    agent, client = make_agent(index, [submit([], id="s1"), submit([GOOD_CLAIM], id="s2")])
    answer = agent.ask("q")
    assert len(answer.claims) == 1
    assert client.calls[1]["messages"][-1]["content"][0]["is_error"] is True


def test_plain_text_reply_is_nudged_to_submit(index):
    agent, client = make_agent(index, [response(text("Margin was 21%.")), submit([GOOD_CLAIM])])
    answer = agent.ask("q")
    assert len(answer.claims) == 1
    nudge = client.calls[1]["messages"][-1]
    assert nudge["role"] == "user" and "submit_answer" in nudge["content"]


def test_step_cap_stops_a_model_that_never_submits(index):
    script = [response(tool_use("search", {"query": "margin"}, id=f"s{i}")) for i in range(MAX_STEPS + 3)]
    agent, client = make_agent(index, script)
    answer = agent.ask("q")
    assert answer.not_found is True
    assert len(client.calls) == MAX_STEPS


def test_unknown_page_and_unknown_tool_return_errors_not_exceptions(index):
    agent, client = make_agent(
        index,
        [
            response(
                tool_use("read_page", {"doc_id": "ghost", "page_no": 9}, id="a"),
                tool_use("teleport", {}, id="b"),
            ),
            submit([GOOD_CLAIM]),
        ],
    )
    agent.ask("q")
    results = client.calls[1]["messages"][-1]["content"]
    assert all(r["is_error"] for r in results)


def test_low_confidence_page_is_flagged_to_the_model(index):
    agent, client = make_agent(
        index,
        [response(tool_use("read_page", {"doc_id": "q2-concall", "page_no": 1}, id="a")), submit(not_found=True)],
    )
    agent.ask("q")
    content = client.calls[1]["messages"][-1]["content"][0]["content"]
    assert "low-confidence" in content


def test_api_failure_is_retried_once(index):
    agent, _ = make_agent(index, [RuntimeError("boom"), submit([GOOD_CLAIM])])
    assert len(agent.ask("q").claims) == 1


def test_api_failing_twice_raises_agent_error(index):
    agent, _ = make_agent(index, [RuntimeError("boom"), RuntimeError("boom again")])
    with pytest.raises(AgentError):
        agent.ask("q")


def test_period_and_company_are_in_the_system_prompt(index):
    agent, client = make_agent(index, [submit(not_found=True)])
    agent.ask("q", period="Q2 FY26")
    system = client.calls[0]["system"]
    assert "Infosys" in system and "Q2 FY26" in system
    assert client.calls[0]["model"] == "test-model"


CUT_OFF = "Your last reply was cut off. Be brief: use at most 3 claims with short verbatim quotes (one sentence each), then call submit_answer."


def truncated(*blocks):
    r = response(*blocks)
    r.stop_reason = "max_tokens"
    return r


def test_max_tokens_constant_is_4096(index):
    agent, client = make_agent(index, [submit(not_found=True)])
    agent.ask("q")
    assert client.calls[0]["max_tokens"] == 4096


def test_truncated_reply_is_dropped_and_model_is_asked_to_be_brief(index):
    partial = truncated(tool_use("submit_answer", {"summary": "cut", "claims": [{"text": "x"}]}, id="p"))
    agent, client = make_agent(index, [partial, submit([GOOD_CLAIM])])
    answer = agent.ask("q")
    assert len(answer.claims) == 1
    msgs = client.calls[1]["messages"]
    assert msgs[-1] == {"role": "user", "content": CUT_OFF}
    assert all(m["role"] != "assistant" for m in msgs)


def test_repeated_truncation_hits_step_cap(index):
    script = [truncated(text("...")) for _ in range(MAX_STEPS + 2)]
    agent, client = make_agent(index, script)
    answer = agent.ask("q")
    assert answer.not_found is True
    assert len(client.calls) == MAX_STEPS


def test_read_page_output_is_wrapped_and_prompt_marks_content_untrusted(index):
    agent, client = make_agent(
        index, [response(tool_use("read_page", {"doc_id": "q2-results", "page_no": 1}, id="a")), submit(not_found=True)]
    )
    agent.ask("q")
    content = client.calls[1]["messages"][-1]["content"][0]["content"]
    assert content.startswith("<page ") and content.rstrip().endswith("</page>")
    assert MARGIN_PAGE in content
    assert "untrusted" in client.calls[0]["system"]


def test_empty_assistant_content_is_not_appended(index):
    agent, client = make_agent(index, [response(), submit([GOOD_CLAIM])])
    assert len(agent.ask("q").claims) == 1
    msgs = client.calls[1]["messages"]
    assert all(m["role"] != "assistant" for m in msgs)
    assert msgs[-1] == {"role": "user", "content": "Finish by calling the submit_answer tool with your answer."}


def test_non_string_search_filters_return_error(index):
    agent, client = make_agent(
        index,
        [response(tool_use("search", {"query": "margin", "period": 5, "doc_type": ["x"]}, id="a")), submit(not_found=True)],
    )
    agent.ask("q")
    r = client.calls[1]["messages"][-1]["content"][0]
    assert r["is_error"] is True and "must be strings" in r["content"]


def test_non_dict_tool_args_return_error(index):
    agent, client = make_agent(
        index,
        [
            response(
                tool_use("search", "margin", id="a"),
                tool_use("read_page", None, id="b"),
            ),
            submit(not_found=True),
        ],
    )
    agent.ask("q")
    results = client.calls[1]["messages"][-1]["content"]
    assert len(results) == 2 and all(r["is_error"] for r in results)
