import pytest
from fastapi.testclient import TestClient

from backend.agent.agent import AgentError
from backend.api.main import create_app
from backend.models import Answer, Claim, DocMeta
from tests.conftest import MARGIN_PAGE

DOCS = [DocMeta("q2-results", "q2-results.pdf", "results", "Q2 FY26", "Q2 FY26 Results")]


class FakeAgent:
    def __init__(self, answer=None, error=None):
        self.answer, self.error, self.calls = answer, error, []

    def ask(self, question, period=None):
        self.calls.append((question, period))
        if self.error:
            raise self.error
        return self.answer


@pytest.fixture()
def pdf_dir(tmp_path):
    (tmp_path / "q2-results.pdf").write_bytes(b"%PDF-1.4 fake")
    return tmp_path


def client_for(index, pdf_dir, agent):
    return TestClient(create_app(index, agent, DOCS, "Infosys", pdf_dir))


def good_answer():
    return Answer(
        summary="Margin was 21.1%.",
        claims=[Claim("Margin was 21.1%.", "q2-results", 1, MARGIN_PAGE)],
    )


def test_health(index, pdf_dir):
    assert client_for(index, pdf_dir, FakeAgent()).get("/health").json() == {"status": "ok"}


def test_documents_lists_company_docs_with_page_counts(index, pdf_dir):
    body = client_for(index, pdf_dir, FakeAgent()).get("/documents").json()
    assert body["company"] == "Infosys"
    assert body["documents"] == [
        {"doc_id": "q2-results", "title": "Q2 FY26 Results", "doc_type": "results", "period": "Q2 FY26", "pages": 2}
    ]


def test_ask_returns_verified_claims_and_disclaimer(index, pdf_dir):
    agent = FakeAgent(answer=good_answer())
    body = client_for(index, pdf_dir, agent).post("/ask", json={"question": "Margin?", "period": "Q2 FY26"}).json()
    assert agent.calls == [("Margin?", "Q2 FY26")]
    assert body["claims"][0]["status"] == "verified"
    assert body["claims"][0]["page_no"] == 1
    assert body["mostly_unverified"] is False
    assert "investment advice" in body["disclaimer"]


def test_fabricated_quote_is_returned_marked_failed_never_verified(index, pdf_dir):
    answer = Answer(summary="s", claims=[Claim("x", "q2-results", 1, "The company announced a giant buyback programme")])
    body = client_for(index, pdf_dir, FakeAgent(answer=answer)).post("/ask", json={"question": "q"}).json()
    assert body["claims"][0]["status"] == "failed"
    assert body["mostly_unverified"] is True


def test_not_found_answer(index, pdf_dir):
    answer = Answer(summary="This is not in the loaded filings.", claims=[], not_found=True)
    body = client_for(index, pdf_dir, FakeAgent(answer=answer)).post("/ask", json={"question": "price?"}).json()
    assert body["not_found"] is True and body["claims"] == []


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_empty_question_is_rejected_without_calling_the_model(index, pdf_dir, question):
    agent = FakeAgent(answer=good_answer())
    response = client_for(index, pdf_dir, agent).post("/ask", json={"question": question})
    assert response.status_code == 422
    assert agent.calls == []


def test_overlong_question_is_rejected(index, pdf_dir):
    agent = FakeAgent(answer=good_answer())
    response = client_for(index, pdf_dir, agent).post("/ask", json={"question": "x" * 1001})
    assert response.status_code == 422 and agent.calls == []


def test_agent_failure_becomes_502(index, pdf_dir):
    agent = FakeAgent(error=AgentError("Claude API call failed: boom"))
    response = client_for(index, pdf_dir, agent).post("/ask", json={"question": "q"})
    assert response.status_code == 502
    assert "boom" in response.json()["detail"]


def test_page_endpoint(index, pdf_dir):
    c = client_for(index, pdf_dir, FakeAgent())
    assert "Operating margin" in c.get("/page/q2-results/1").json()["text"]
    assert c.get("/page/q2-results/99").status_code == 404


def test_pdf_endpoint_serves_the_file_and_404s_unknown_docs(index, pdf_dir):
    c = client_for(index, pdf_dir, FakeAgent())
    response = c.get("/pdf/q2-results")
    assert response.status_code == 200 and response.headers["content-type"] == "application/pdf"
    assert c.get("/pdf/ghost").status_code == 404
    assert c.get("/pdf/..%2Fmanifest").status_code == 404
