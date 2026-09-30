import pytest

from backend.index.embedder import HashingEmbedder
from backend.index.store import PageIndex
from backend.models import Page


def make_pages():
    return [
        Page("q2-results", 1, "results", "Q2 FY26", "Operating margin was 21.1% in the second quarter, down 40 basis points."),
        Page("q2-results", 2, "results", "Q2 FY26", "Revenue grew 3.1% in constant currency driven by financial services."),
        Page("q2-concall", 1, "concall", "Q2 FY26", "Management said wage hikes and visa costs pressured margin this quarter."),
        Page("ar-fy25", 1, "annual_report", "FY25", "The board recommended a final dividend of 22 rupees per share."),
    ]


@pytest.fixture()
def index():
    idx = PageIndex(":memory:", HashingEmbedder())
    idx.add_pages(make_pages())
    return idx


def test_keyword_match_ranks_relevant_page_first(index):
    hits = index.search("operating margin second quarter")
    assert hits[0].doc_id == "q2-results" and hits[0].page_no == 1


def test_doc_type_filter(index):
    hits = index.search("margin", doc_type="concall")
    assert hits and all(h.doc_type == "concall" for h in hits)


def test_period_filter(index):
    hits = index.search("dividend", period="FY25")
    assert hits and all(h.period == "FY25" for h in hits)
    assert index.search("dividend", period="Q2 FY26") == [] or all(
        h.period == "Q2 FY26" for h in index.search("dividend", period="Q2 FY26")
    )


def test_fts_syntax_in_query_never_raises(index):
    for nasty in ['margin" AND (* OR NEAR', "'; DROP TABLE pages; --", "a* -b ^c", '"', "()"]:
        assert isinstance(index.search(nasty), list)


def test_empty_query_returns_nothing(index):
    assert index.search("   ") == []


def test_get_page_returns_full_text_or_none(index):
    page = index.get_page("q2-results", 2)
    assert page is not None and "constant currency" in page.text
    assert index.get_page("q2-results", 99) is None
    assert index.get_page("nope", 1) is None


def test_reingesting_a_doc_replaces_pages_instead_of_duplicating(index):
    index.add_pages(make_pages())
    assert index.page_counts() == {"q2-results": 2, "q2-concall": 1, "ar-fy25": 1}


def test_hit_snippet_is_single_line_and_bounded(index):
    hit = index.search("margin")[0]
    assert "\n" not in hit.snippet and len(hit.snippet) <= 300
