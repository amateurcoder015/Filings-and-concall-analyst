import threading

import pytest

from backend.index.embedder import HashingEmbedder
from backend.index.store import PageIndex
from backend.models import Page


def make_pages():
    return [
        Page("q2-results", 1, "results", "Q2 FY26", "Operating margin was 21.1% in the second quarter, down 40 basis points."),
        Page("q2-results", 2, "results", "Q2 FY26", "Revenue grew 3.1% in constant currency driven by financial services."),
        Page("q2-concall", 1, "concall", "Q2 FY26", "Management said wage hikes and visa costs pressured margins this quarter."),
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


def test_stemming_finds_plural_and_singular_forms(index):
    # Porter stemmer should match "margin" to "margins"
    margin_hits = index.search("margin", doc_type="concall")
    assert margin_hits and any(h.doc_id == "q2-concall" for h in margin_hits)
    # And "hike" should match "hikes" in the same page
    hike_hits = index.search("hike")
    assert hike_hits and any(h.doc_id == "q2-concall" for h in hike_hits)


def test_concurrent_search_and_ingest(tmp_path):
    """Test that search and ingest can safely run concurrently from multiple threads."""
    index = PageIndex(str(tmp_path / "test.db"), HashingEmbedder())
    index.add_pages(make_pages())

    exceptions = []

    def search_worker():
        try:
            for _ in range(50):
                index.search("margin")
                index.search("revenue")
                index.search("dividend")
        except Exception as e:
            exceptions.append(e)

    def ingest_worker():
        try:
            for _ in range(20):
                index.add_pages(make_pages())
        except Exception as e:
            exceptions.append(e)

    # Start 4 search threads and 1 ingest thread
    threads = []
    for _ in range(4):
        t = threading.Thread(target=search_worker)
        threads.append(t)
        t.start()

    ingest_t = threading.Thread(target=ingest_worker)
    threads.append(ingest_t)
    ingest_t.start()

    # Wait for all threads to finish
    for t in threads:
        t.join()

    # Check no exceptions occurred
    assert exceptions == [], f"Exceptions in threads: {exceptions}"

    # Verify final state
    assert index.page_counts() == {"q2-results": 2, "q2-concall": 1, "ar-fy25": 1}


def three_page_doc():
    return [
        Page("ar", 1, "annual_report", "FY25", "Chairman letter on strategy and talent."),
        Page("ar", 2, "annual_report", "FY25", "Segment revenue table for financial services."),
        Page("ar", 3, "annual_report", "FY25", "Obsolete zebra paragraph about legacy subsidiaries."),
    ]


def test_replace_document_drops_pages_missing_from_new_version(index):
    index.replace_document("ar", three_page_doc())
    index.replace_document("ar", three_page_doc()[:2])
    assert index.page_counts()["ar"] == 2
    assert index.get_page("ar", 3) is None
    assert all(not (h.doc_id == "ar" and h.page_no == 3) for h in index.search("zebra legacy subsidiaries"))
    assert index.fts_row_count() == sum(index.page_counts().values())
    # other documents are untouched
    assert index.page_counts()["q2-results"] == 2


def test_replace_document_keeps_fts_in_step_with_pages(index):
    for _ in range(3):
        index.replace_document("ar", three_page_doc())
    assert index.fts_row_count() == sum(index.page_counts().values()) == 7


def test_prune_documents_removes_docs_not_kept(index):
    index.prune_documents({"q2-results", "q2-concall"})
    assert set(index.page_counts()) == {"q2-results", "q2-concall"}
    assert index.get_page("ar-fy25", 1) is None
    assert index.search("dividend rupees per share", doc_type="annual_report") == []
    assert index.fts_row_count() == sum(index.page_counts().values())
