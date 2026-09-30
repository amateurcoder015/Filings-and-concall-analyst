import pytest

from backend.index.embedder import HashingEmbedder
from backend.index.store import PageIndex
from backend.models import Page

MARGIN_PAGE = "Operating margin was 21.1% in the second quarter, down 40 basis points from the prior quarter."


@pytest.fixture()
def index():
    idx = PageIndex(":memory:", HashingEmbedder())
    idx.add_pages(
        [
            Page("q2-results", 1, "results", "Q2 FY26", MARGIN_PAGE),
            Page("q2-results", 2, "results", "Q2 FY26", "Revenue grew 3.1% in constant currency driven by financial services."),
            Page("q2-concall", 1, "concall", "Q2 FY26", "Management said wage hikes and visa costs pressured margins this quarter.", True),
        ]
    )
    return idx
