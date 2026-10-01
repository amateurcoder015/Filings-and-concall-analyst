from __future__ import annotations

import os

from backend import config
from backend.agent.agent import FilingsAgent
from backend.index.embedder import LocalEmbedder
from backend.index.store import PageIndex
from backend.manifest import load_manifest


def build_components():
    """Builds the real index and agent. Needs ANTHROPIC_API_KEY and an ingested index."""
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Export it before starting the server.")
    import anthropic

    company, docs = load_manifest(config.DATA_DIR)
    index = PageIndex(config.DB_PATH, LocalEmbedder())
    if not index.page_counts():
        raise RuntimeError("The index is empty. Run: python -m backend.ingest.cli")
    stale = index.doc_ids() - {d.doc_id for d in docs}
    if stale:
        raise RuntimeError(
            f"The index holds documents that are not in manifest.json ({', '.join(sorted(stale))}). "
            "Re-run: python -m backend.ingest.cli"
        )
    agent = FilingsAgent(
        anthropic.Anthropic(timeout=60.0, max_retries=0), index, model=config.MODEL, company=company
    )
    return company, docs, index, agent
