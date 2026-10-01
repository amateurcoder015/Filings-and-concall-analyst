from __future__ import annotations

import argparse
from pathlib import Path

from backend import config
from backend.index.store import PageIndex
from backend.ingest.pdf import ingest_pdf
from backend.manifest import ManifestError, load_manifest


def run_ingest(directory: Path, index: PageIndex) -> dict[str, int]:
    _, docs = load_manifest(directory)
    counts: dict[str, int] = {}
    for meta in docs:
        try:
            pages = ingest_pdf(Path(directory) / meta.file, meta)
        except Exception as exc:
            raise ManifestError(f"cannot read {meta.file}: {exc}") from exc
        index.replace_document(meta.doc_id, pages)
        counts[meta.doc_id] = len(pages)
        flagged = sum(1 for p in pages if p.low_confidence)
        note = f" ({flagged} low-confidence)" if flagged else ""
        print(f"  {meta.doc_id}: {len(pages)} pages{note}")
    # Documents no longer in the manifest (removed or renamed files) must not stay searchable.
    index.prune_documents({meta.doc_id for meta in docs})
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse filings and build the page index.")
    parser.add_argument("directory", nargs="?", default=str(config.DATA_DIR))
    args = parser.parse_args(argv)

    try:
        # Load manifest first to fail fast on bad manifests before loading embedder
        load_manifest(Path(args.directory))

        from backend.index.embedder import LocalEmbedder

        index = PageIndex(config.DB_PATH, LocalEmbedder())
        print(f"Ingesting {args.directory}")
        counts = run_ingest(Path(args.directory), index)
    except ManifestError as exc:
        print(f"Error: {exc}")
        return 1
    print(f"Done: {sum(counts.values())} pages indexed into {config.DB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
