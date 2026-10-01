from __future__ import annotations

import json
from pathlib import Path

from backend.models import DOC_TYPES, DocMeta

REQUIRED_FIELDS = ("file", "doc_type", "period", "title")


class ManifestError(Exception):
    pass


def load_manifest(directory: Path) -> tuple[str, list[DocMeta]]:
    directory = Path(directory)
    path = directory / "manifest.json"
    if not path.exists():
        raise ManifestError(f"manifest.json not found in {directory}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest.json is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ManifestError("manifest.json must be a JSON object with 'company' and 'documents'")

    company = data.get("company")
    if not company:
        raise ManifestError("manifest.json needs a 'company' field")

    docs: list[DocMeta] = []
    seen: set[str] = set()
    entries = data.get("documents", [])
    if not isinstance(entries, list):
        raise ManifestError("manifest.json 'documents' must be a list of document entries")
    for position, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ManifestError(f"document entry {position} must be an object, got: {entry!r}")
        for key in REQUIRED_FIELDS:
            if key not in entry:
                raise ManifestError(f"document entry is missing '{key}': {entry}")
        if entry["doc_type"] not in DOC_TYPES:
            raise ManifestError(
                f"doc_type '{entry['doc_type']}' is not one of {list(DOC_TYPES)}"
            )
        if not (directory / entry["file"]).exists():
            raise ManifestError(f"listed file not found: {entry['file']}")
        doc_id = Path(entry["file"]).stem
        if doc_id in seen:
            raise ManifestError(f"duplicate document id '{doc_id}'")
        seen.add(doc_id)
        docs.append(
            DocMeta(
                doc_id=doc_id,
                file=entry["file"],
                doc_type=entry["doc_type"],
                period=entry["period"],
                title=entry["title"],
            )
        )
    if not docs:
        raise ManifestError("manifest.json lists no documents")
    return company, docs
