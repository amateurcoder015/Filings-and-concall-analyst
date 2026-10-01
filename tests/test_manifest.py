import json

import pytest

from backend.manifest import ManifestError, load_manifest


def write_manifest(directory, documents, company="Infosys"):
    (directory / "manifest.json").write_text(
        json.dumps({"company": company, "documents": documents})
    )


def entry(file="ar-fy25.pdf", doc_type="annual_report", period="FY25", title="Annual Report FY25"):
    return {"file": file, "doc_type": doc_type, "period": period, "title": title}


def test_loads_valid_manifest(tmp_path):
    (tmp_path / "ar-fy25.pdf").write_bytes(b"%PDF")
    write_manifest(tmp_path, [entry()])
    company, docs = load_manifest(tmp_path)
    assert company == "Infosys"
    assert docs[0].doc_id == "ar-fy25"
    assert docs[0].doc_type == "annual_report"
    assert docs[0].period == "FY25"


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(ManifestError, match="manifest.json not found"):
        load_manifest(tmp_path)


def test_listed_file_missing_names_the_file(tmp_path):
    write_manifest(tmp_path, [entry(file="nope.pdf")])
    with pytest.raises(ManifestError, match="nope.pdf"):
        load_manifest(tmp_path)


def test_unknown_doc_type_rejected(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    write_manifest(tmp_path, [entry(file="a.pdf", doc_type="tweet")])
    with pytest.raises(ManifestError, match="doc_type"):
        load_manifest(tmp_path)


def test_missing_field_rejected(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    bad = entry(file="a.pdf")
    del bad["period"]
    write_manifest(tmp_path, [bad])
    with pytest.raises(ManifestError, match="period"):
        load_manifest(tmp_path)


def test_duplicate_doc_ids_rejected(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    write_manifest(tmp_path, [entry(file="a.pdf"), entry(file="a.pdf")])
    with pytest.raises(ManifestError, match="duplicate"):
        load_manifest(tmp_path)


def test_invalid_json_rejected(tmp_path):
    (tmp_path / "manifest.json").write_text("{not json")
    with pytest.raises(ManifestError, match="not valid JSON"):
        load_manifest(tmp_path)


@pytest.mark.parametrize("payload", ["[]", "5", '"text"', "null"])
def test_non_object_manifest_rejected(tmp_path, payload):
    (tmp_path / "manifest.json").write_text(payload)
    with pytest.raises(ManifestError, match="must be a JSON object"):
        load_manifest(tmp_path)


@pytest.mark.parametrize("documents", [{"file": "a.pdf"}, "a.pdf", 5])
def test_non_list_documents_rejected(tmp_path, documents):
    (tmp_path / "manifest.json").write_text(json.dumps({"company": "Infosys", "documents": documents}))
    with pytest.raises(ManifestError, match="'documents' must be a list"):
        load_manifest(tmp_path)


@pytest.mark.parametrize("item", ["a.pdf", 5, None, ["a.pdf"]])
def test_non_object_document_entry_rejected(tmp_path, item):
    (tmp_path / "manifest.json").write_text(json.dumps({"company": "Infosys", "documents": [item]}))
    with pytest.raises(ManifestError, match="document entry 1 must be an object"):
        load_manifest(tmp_path)
