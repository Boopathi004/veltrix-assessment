from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pymupdf
from docx import Document as DocxDocument

from app.ingestion import extract_public_holidays, ingest
from app.models import Document, ManifestEntry


def test_ingestion_extracts_all_formats_and_ocr(
    tmp_path: Path, monkeypatch
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    (corpus / "policy.md").write_text(
        "# Annual leave\nEmployees may carry leave forward.", encoding="utf-8"
    )
    (corpus / "faq.html").write_text(
        "<h1>FAQ</h1><p>Questions and answers.</p>"
        "<table><tr><th>Days</th><th>Type</th></tr>"
        "<tr><td>12</td><td>Annual</td></tr></table>",
        encoding="utf-8",
    )
    docx = DocxDocument()
    docx.add_heading("Benefits", level=1)
    docx.add_paragraph("Employees can use the learning benefit.")
    docx.add_table(rows=2, cols=2)
    docx.tables[0].cell(0, 0).text = "Course"
    docx.tables[0].cell(0, 1).text = "Limit"
    docx.tables[0].cell(1, 0).text = "Training"
    docx.tables[0].cell(1, 1).text = "12"
    docx.save(corpus / "benefits.docx")

    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Policy PDF content")
    pdf.save(corpus / "text.pdf")
    pdf.close()
    scanned = pymupdf.open()
    scanned.new_page()
    scanned.save(corpus / "scanned.pdf")
    scanned.close()
    monkeypatch.setattr(
        "app.ingestion.pytesseract.image_to_string",
        lambda image, lang: "OCR extracted text from scanned memo.",
    )

    names = ["policy.md", "faq.html", "benefits.docx", "text.pdf", "scanned.pdf"]
    manifest = [
        {
            "doc_id": f"DOC-{index}",
            "file": name,
            "title": name,
            "doc_type": "policy",
            "regions": ["GLOBAL"],
            "effective_date": "2026-01-01",
            "access_level": "all",
        }
        for index, name in enumerate(names)
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    documents = ingest(corpus, manifest_path)
    text_by_file = {document.metadata.file: document.text for document in documents}
    assert "Employees may carry leave forward." in text_by_file["policy.md"]
    assert "12 | Annual" in text_by_file["faq.html"]
    assert "Training | 12" in text_by_file["benefits.docx"]
    assert "Policy PDF content" in text_by_file["text.pdf"]
    assert "OCR extracted text" in text_by_file["scanned.pdf"]
    assert any(passage.locator == "Annual leave" for passage in documents[0].passages)


def test_manifest_rejects_path_traversal(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "doc_id": "BAD",
                    "file": "../secret.txt",
                    "title": "Bad path",
                    "doc_type": "policy",
                    "regions": ["GLOBAL"],
                    "effective_date": "2026-01-01",
                    "access_level": "all",
                }
            ]
        ),
        encoding="utf-8",
    )
    try:
        ingest(corpus, manifest_path)
    except ValueError as exc:
        assert "escapes corpus" in str(exc)
    else:
        raise AssertionError("A corpus path traversal was accepted")


def test_calendar_extraction_ignores_restricted_holidays() -> None:
    metadata = ManifestEntry(
        "CAL-TEST",
        "calendar.md",
        "Test calendar",
        "calendar",
        ("IN",),
        date(2026, 1, 1),
        "all",
    )
    text = (
        "Public holidays 2026\n1 January 2026 | New Year's Day\n"
        "2. Restricted holidays 2026\n4 March 2026 | Optional holiday"
    )
    document = Document(metadata, text, ())
    assert extract_public_holidays(document) == {date(2026, 1, 1)}
