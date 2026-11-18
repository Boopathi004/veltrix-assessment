from __future__ import annotations

import json
import re
from datetime import date, datetime
from io import BytesIO
from pathlib import Path

import pymupdf
import pytesseract
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph
from PIL import Image
from pypdf import PdfReader

from app.models import Document, ManifestEntry, Passage

MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
)


def load_manifest(path: Path, corpus_dir: Path) -> tuple[ManifestEntry, ...]:
    try:
        raw_entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read manifest at {path}") from exc
    if not isinstance(raw_entries, list):
        raise ValueError("The manifest must be a JSON array")

    entries: list[ManifestEntry] = []
    seen_ids: set[str] = set()
    seen_files: set[str] = set()
    for raw in raw_entries:
        try:
            entry = ManifestEntry(
                doc_id=str(raw["doc_id"]),
                file=str(raw["file"]),
                title=str(raw["title"]),
                doc_type=str(raw["doc_type"]).lower(),
                regions=tuple(str(region).upper() for region in raw["regions"]),
                effective_date=date.fromisoformat(raw["effective_date"]),
                access_level=str(raw["access_level"]).lower(),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Manifest entry is missing or has invalid metadata") from exc
        file_path = (corpus_dir / entry.file).resolve()
        if not file_path.is_relative_to(corpus_dir.resolve()):
            raise ValueError(f"Manifest path escapes corpus directory: {entry.file}")
        if not file_path.is_file():
            raise ValueError(f"Manifest document is missing: {entry.file}")
        if not entry.doc_id or entry.doc_id in seen_ids:
            raise ValueError("Manifest doc_id values must be non-empty and unique")
        if entry.file in seen_files:
            raise ValueError("Manifest file values must be unique")
        if entry.access_level not in {"all", "managers", "hr"}:
            raise ValueError(f"Unsupported access level for {entry.doc_id}")
        seen_ids.add(entry.doc_id)
        seen_files.add(entry.file)
        entries.append(entry)

    if not entries:
        raise ValueError("Manifest contains no documents")
    return tuple(entries)


def ingest(corpus_dir: Path, manifest_path: Path) -> tuple[Document, ...]:
    documents: list[Document] = []
    for metadata in load_manifest(manifest_path, corpus_dir):
        path = corpus_dir / metadata.file
        sections = _extract(path)
        passages = tuple(
            Passage(
                doc_id=metadata.doc_id,
                title=metadata.title,
                doc_type=metadata.doc_type,
                regions=metadata.regions,
                effective_date=metadata.effective_date,
                access_level=metadata.access_level,
                locator=locator,
                text=text,
            )
            for locator, text in sections
            if text.strip()
        )
        full_text = "\n\n".join(text for _, text in sections)
        if not passages:
            raise ValueError(f"No text could be extracted from {metadata.file}")
        documents.append(Document(metadata, full_text, passages))
    return tuple(documents)


def extract_public_holidays(document: Document) -> frozenset[date]:
    if document.metadata.doc_type != "calendar":
        return frozenset()
    holidays: set[date] = set()
    in_restricted_section = False
    skip_effective_date_value = False
    for line in document.text.splitlines():
        normalized = line.strip().lower()
        if re.match(r"^\d+(?:\.\d+)*\.\s+", normalized):
            in_restricted_section = "restricted" in normalized
        if "restricted holiday" in normalized:
            in_restricted_section = True
        if "effective date" in normalized:
            dates = _dates_on_line(line)
            if dates:
                skip_effective_date_value = False
                continue
            skip_effective_date_value = True
            continue
        if skip_effective_date_value:
            skip_effective_date_value = False
            if _dates_on_line(line):
                continue
        if not normalized:
            continue
        parsed = _dates_on_line(line)
        if not in_restricted_section:
            holidays.update(parsed)
    return frozenset(holidays)


def _extract(path: Path) -> list[tuple[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".docx":
        return _extract_docx(path)
    if suffix == ".html":
        return _extract_html(path)
    if suffix == ".md":
        return _extract_markdown(path)
    raise ValueError(f"Unsupported corpus format: {path.name}")


def _extract_pdf(path: Path) -> list[tuple[str, str]]:
    text_pages: list[str] = []
    pdf = PdfReader(path)
    for page_number, page in enumerate(pdf.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            with pymupdf.open(path) as rendered:
                pixmap = rendered[page_number - 1].get_pixmap(dpi=250)
                image = Image.open(BytesIO(pixmap.tobytes("png")))
                text = pytesseract.image_to_string(image, lang="eng").strip()
            if not text:
                raise ValueError(
                    f"OCR produced no text for {path.name}, page {page_number}"
                )
        text_pages.append(text)

    sections: list[tuple[str, str]] = []
    for page_number, text in enumerate(text_pages, start=1):
        sections.extend(_split_sections(text, f"p. {page_number}", detect_headings=True))
    return sections


def _extract_docx(path: Path) -> list[tuple[str, str]]:
    doc = DocxDocument(path)
    sections: list[tuple[str, str]] = []
    current_heading = "Document"
    content: list[str] = []

    def flush() -> None:
        nonlocal content
        if content:
            sections.extend(_chunk_text(current_heading, "\n".join(content)))
            content = []

    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            paragraph = Paragraph(child, doc)
            text = paragraph.text.strip()
            if not text:
                continue
            if (
                paragraph.style and paragraph.style.name.startswith("Heading")
            ) or re.match(r"^\d+(?:\.\d+)*\.\s+\S.{0,110}$", text):
                flush()
                current_heading = text
            else:
                content.append(text)
        elif child.tag.endswith("}tbl"):
            table = Table(child, doc)
            for row in table.rows:
                cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                content.append(" | ".join(cells))
    flush()
    return sections


def _extract_html(path: Path) -> list[tuple[str, str]]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    sections: list[tuple[str, str]] = []
    current_heading = "Document"
    content: list[str] = []

    def flush() -> None:
        nonlocal content
        if content:
            sections.extend(_chunk_text(current_heading, "\n".join(content)))
            content = []

    for element in soup.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr"]
    ):
        text = " | ".join(element.stripped_strings)
        if not text:
            continue
        if element.name and element.name.startswith("h"):
            flush()
            current_heading = text
        else:
            content.append(text)
    flush()
    return sections


def _extract_markdown(path: Path) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_heading = "Document"
    content: list[str] = []

    def flush() -> None:
        nonlocal content
        if content:
            sections.extend(_chunk_text(current_heading, "\n".join(content)))
            content = []

    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+)$", line)
        if match:
            flush()
            current_heading = match.group(1).strip()
        elif line.strip():
            content.append(line.strip())
    flush()
    return sections


def _split_sections(
    text: str, fallback_locator: str, detect_headings: bool
) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_heading = fallback_locator
    content: list[str] = []

    def flush() -> None:
        nonlocal content
        if content:
            sections.extend(_chunk_text(current_heading, "\n".join(content)))
            content = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        is_heading = detect_headings and (
            re.match(r"^\d+(?:\.\d+)*\.\s+\S.{0,110}$", stripped)
            or (
                len(stripped) <= 90
                and stripped.isupper()
                and len(stripped) > 4
                and "|" not in stripped
                and not re.fullmatch(r"[A-Z0-9][A-Z0-9-]*", stripped)
            )
        )
        if is_heading:
            flush()
            current_heading = stripped
        else:
            content.append(stripped)
    flush()
    return sections


def _chunk_text(locator: str, text: str, max_chars: int = 1400) -> list[tuple[str, str]]:
    paragraphs = [part.strip() for part in re.split(r"\n+", text) if part.strip()]
    chunks: list[tuple[str, str]] = []
    current = ""
    for paragraph in paragraphs:
        pieces = (
            [paragraph[index : index + max_chars] for index in range(0, len(paragraph), max_chars)]
            if len(paragraph) > max_chars
            else [paragraph]
        )
        for piece in pieces:
            if current and len(current) + len(piece) + 1 > max_chars:
                chunks.append((locator, current))
                current = ""
            current = f"{current}\n{piece}".strip()
    if current:
        chunks.append((locator, current))
    return chunks


def _dates_on_line(line: str) -> set[date]:
    result: set[date] = set()
    for match in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", line):
        result.add(date.fromisoformat(match.group(0)))
    for match in re.finditer(
        rf"\b(\d{{1,2}})\s+({MONTHS})\s+(\d{{4}})\b", line, re.IGNORECASE
    ):
        result.add(
            datetime.strptime(
                f"{match.group(1)} {match.group(2)} {match.group(3)}", "%d %B %Y"
            ).date()
        )
    for match in re.finditer(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", line):
        first, second, year = (int(value) for value in match.groups())
        month, day = (first, second)
        if first > 12:
            day, month = first, second
        result.add(date(year, month, day))
    return result
