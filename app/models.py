from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class ManifestEntry:
    doc_id: str
    file: str
    title: str
    doc_type: str
    regions: tuple[str, ...]
    effective_date: date
    access_level: str


@dataclass(frozen=True)
class Document:
    metadata: ManifestEntry
    text: str
    passages: tuple[Passage, ...]


@dataclass(frozen=True)
class Passage:
    doc_id: str
    title: str
    doc_type: str
    regions: tuple[str, ...]
    effective_date: date
    access_level: str
    locator: str
    text: str


@dataclass(frozen=True)
class Employee:
    employee_id: str
    full_name: str
    email: str
    phone: str
    region: str
    join_date: date
    level: str
    role: str
    manager_id: str | None


@dataclass(frozen=True)
class Citation:
    doc_id: str
    locator: str


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, str]
    status: str


@dataclass(frozen=True)
class ChatResponse:
    request_id: str
    session_id: str
    message: str
    route: str
    citations: tuple[Citation, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    pending_confirmation: bool = False


@dataclass(frozen=True)
class RetrievedPassage:
    passage: Passage
    score: float
    upcoming: bool
