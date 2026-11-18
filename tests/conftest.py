from __future__ import annotations

import re
import sqlite3
from datetime import date
from pathlib import Path

import pytest

from app.config import Settings
from app.database import Database
from app.llm import ModelAnswer
from app.models import Citation, Document, Employee, ManifestEntry, Passage
from app.tools import LeaveTools


class FakeAnswerModel:
    def __init__(
        self,
        answer: str = "Evidence-backed answer.",
        citation: Citation | None = None,
    ) -> None:
        self.text = answer
        self.citation = citation
        self.calls: list[tuple[str, str, int, str]] = []

    def answer(
        self, question: str, context: str, corpus_tokens: int, tool_data: str = ""
    ) -> ModelAnswer:
        self.calls.append((question, context, corpus_tokens, tool_data))
        citation = self.citation
        if citation is None:
            match = re.search(
                r"\[doc_id=([^;]+);.*?locator=([^\]]+)\]", context, re.DOTALL
            )
            if match:
                citation = Citation(match.group(1), match.group(2))
        return ModelAnswer(self.text, (citation,) if citation else ())


def create_database(path: Path) -> Database:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE employees (
            employee_id TEXT PRIMARY KEY,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT NOT NULL,
            region TEXT NOT NULL,
            join_date TEXT NOT NULL,
            level TEXT NOT NULL,
            role TEXT NOT NULL,
            manager_id TEXT REFERENCES employees(employee_id)
        );
        CREATE TABLE leave_balances (
            employee_id TEXT NOT NULL REFERENCES employees(employee_id),
            leave_type TEXT NOT NULL,
            balance_days REAL NOT NULL,
            PRIMARY KEY (employee_id, leave_type)
        );
        CREATE TABLE leave_requests (
            request_id TEXT PRIMARY KEY,
            employee_id TEXT NOT NULL REFERENCES employees(employee_id),
            leave_type TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            working_days INTEGER NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        INSERT INTO employees VALUES
            ('E1001', 'Avery Employee', 'avery@example.test', '+1 555 010 1001',
             'IN', '2020-06-15', 'L3', 'employee', NULL),
            ('E2001', 'Morgan Manager', 'morgan@example.test', '+1 555 010 2001',
             'IN', '2018-01-01', 'L5', 'manager', NULL),
            ('E2002', 'Jordan Report', 'jordan@example.test', '+1 555 010 2002',
             'IN', '2022-01-01', 'L3', 'employee', 'E2001'),
            ('E2003', 'Casey Other', 'casey@example.test', '+1 555 010 2003',
             'IN', '2022-01-01', 'L3', 'employee', NULL),
            ('E9001', 'Taylor HR', 'taylor@example.test', '+1 555 010 9001',
             'IN', '2017-01-01', 'L6', 'hr', NULL),
            ('E3001', 'Robin Germany', 'robin@example.test', '+49 30 555 0101',
             'DE', '2021-01-01', 'L3', 'employee', NULL);
        INSERT INTO leave_balances VALUES
            ('E1001', 'annual', 5), ('E1001', 'sick', 8),
            ('E2001', 'annual', 10), ('E2001', 'sick', 8),
            ('E2002', 'annual', 10), ('E2002', 'sick', 8),
            ('E2003', 'annual', 10), ('E2003', 'sick', 8),
            ('E9001', 'annual', 10), ('E9001', 'sick', 8),
            ('E3001', 'annual', 10), ('E3001', 'sick', 8);
        """
    )
    connection.commit()
    connection.close()
    database = Database(path)
    database.initialize()
    return database


def make_document(
    doc_id: str,
    text: str,
    *,
    doc_type: str = "policy",
    regions: tuple[str, ...] = ("GLOBAL",),
    access_level: str = "all",
    effective_date: date = date(2026, 1, 1),
    locator: str = "Section 1",
) -> Document:
    metadata = ManifestEntry(
        doc_id=doc_id,
        file=f"{doc_id}.md",
        title=doc_id,
        doc_type=doc_type,
        regions=regions,
        effective_date=effective_date,
        access_level=access_level,
    )
    passage = Passage(
        doc_id=doc_id,
        title=metadata.title,
        doc_type=doc_type,
        regions=regions,
        effective_date=effective_date,
        access_level=access_level,
        locator=locator,
        text=text,
    )
    return Document(metadata, text, (passage,))


def make_calendar(region: str = "IN", dates: str = "1 January 2026") -> Document:
    return make_document(
        f"CAL-{region}",
        f"Public holidays 2026\n{dates}",
        doc_type="calendar",
        regions=(region,),
        locator="Public holidays",
    )


def make_settings(
    root: Path,
    *,
    today: date = date(2026, 1, 1),
    token_limit: int = 3000,
) -> Settings:
    return Settings(
        corpus_dir=root / "corpus",
        manifest_path=root / "manifest.json",
        db_path=root / "veltrix.db",
        today=today,
        llm_provider="openai",
        openai_api_key="",
        openai_model="gpt-4o-mini",
        log_level="INFO",
        max_context_tokens=token_limit,
        retrieval_top_k=8,
    )


@pytest.fixture
def database(tmp_path: Path) -> Database:
    return create_database(tmp_path / "test.db")


@pytest.fixture
def employee(database: Database) -> Employee:
    result = database.get_employee("E1001")
    assert result is not None
    return result


@pytest.fixture
def calendar_document() -> Document:
    return make_calendar()


@pytest.fixture
def leave_tools(database: Database, calendar_document: Document) -> LeaveTools:
    return LeaveTools(database, (calendar_document,), date(2026, 1, 1))
