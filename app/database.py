from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Iterator

from app.models import Employee


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(f"Database file does not exist: {self.path}")
        with self.connect() as connection:
            tables = {
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            required = {"employees", "leave_balances", "leave_requests"}
            if not required.issubset(tables):
                raise ValueError("Database is missing required assessment tables")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    employee_id TEXT NOT NULL,
                    proposal_name TEXT,
                    proposal_args_json TEXT
                )
                """
            )
            connection.commit()

    def get_employee(self, employee_id: str) -> Employee | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT employee_id, full_name, email, phone, region, join_date,
                       level, role, manager_id
                FROM employees WHERE employee_id = ?
                """,
                (employee_id,),
            ).fetchone()
        if row is None:
            return None
        return Employee(
            employee_id=row["employee_id"],
            full_name=row["full_name"],
            email=row["email"],
            phone=row["phone"],
            region=row["region"],
            join_date=date.fromisoformat(row["join_date"]),
            level=row["level"],
            role=row["role"],
            manager_id=row["manager_id"],
        )

    def get_join_date(self, employee_id: str) -> date:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT join_date FROM employees WHERE employee_id = ?",
                (employee_id,),
            ).fetchone()
        if row is None:
            raise ValueError("Employee record is unavailable")
        return date.fromisoformat(row["join_date"])

    def find_employee_id_by_name(self, message: str) -> str | None:
        normalized = message.casefold()
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT employee_id, full_name FROM employees"
            ).fetchall()
        matches = [
            row["employee_id"]
            for row in rows
            if row["full_name"].casefold() in normalized
        ]
        return matches[0] if len(matches) == 1 else None

    def redact_personal_data(self, text: str) -> str:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT full_name, email, phone FROM employees"
            ).fetchall()
        redacted = text
        for row in rows:
            for field in ("email", "phone", "full_name"):
                value = row[field]
                if value:
                    replacement = (
                        self._masked_email(value)
                        if field == "email"
                        else "[redacted]"
                    )
                    redacted = re.sub(
                        re.escape(value), replacement, redacted, flags=re.IGNORECASE
                    )
        redacted = re.sub(
            r"\b[A-Z0-9._%+-]+@([A-Z0-9.-]+\.[A-Z]{2,})\b",
            lambda match: f"{match.group(0)[0]}***@{match.group(1)}",
            redacted,
            flags=re.IGNORECASE,
        )
        redacted = re.sub(
            r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)",
            self._redact_phone_or_date,
            redacted,
        )
        return redacted

    @staticmethod
    def _masked_email(email: str) -> str:
        local, separator, domain = email.partition("@")
        return f"{local[:1]}***{separator}{domain}" if separator else "[redacted]"

    @staticmethod
    def _redact_phone_or_date(match: re.Match[str]) -> str:
        value = match.group(0)
        if re.fullmatch(r"20\d{2}-\d{2}-\d{2}", value):
            return value
        return "[redacted]"

    def bind_session(self, session_id: str, employee_id: str) -> bool:
        with self.connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO chat_sessions (session_id, employee_id) VALUES (?, ?)",
                (session_id, employee_id),
            )
            connection.commit()
            row = connection.execute(
                "SELECT employee_id FROM chat_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row is not None and row["employee_id"] == employee_id

    def get_proposal(self, session_id: str) -> tuple[str, dict[str, str]] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT proposal_name, proposal_args_json
                FROM chat_sessions WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if row is None or row["proposal_name"] is None:
            return None
        args = json.loads(row["proposal_args_json"] or "{}")
        if not isinstance(args, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in args.items()
        ):
            raise ValueError("Stored leave proposal has an invalid shape")
        return row["proposal_name"], args

    def set_proposal(
        self, session_id: str, name: str, args: dict[str, str]
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE chat_sessions
                SET proposal_name = ?, proposal_args_json = ?
                WHERE session_id = ?
                """,
                (name, json.dumps(args, separators=(",", ":")), session_id),
            )
            connection.commit()

    def clear_proposal(self, session_id: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE chat_sessions
                SET proposal_name = NULL, proposal_args_json = NULL
                WHERE session_id = ?
                """,
                (session_id,),
            )
            connection.commit()
