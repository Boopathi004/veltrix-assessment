from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any
from uuid import uuid4

from app.database import Database
from app.ingestion import extract_public_holidays
from app.models import Document, Employee

REFUSAL = "I can’t share information about that employee."
VALID_LEAVE_TYPES = {"annual", "sick"}


class ToolError(Exception):
    pass


@dataclass(frozen=True)
class ToolResult:
    message: str
    data: dict[str, Any]


class LeaveTools:
    def __init__(
        self, database: Database, documents: tuple[Document, ...], today: date
    ) -> None:
        self.database = database
        self.documents = documents
        self.today = today
        self._holidays = {
            document.metadata.doc_id: extract_public_holidays(document)
            for document in documents
            if document.metadata.doc_type == "calendar"
        }

    def get_leave_balance(
        self, requester: Employee, employee_id: str, leave_type: str | None = None
    ) -> ToolResult:
        target = self._authorized_target(requester, employee_id)
        if leave_type is not None and leave_type not in VALID_LEAVE_TYPES:
            raise ToolError("leave_type must be annual or sick.")
        with self.database.connect() as connection:
            if leave_type:
                rows = connection.execute(
                    """
                    SELECT leave_type, balance_days FROM leave_balances
                    WHERE employee_id = ? AND leave_type = ?
                    """,
                    (target.employee_id, leave_type),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT leave_type, balance_days FROM leave_balances
                    WHERE employee_id = ? ORDER BY leave_type
                    """,
                    (target.employee_id,),
                ).fetchall()
        if not rows:
            raise ToolError("No leave balance is recorded for that employee.")
        balances = {row["leave_type"]: row["balance_days"] for row in rows}
        details = "; ".join(
            f"{kind}: {self._format_days(days)} days" for kind, days in balances.items()
        )
        return ToolResult(
            f"Current leave balance for {target.employee_id}: {details}.",
            {"employee_id": target.employee_id, "balances": balances},
        )

    def list_leave_requests(
        self, requester: Employee, employee_id: str, status: str | None = None
    ) -> ToolResult:
        target = self._authorized_target(requester, employee_id)
        if status is not None and status not in {"pending", "approved", "cancelled"}:
            raise ToolError("status must be pending, approved, or cancelled.")
        with self.database.connect() as connection:
            query = """
                SELECT request_id, leave_type, start_date, end_date, working_days, status
                FROM leave_requests WHERE employee_id = ?
            """
            args: list[str] = [target.employee_id]
            if status:
                query += " AND status = ?"
                args.append(status)
            query += " ORDER BY start_date, request_id"
            rows = connection.execute(query, args).fetchall()
        if not rows:
            return ToolResult(
                f"No leave requests found for {target.employee_id}.",
                {"employee_id": target.employee_id, "requests": []},
            )
        requests = [dict(row) for row in rows]
        rendered = "; ".join(
            f"{row['request_id']} ({row['leave_type']}, {row['start_date']} to "
            f"{row['end_date']}, {row['working_days']} working days, {row['status']})"
            for row in requests
        )
        return ToolResult(
            f"Leave requests for {target.employee_id}: {rendered}.",
            {"employee_id": target.employee_id, "requests": requests},
        )

    def get_team_leave(
        self, requester: Employee, start_date: str, end_date: str
    ) -> ToolResult:
        if requester.role != "manager":
            raise ToolError(REFUSAL)
        start, end = self._parse_range(start_date, end_date)
        if start.year != 2026 or end.year != 2026:
            raise ToolError("Team leave lookups are limited to 2026 dates.")
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT lr.employee_id, lr.leave_type, lr.start_date, lr.end_date,
                       lr.working_days, lr.status
                FROM leave_requests AS lr
                JOIN employees AS e ON e.employee_id = lr.employee_id
                WHERE e.manager_id = ?
                  AND lr.status IN ('pending', 'approved')
                  AND lr.start_date <= ? AND lr.end_date >= ?
                ORDER BY lr.start_date, lr.employee_id
                """,
                (requester.employee_id, end.isoformat(), start.isoformat()),
            ).fetchall()
        leave = [dict(row) for row in rows]
        if not leave:
            return ToolResult(
                "No pending or approved direct-report leave overlaps that range.",
                {"start_date": start.isoformat(), "end_date": end.isoformat(), "leave": []},
            )
        rendered = "; ".join(
            f"{row['employee_id']} ({row['leave_type']}, {row['start_date']} to "
            f"{row['end_date']}, {row['status']})"
            for row in leave
        )
        return ToolResult(
            f"Direct-report leave overlapping {start} through {end}: {rendered}.",
            {"start_date": start.isoformat(), "end_date": end.isoformat(), "leave": leave},
        )

    def validate_apply(
        self, requester: Employee, args: dict[str, str]
    ) -> tuple[date, date, int]:
        leave_type = args.get("leave_type", "")
        if leave_type not in VALID_LEAVE_TYPES:
            raise ToolError("leave_type must be annual or sick.")
        start, end = self._parse_range(args.get("start_date", ""), args.get("end_date", ""))
        if start.year != 2026 or end.year != 2026:
            raise ToolError("Leave requests must be within the 2026 calendar year.")
        if start < self.today:
            raise ToolError("The start_date must be today or a future date.")
        holidays = self._holidays_for(requester.region, start.year)
        working_days = self._working_days(start, end, holidays)
        if working_days == 0:
            raise ToolError("The requested range contains zero working days.")
        with self.database.connect() as connection:
            balance_row = connection.execute(
                """
                SELECT balance_days FROM leave_balances
                WHERE employee_id = ? AND leave_type = ?
                """,
                (requester.employee_id, leave_type),
            ).fetchone()
            if balance_row is None:
                raise ToolError("No balance is recorded for that leave type.")
            if working_days > balance_row["balance_days"]:
                raise ToolError(
                    f"The request needs {working_days} working days, but the current "
                    f"{leave_type} balance is {self._format_days(balance_row['balance_days'])} days."
                )
            overlap = connection.execute(
                """
                SELECT 1 FROM leave_requests
                WHERE employee_id = ? AND status IN ('pending', 'approved')
                  AND start_date <= ? AND end_date >= ?
                LIMIT 1
                """,
                (requester.employee_id, end.isoformat(), start.isoformat()),
            ).fetchone()
            if overlap:
                raise ToolError(
                    "The requested dates overlap an existing pending or approved leave request."
                )
        return start, end, working_days

    def apply_leave(
        self,
        requester: Employee,
        leave_type: str,
        start_date: str,
        end_date: str,
    ) -> ToolResult:
        args = {
            "leave_type": leave_type,
            "start_date": start_date,
            "end_date": end_date,
        }
        start, end, working_days = self.validate_apply(requester, args)
        request_id = str(uuid4())
        try:
            with self.database.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                balance_row = connection.execute(
                    """
                    SELECT balance_days FROM leave_balances
                    WHERE employee_id = ? AND leave_type = ?
                    """,
                    (requester.employee_id, leave_type),
                ).fetchone()
                if balance_row is None or working_days > balance_row["balance_days"]:
                    raise ToolError(
                        "The current leave balance changed; the request cannot be applied."
                    )
                overlap = connection.execute(
                    """
                    SELECT 1 FROM leave_requests
                    WHERE employee_id = ? AND status IN ('pending', 'approved')
                      AND start_date <= ? AND end_date >= ?
                    LIMIT 1
                    """,
                    (requester.employee_id, end.isoformat(), start.isoformat()),
                ).fetchone()
                if overlap:
                    raise ToolError(
                        "The requested dates now overlap an existing pending or approved leave request."
                    )
                connection.execute(
                    """
                    INSERT INTO leave_requests
                        (request_id, employee_id, leave_type, start_date, end_date,
                         working_days, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
                    """,
                    (
                        request_id,
                        requester.employee_id,
                        leave_type,
                        start.isoformat(),
                        end.isoformat(),
                        working_days,
                        self._now_timestamp(),
                    ),
                )
                updated_balance = connection.execute(
                    """
                    UPDATE leave_balances SET balance_days = balance_days - ?
                    WHERE employee_id = ? AND leave_type = ?
                    """,
                    (working_days, requester.employee_id, leave_type),
                ).rowcount
                if updated_balance != 1:
                    raise ToolError("The current leave balance could not be updated.")
                connection.commit()
        except sqlite3.Error as exc:
            raise ToolError(
                "A database error prevented the request; no changes were saved."
            ) from exc
        return ToolResult(
            f"Leave request {request_id} was submitted as pending for {working_days} "
            f"working days ({leave_type}, {start} to {end}).",
            {"request_id": request_id, "working_days": working_days, "status": "pending"},
        )

    def validate_cancel(self, requester: Employee, request_id: str) -> ToolResult:
        row = self._get_request(request_id)
        if row is None or row["employee_id"] != requester.employee_id:
            raise ToolError(REFUSAL)
        if row["status"] not in {"pending", "approved"}:
            raise ToolError("Only pending or approved leave requests can be cancelled.")
        if date.fromisoformat(row["start_date"]) <= self.today:
            raise ToolError("A leave request can only be cancelled before its start_date.")
        return ToolResult(
            f"Cancel request {request_id} ({row['leave_type']}, {row['start_date']} to "
            f"{row['end_date']}, {row['working_days']} working days)?",
            {"request_id": request_id},
        )

    def cancel_leave(self, requester: Employee, request_id: str) -> ToolResult:
        try:
            with self.database.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """
                    SELECT employee_id, leave_type, start_date, working_days, status
                    FROM leave_requests WHERE request_id = ?
                    """,
                    (request_id,),
                ).fetchone()
                if row is None or row["employee_id"] != requester.employee_id:
                    raise ToolError(REFUSAL)
                if row["status"] not in {"pending", "approved"}:
                    raise ToolError(
                        "Only pending or approved leave requests can be cancelled."
                    )
                if date.fromisoformat(row["start_date"]) <= self.today:
                    raise ToolError(
                        "A leave request can only be cancelled before its start_date."
                    )
                updated = connection.execute(
                    """
                    UPDATE leave_requests SET status = 'cancelled'
                    WHERE request_id = ? AND status IN ('pending', 'approved')
                    """,
                    (request_id,),
                ).rowcount
                if updated != 1:
                    raise ToolError("The request changed; it could not be cancelled.")
                balance_update = connection.execute(
                    """
                    UPDATE leave_balances SET balance_days = balance_days + ?
                    WHERE employee_id = ? AND leave_type = ?
                    """,
                    (row["working_days"], requester.employee_id, row["leave_type"]),
                ).rowcount
                if balance_update != 1:
                    raise ToolError("The leave balance could not be restored.")
                connection.commit()
        except sqlite3.Error as exc:
            raise ToolError(
                "A database error prevented cancellation; no changes were saved."
            ) from exc
        return ToolResult(
            f"Request {request_id} was cancelled and {row['working_days']} days were "
            f"returned to the {row['leave_type']} balance.",
            {"request_id": request_id, "status": "cancelled"},
        )

    def _authorized_target(self, requester: Employee, employee_id: str) -> Employee:
        with self.database.connect() as connection:
            target_row = connection.execute(
                "SELECT employee_id FROM employees WHERE employee_id = ?",
                (employee_id,),
            ).fetchone()
            if target_row is None:
                raise ToolError(REFUSAL)
            target_id = target_row["employee_id"]
            if target_id != requester.employee_id:
                if requester.role == "manager":
                    permitted = connection.execute(
                        """
                        SELECT 1 FROM employees
                        WHERE employee_id = ? AND manager_id = ?
                        """,
                        (target_id, requester.employee_id),
                    ).fetchone()
                elif requester.role == "hr":
                    permitted = True
                else:
                    permitted = False
                if not permitted:
                    raise ToolError(REFUSAL)
        target = self.database.get_employee(target_id)
        if target is None:
            raise ToolError(REFUSAL)
        return target

    def _get_request(self, request_id: str):
        with self.database.connect() as connection:
            return connection.execute(
                """
                SELECT employee_id, leave_type, start_date, end_date,
                       working_days, status
                FROM leave_requests WHERE request_id = ?
                """,
                (request_id,),
            ).fetchone()

    def _holidays_for(self, region: str, year: int) -> frozenset[date]:
        matching = [
            document
            for document in self.documents
            if document.metadata.doc_type == "calendar"
            and document.metadata.effective_date <= self.today
            and ("GLOBAL" in document.metadata.regions or region in document.metadata.regions)
            and any(day.year == year for day in self._holidays[document.metadata.doc_id])
        ]
        if not matching:
            raise ToolError(
                f"No applicable {year} public holiday calendar is available for region {region}."
            )
        best_scope = max(int("GLOBAL" not in doc.metadata.regions) for doc in matching)
        matching = [
            doc
            for doc in matching
            if int("GLOBAL" not in doc.metadata.regions) == best_scope
        ]
        latest_date = max(doc.metadata.effective_date for doc in matching)
        matching = [doc for doc in matching if doc.metadata.effective_date == latest_date]
        return frozenset(
            day
            for document in matching
            for day in self._holidays[document.metadata.doc_id]
            if day.year == year
        )

    @staticmethod
    def _working_days(start: date, end: date, holidays: frozenset[date]) -> int:
        count = 0
        current = start
        while current <= end:
            if current.weekday() < 5 and current not in holidays:
                count += 1
            current += timedelta(days=1)
        return count

    @staticmethod
    def _parse_range(start_value: str, end_value: str) -> tuple[date, date]:
        try:
            start = date.fromisoformat(start_value)
            end = date.fromisoformat(end_value)
        except ValueError as exc:
            raise ToolError("Both dates must use YYYY-MM-DD format.") from exc
        if end < start:
            raise ToolError("end_date must be on or after start_date.")
        return start, end

    @staticmethod
    def _format_days(value: float) -> str:
        return f"{value:g}"

    @staticmethod
    def _now_timestamp() -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat(timespec="seconds")
