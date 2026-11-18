from __future__ import annotations

import sqlite3
from datetime import date

import pytest

from app.database import Database
from app.tools import REFUSAL, LeaveTools, ToolError


def test_working_days_exclude_weekends_and_documented_holidays(
    leave_tools: LeaveTools, employee
) -> None:
    start, end, days = leave_tools.validate_apply(
        employee,
        {
            "leave_type": "annual",
            "start_date": "2026-01-01",
            "end_date": "2026-01-05",
        },
    )
    assert (start, end, days) == (
        date(2026, 1, 1),
        date(2026, 1, 5),
        2,
    )


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (
            {
                "leave_type": "annual",
                "start_date": "2026-01-03",
                "end_date": "2026-01-04",
            },
            "zero working days",
        ),
        (
            {
                "leave_type": "annual",
                "start_date": "2025-12-30",
                "end_date": "2026-01-02",
            },
            "within the 2026 calendar year",
        ),
        (
            {
                "leave_type": "annual",
                "start_date": "2026-01-02",
                "end_date": "2026-01-01",
            },
            "on or after",
        ),
        (
            {
                "leave_type": "other",
                "start_date": "2026-01-02",
                "end_date": "2026-01-02",
            },
            "annual or sick",
        ),
    ],
)
def test_apply_rejects_invalid_date_ranges_and_leave_types(
    leave_tools: LeaveTools, employee, args, expected
) -> None:
    with pytest.raises(ToolError, match=expected):
        leave_tools.validate_apply(employee, args)


def test_apply_rejects_insufficient_balance_and_overlapping_request(
    database: Database, leave_tools: LeaveTools, employee
) -> None:
    with database.connect() as connection:
        connection.execute(
            "UPDATE leave_balances SET balance_days = 1 "
            "WHERE employee_id = 'E1001' AND leave_type = 'annual'"
        )
        connection.commit()
    with pytest.raises(ToolError, match="current annual balance"):
        leave_tools.validate_apply(
            employee,
            {
                "leave_type": "annual",
                "start_date": "2026-01-02",
                "end_date": "2026-01-05",
            },
        )

    with database.connect() as connection:
        connection.execute(
            "UPDATE leave_balances SET balance_days = 5 "
            "WHERE employee_id = 'E1001' AND leave_type = 'annual'"
        )
        connection.execute(
            """
            INSERT INTO leave_requests
                (request_id, employee_id, leave_type, start_date, end_date,
                 working_days, status, created_at)
            VALUES ('existing', 'E1001', 'annual', '2026-01-05', '2026-01-05',
                    1, 'pending', '2026-01-01T00:00:00')
            """
        )
        connection.commit()
    with pytest.raises(ToolError, match="overlap"):
        leave_tools.validate_apply(
            employee,
            {
                "leave_type": "annual",
                "start_date": "2026-01-05",
                "end_date": "2026-01-06",
            },
        )


def test_apply_is_atomic_and_subtracts_balance(
    database: Database, leave_tools: LeaveTools, employee
) -> None:
    result = leave_tools.apply_leave(
        employee,
        leave_type="annual",
        start_date="2026-01-01",
        end_date="2026-01-05",
    )
    with database.connect() as connection:
        request = connection.execute(
            "SELECT employee_id, leave_type, start_date, end_date, working_days, status "
            "FROM leave_requests WHERE request_id = ?",
            (result.data["request_id"],),
        ).fetchone()
        balance = connection.execute(
            "SELECT balance_days FROM leave_balances "
            "WHERE employee_id = 'E1001' AND leave_type = 'annual'"
        ).fetchone()["balance_days"]
    assert request["status"] == "pending"
    assert request["working_days"] == 2
    assert balance == 3


def test_cancel_is_atomic_and_restores_balance(
    database: Database, leave_tools: LeaveTools, employee
) -> None:
    with database.connect() as connection:
        connection.execute(
            "UPDATE leave_balances SET balance_days = 3 "
            "WHERE employee_id = 'E1001' AND leave_type = 'annual'"
        )
        connection.execute(
            """
            INSERT INTO leave_requests
                (request_id, employee_id, leave_type, start_date, end_date,
                 working_days, status, created_at)
            VALUES ('future-request', 'E1001', 'annual', '2026-01-05', '2026-01-06',
                    2, 'approved', '2026-01-01T00:00:00')
            """
        )
        connection.commit()
    preview = leave_tools.validate_cancel(employee, "future-request")
    assert "Cancel request future-request" in preview.message
    result = leave_tools.cancel_leave(employee, "future-request")
    with database.connect() as connection:
        status = connection.execute(
            "SELECT status FROM leave_requests WHERE request_id = 'future-request'"
        ).fetchone()["status"]
        balance = connection.execute(
            "SELECT balance_days FROM leave_balances "
            "WHERE employee_id = 'E1001' AND leave_type = 'annual'"
        ).fetchone()["balance_days"]
    assert result.data["status"] == "cancelled"
    assert status == "cancelled"
    assert balance == 5


def test_cancel_rejects_started_and_other_employee_requests(
    database: Database, leave_tools: LeaveTools, employee
) -> None:
    with database.connect() as connection:
        connection.execute(
            """
            INSERT INTO leave_requests
                (request_id, employee_id, leave_type, start_date, end_date,
                 working_days, status, created_at)
            VALUES ('started', 'E1001', 'annual', '2026-01-01', '2026-01-01',
                    1, 'pending', '2025-12-20T00:00:00'),
                   ('other', 'E2002', 'annual', '2026-01-05', '2026-01-05',
                    1, 'pending', '2025-12-20T00:00:00')
            """
        )
        connection.commit()
    with pytest.raises(ToolError, match="before its start_date"):
        leave_tools.validate_cancel(employee, "started")
    with pytest.raises(ToolError) as existing:
        leave_tools.validate_cancel(employee, "other")
    with pytest.raises(ToolError) as missing:
        leave_tools.validate_cancel(employee, "missing")
    assert str(existing.value) == str(missing.value) == REFUSAL


def test_leave_data_authorization_for_employee_manager_and_hr(
    database: Database, leave_tools: LeaveTools, employee
) -> None:
    with pytest.raises(ToolError) as existing:
        leave_tools.get_leave_balance(employee, "E2002")
    with pytest.raises(ToolError) as missing:
        leave_tools.get_leave_balance(employee, "E9999")
    assert str(existing.value) == str(missing.value) == REFUSAL

    manager = database.get_employee("E2001")
    hr = database.get_employee("E9001")
    assert manager is not None and hr is not None
    direct_report = leave_tools.get_leave_balance(manager, "E2002", "annual")
    assert direct_report.data["employee_id"] == "E2002"
    with pytest.raises(ToolError, match=REFUSAL):
        leave_tools.get_leave_balance(manager, "E2003")
    hr_result = leave_tools.list_leave_requests(hr, "E2003")
    assert hr_result.data["employee_id"] == "E2003"


def test_team_leave_is_manager_only_and_limited_to_direct_reports(
    database: Database, leave_tools: LeaveTools, employee
) -> None:
    with database.connect() as connection:
        connection.executemany(
            """
            INSERT INTO leave_requests
                (request_id, employee_id, leave_type, start_date, end_date,
                 working_days, status, created_at)
            VALUES (?, ?, 'annual', ?, ?, 1, ?, '2026-01-01T00:00:00')
            """,
            [
                ("report", "E2002", "2026-01-05", "2026-01-05", "pending"),
                ("other", "E2003", "2026-01-05", "2026-01-05", "approved"),
            ],
        )
        connection.commit()
    manager = database.get_employee("E2001")
    assert manager is not None
    result = leave_tools.get_team_leave(manager, "2026-01-01", "2026-01-10")
    assert [row["employee_id"] for row in result.data["leave"]] == ["E2002"]
    with pytest.raises(ToolError, match=REFUSAL):
        leave_tools.get_team_leave(employee, "2026-01-01", "2026-01-10")
