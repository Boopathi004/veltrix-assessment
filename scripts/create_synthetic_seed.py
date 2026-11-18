from __future__ import annotations

import shutil
import sqlite3
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATABASE = ROOT / "data" / "veltrix.db"
OUTPUT_DATABASE = ROOT / "veltrix.db"
DATA_DATABASE = ROOT / "data" / "veltrix.db"


def create_synthetic_seed() -> None:
    if not SOURCE_DATABASE.exists():
        raise FileNotFoundError(
            f"Source database is missing: {SOURCE_DATABASE}"
        )

    temp_database = OUTPUT_DATABASE.with_suffix(".db.tmp")
    shutil.copy2(SOURCE_DATABASE, temp_database)

    connection = sqlite3.connect(temp_database)
    try:
        rows = connection.execute(
            """
            SELECT employee_id
            FROM employees
            ORDER BY employee_id
            """
        ).fetchall()
        for index, (employee_id,) in enumerate(rows, start=1):
            connection.execute(
                """
                UPDATE employees
                SET full_name = ?, email = ?, phone = ?
                WHERE employee_id = ?
                """,
                (
                    f"Employee {index:03d}",
                    f"employee{index:03d}@example.test",
                    f"+155500{index:04d}",
                    employee_id,
                ),
            )

        table = connection.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = 'chat_sessions'
            """
        ).fetchone()
        if table:
            connection.execute("DELETE FROM chat_sessions")

        connection.commit()
    finally:
        connection.close()

    if OUTPUT_DATABASE.exists():
        OUTPUT_DATABASE.unlink()
    os.replace(temp_database, OUTPUT_DATABASE)
    DATA_DATABASE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUTPUT_DATABASE, DATA_DATABASE)


if __name__ == "__main__":
    create_synthetic_seed()
    print("Created synthetic anonymized database seed.")
