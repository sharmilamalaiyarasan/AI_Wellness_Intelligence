"""
Database Module — SQLite workout history storage.
"""
import os
import sqlite3
from datetime import datetime

# Place database next to this file's parent package
_HERE = os.path.dirname(__file__)
DB_PATH = os.path.abspath(os.path.join(_HERE, "..", "fitness_history.db"))


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist yet."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS workout_sessions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                date             TEXT    NOT NULL,
                exercise         TEXT    NOT NULL,
                total_reps       INTEGER DEFAULT 0,
                correct_reps     INTEGER DEFAULT 0,
                incorrect_reps   INTEGER DEFAULT 0,
                duration_seconds INTEGER DEFAULT 0,
                form_accuracy    REAL    DEFAULT 0.0,
                rep_type         TEXT    DEFAULT 'reps'
            )
        """)
        conn.commit()


def save_workout(
    exercise: str,
    total_reps: int,
    correct_reps: int,
    incorrect_reps: int,
    duration_seconds: int,
    form_accuracy: float,
    rep_type: str = "reps",
) -> int:
    """Insert a completed workout session. Returns the new row id."""
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO workout_sessions
                (date, exercise, total_reps, correct_reps, incorrect_reps,
                 duration_seconds, form_accuracy, rep_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                exercise,
                total_reps,
                correct_reps,
                incorrect_reps,
                duration_seconds,
                round(form_accuracy, 1),
                rep_type,
            ),
        )
        conn.commit()
        return cur.lastrowid


def get_all_workouts() -> list[dict]:
    """Return all sessions newest-first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM workout_sessions ORDER BY id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_recent_workouts(limit: int = 5) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM workout_sessions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_aggregate_stats() -> dict:
    """Return totals across all sessions."""
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*)           AS total_workouts,
                COALESCE(SUM(total_reps), 0)       AS total_reps,
                COALESCE(AVG(form_accuracy), 0)    AS avg_accuracy,
                COALESCE(SUM(duration_seconds), 0) AS total_seconds
            FROM workout_sessions
            """
        ).fetchone()
    return dict(row) if row else {}
