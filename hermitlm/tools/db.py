import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = "data/hermit.db"


def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_conn()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        username TEXT,
        user_input TEXT,
        bot_response TEXT,
        channel_id TEXT,
        timestamp TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS intent_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_input TEXT,
        intent_label TEXT,
        intent_name TEXT,
        confidence REAL,
        p_confused REAL,
        p_yes REAL,
        fine_label TEXT,
        tool TEXT,
        expected_score REAL,
        clarified INTEGER,
        escalated INTEGER,
        timestamp TEXT
    )
    """)

    # Existing databases predate p_yes / expected_score / gate flags. Add the
    # columns in place so an upgrade never drops logged turns.
    columns = {row[1] for row in cursor.execute("PRAGMA table_info(intent_log)").fetchall()}
    for column, definition in (
        ("p_yes", "REAL"),
        ("expected_score", "REAL"),
        ("clarified", "INTEGER"),
        ("escalated", "INTEGER"),
    ):
        if column not in columns:
            cursor.execute(f"ALTER TABLE intent_log ADD COLUMN {column} {definition}")

    conn.commit()
    conn.close()


def insert_conversation(user_id, username, user_input, bot_response, channel_id):
    conn = get_conn()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO conversations (user_id, username, user_input, bot_response, channel_id, timestamp)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        str(user_id),
        username,
        user_input,
        bot_response,
        str(channel_id),
        datetime.now(timezone.utc).isoformat()
    ))

    conn.commit()
    conn.close()


def insert_intent(user_input, intent, tool, expected=None):
    """Log one decision-model verdict. intent is the classify_intent() dict or None."""
    conn = get_conn()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO intent_log (user_input, intent_label, intent_name, confidence, p_confused, p_yes, fine_label, tool, expected_score, clarified, escalated, timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        (user_input or "")[:200],
        intent["label"] if intent else None,
        intent["name"] if intent else None,
        intent["confidence"] if intent else None,
        intent["p_confused"] if intent else None,
        (intent or {}).get("p_yes"),
        (intent or {}).get("fine_label"),
        tool,
        float(expected) if expected is not None else None,
        int(bool((intent or {}).get("clarified"))),
        int(bool((intent or {}).get("escalated"))),
        datetime.now(timezone.utc).isoformat()
    ))

    conn.commit()
    conn.close()


def intent_stats():
    """Return counts plus the observed gate abstention rate over intent_log."""
    conn = get_conn()
    cursor = conn.cursor()

    total = cursor.execute("SELECT COUNT(*) FROM intent_log").fetchone()[0]
    by_label = dict(cursor.execute(
        "SELECT intent_label, COUNT(*) FROM intent_log GROUP BY intent_label"
    ).fetchall())
    by_tool = dict(cursor.execute(
        "SELECT tool, COUNT(*) FROM intent_log GROUP BY tool"
    ).fetchall())
    # The old hard-coded 0.9 was unrelated to the calibrated escalation gate.
    # Count log rows by the gate decisions that actually fired instead.
    abstained = cursor.execute(
        "SELECT COUNT(*) FROM intent_log WHERE COALESCE(clarified, 0) = 1 "
        "OR COALESCE(escalated, 0) = 1"
    ).fetchone()[0]
    escalated = cursor.execute(
        "SELECT COUNT(*) FROM intent_log WHERE COALESCE(escalated, 0) = 1"
    ).fetchone()[0]

    conn.close()
    return {
        "total": total,
        "by_label": by_label,
        "by_tool": {k if k is not None else "none": v for k, v in by_tool.items()},
        "confused_rate": round(escalated / total, 4) if total else 0.0,
        "abstention_rate": round(abstained / total, 4) if total else 0.0,
    }
