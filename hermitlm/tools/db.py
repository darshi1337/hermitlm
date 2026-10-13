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
        fine_label TEXT,
        tool TEXT,
        timestamp TEXT
    )
    """)

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


def insert_intent(user_input, intent, tool):
    """Log one decision-model verdict. intent is the classify_intent() dict or None."""
    conn = get_conn()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO intent_log (user_input, intent_label, intent_name, confidence, p_confused, fine_label, tool, timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        (user_input or "")[:200],
        intent["label"] if intent else None,
        intent["name"] if intent else None,
        intent["confidence"] if intent else None,
        intent["p_confused"] if intent else None,
        (intent or {}).get("fine_label"),
        tool,
        datetime.now(timezone.utc).isoformat()
    ))

    conn.commit()
    conn.close()


def intent_stats():
    """Return {total, by_label, by_tool, confused_rate} over intent_log."""
    conn = get_conn()
    cursor = conn.cursor()

    total = cursor.execute("SELECT COUNT(*) FROM intent_log").fetchone()[0]
    by_label = dict(cursor.execute(
        "SELECT intent_label, COUNT(*) FROM intent_log GROUP BY intent_label"
    ).fetchall())
    by_tool = dict(cursor.execute(
        "SELECT tool, COUNT(*) FROM intent_log GROUP BY tool"
    ).fetchall())
    confused = cursor.execute(
        "SELECT COUNT(*) FROM intent_log WHERE p_confused >= 0.9"
    ).fetchone()[0]

    conn.close()
    return {
        "total": total,
        "by_label": by_label,
        "by_tool": {k if k is not None else "none": v for k, v in by_tool.items()},
        "confused_rate": round(confused / total, 4) if total else 0.0,
    }
