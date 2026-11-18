import re
import sqlite3
from datetime import datetime, timezone

DB_PATH = "data/hermit.db"

# Case-insensitive patterns applied to the *original* text. Using a lowercased
# copy to find indices and then slicing the original breaks on Unicode where
# lowercasing changes string length.
_FACT_PATTERNS = [
    ("name", re.compile(r"\bmy name is\s+([^,.;!?\n]+)", re.IGNORECASE)),
    ("likes", re.compile(r"\bi like\s+([^,.;!?\n]+)", re.IGNORECASE)),
    ("location", re.compile(r"\bi am from\s+([^,.;!?\n]+)", re.IGNORECASE)),
]


def extract_fact(text):
    """Return (key, value) for the first remembered fact in `text`, or None."""
    for key, pattern in _FACT_PATTERNS:
        match = pattern.search(text or "")
        if match:
            value = match.group(1).strip()
            if value:
                return key, value
    return None

def save_memory(user_id, key, value):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO memories (user_id, key, value, timestamp)
        VALUES (?, ?, ?, ?)
    """, (
        str(user_id),
        key,
        value,
        datetime.now(timezone.utc).isoformat()
    ))

    conn.commit()
    conn.close()

def get_memory(user_id, key):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    row = cursor.execute("""
        SELECT value
        FROM memories
        WHERE user_id = ? AND key = ?
        ORDER BY id DESC
        LIMIT 1
    """, (
        str(user_id),
        key
    )).fetchone()

    conn.close()

    if row:
        return row[0]

    return None

def get_all_memories(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    rows = cursor.execute("""
        SELECT key, value
        FROM memories
        WHERE user_id = ?
        ORDER BY id DESC
    """, (
        str(user_id),
    )).fetchall()

    conn.close()

    return rows

def init_memory_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        key TEXT,
        value TEXT,
        timestamp TEXT
    )
    """)

    conn.commit()
    conn.close()