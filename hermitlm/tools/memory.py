import sqlite3
from datetime import datetime, timezone

DB_PATH = "data/hermit.db"

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