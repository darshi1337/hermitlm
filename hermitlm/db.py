import sqlite3
from datetime import datetime

DB_PATH = "data/hermit.db"


def get_conn():
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
        datetime.utcnow().isoformat()
    ))

    conn.commit()
    conn.close()