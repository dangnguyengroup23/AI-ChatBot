import sqlite3

def create_connection():
    return sqlite3.connect("conversation.db")

def create_table():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            timestamp TEXT,
            user_message TEXT,
            bot_response TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_conversation(session_id,timestamp,user_message,bot_response):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO conversations (session_id, timestamp, user_message, bot_response)
        VALUES (?, ?, ?, ?)
    """, (session_id, timestamp, user_message, bot_response))
    conn.commit()
    conn.close()
