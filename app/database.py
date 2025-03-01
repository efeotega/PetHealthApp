import sqlite3
from datetime import datetime

def get_db():
    return sqlite3.connect('../data/pet_health.db')

def init_db():
    with get_db() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS health_logs
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      diet TEXT,
                      behavior INTEGER,
                      stool_appearance TEXT,
                      timestamp DATETIME)''')
        conn.commit()

init_db()