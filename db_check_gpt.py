import sqlite3

conn = sqlite3.connect("data/workflow_data.db")
cur = conn.cursor()

print("Tables:")
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
for row in cur.fetchall():
    print(row)

print("\nGPT suggestions schema:")
cur.execute("PRAGMA table_info(gpt_suggestions)")
for row in cur.fetchall():
    print(row)

conn.close()