import sqlite3
import os

DB_PATH = os.path.join("data", "workflow_data.db")

def col_exists(cur, table, col):
    cur.execute(f"PRAGMA table_info({table})")
    return col in [r[1] for r in cur.fetchall()]

def add_col(cur, table, col, col_type):
    if not col_exists(cur, table, col):
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
        print(f"✅ Added column: {table}.{col} ({col_type})")
    else:
        print(f"OK column exists: {table}.{col}")

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Ensure tasks has required columns for your codebase
    add_col(cur, "tasks", "created_date", "DATE")
    add_col(cur, "tasks", "planned_duration", "INTEGER")
    add_col(cur, "tasks", "actual_duration", "INTEGER")
    add_col(cur, "tasks", "is_delayed", "BOOLEAN DEFAULT 0")
    add_col(cur, "tasks", "is_overdue", "BOOLEAN DEFAULT 0")
    add_col(cur, "tasks", "bottleneck_type", "TEXT")
    add_col(cur, "tasks", "reassignment_count", "INTEGER DEFAULT 0")
    add_col(cur, "tasks", "created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")

    conn.commit()
    conn.close()
    print("\n✅ Schema fix for tasks completed.")

if __name__ == "__main__":
    main()