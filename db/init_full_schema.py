import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "workflow_data.db"

def create_full_schema():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Tasks table (must match dummy + model fields)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tasks (
        task_id TEXT PRIMARY KEY,
        task_name TEXT,
        assignee TEXT,
        project TEXT,
        status TEXT,
        created_at TEXT,
        updated_at TEXT,
        priority TEXT,
        estimated_duration INTEGER,
        actual_duration INTEGER,
        bottleneck_type TEXT,
        start_date TEXT,
        is_delayed INTEGER,
        delay_reason TEXT,
        completion_percentage INTEGER
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS gpt_suggestions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT,
        suggestion_text TEXT,
        root_causes TEXT,
        recommendations TEXT,
        prompt_version TEXT,
        model_used TEXT,
        sentiment TEXT,
        urgency TEXT,
        quality_score REAL,
        needs_manual_review INTEGER,
        applied INTEGER,
        created_at TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ml_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT,
        predicted_duration REAL,
        predicted_status TEXT,
        predicted_priority TEXT,
        created_at TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS bottleneck_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT,
        assignee TEXT,
        bottleneck_type TEXT,
        detected_at TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS improvement_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT,
        improvement_type TEXT,
        notes TEXT,
        logged_at TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS task_reassignments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT,
        from_assignee TEXT,
        to_assignee TEXT,
        reassigned_at TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feedback_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        page_name TEXT,
        feedback_text TEXT,
        rating INTEGER,
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()
    print("✅ Full schema initialized successfully.")

if __name__ == "__main__":
    create_full_schema()
