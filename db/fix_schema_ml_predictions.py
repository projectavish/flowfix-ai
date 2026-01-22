import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "workflow_data.db"

def fix_schema():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Drop old table if exists
        cursor.execute("DROP TABLE IF EXISTS ml_predictions")

        # Create corrected schema
        cursor.execute("""
        CREATE TABLE ml_predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT,
            predicted_duration REAL,
            predicted_status TEXT,
            predicted_priority TEXT,
            model_type TEXT,
            model_version TEXT,
            run_id TEXT,
            created_at TEXT
        )
        """)

        conn.commit()
        print("✅ ml_predictions table recreated successfully with all required columns.")
    except Exception as e:
        print(f"❌ Failed to recreate table: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    fix_schema()
