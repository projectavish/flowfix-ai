import os
import sqlite3

DB_PATH = os.path.join("data", "workflow_data.db")

def col_exists(cur, table, col):
    cur.execute(f"PRAGMA table_info({table})")
    return any(r[1] == col for r in cur.fetchall())

def table_exists(cur, table):
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
    return cur.fetchone() is not None

def add_col(cur, table, col, ddl_type):
    if not col_exists(cur, table, col):
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl_type}")
        print(f"✅ Added column: {table}.{col} {ddl_type}")

def main():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"DB not found at: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # --- Ensure tables exist ---
    # ml_training_log (your ML code expects JSON fields)
    if not table_exists(cur, "ml_training_log"):
        cur.execute("""
        CREATE TABLE ml_training_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_type TEXT,
            model_version TEXT,
            metrics_json TEXT,
            feature_importance_json TEXT
        )
        """)
        print("✅ Created table: ml_training_log")

    # ml_predictions (your ML code expects these fields)
    if not table_exists(cur, "ml_predictions"):
        cur.execute("""
        CREATE TABLE ml_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            model_type TEXT,
            prediction_value REAL,
            confidence_score REAL,
            model_version TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        print("✅ Created table: ml_predictions")
    else:
        # Add missing columns to existing ml_predictions
        add_col(cur, "ml_predictions", "model_type", "TEXT")
        add_col(cur, "ml_predictions", "prediction_value", "REAL")
        add_col(cur, "ml_predictions", "confidence_score", "REAL")
        add_col(cur, "ml_predictions", "model_version", "TEXT")
        add_col(cur, "ml_predictions", "created_at", "TIMESTAMP")

    conn.commit()
    conn.close()
    print("\n✅ Schema fix complete.")

if __name__ == "__main__":
    main()