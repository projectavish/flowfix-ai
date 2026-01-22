import sqlite3

def create_tables():
    conn = sqlite3.connect('data/workflow_data.db')
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tasks (
        task_id TEXT PRIMARY KEY,
        task_name TEXT,
        assignee TEXT,
        project TEXT,
        priority TEXT,
        status TEXT,
        actual_duration REAL,
        is_delayed INTEGER
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ml_predictions (
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT,
        model_type TEXT,
        predicted_duration REAL,
        predicted_delay_prob REAL,
        confidence_score REAL,
        model_version TEXT,
        prediction_date TEXT,
        actual_outcome TEXT,
        prediction_correct INTEGER,
        FOREIGN KEY(task_id) REFERENCES tasks(task_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS gpt_suggestions (
        suggestion_id INTEGER PRIMARY KEY AUTOINCREMENT,
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
        applied_action TEXT,
        created_at TEXT,
        feedback_status TEXT,
        feedback_notes TEXT,
        feedback_date TEXT,
        was_helpful INTEGER,
        FOREIGN KEY(task_id) REFERENCES tasks(task_id)
    )
    ''')

    conn.commit()
    conn.close()
    print("✅ Database and tables created successfully!")

if __name__ == "__main__":
    create_tables()
