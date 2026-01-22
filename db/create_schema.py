import sqlite3
from pathlib import Path

db_path = Path("data/workflow_data.db")
db_path.parent.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS tasks")
cursor.execute("DROP TABLE IF EXISTS ml_predictions")
cursor.execute("DROP TABLE IF EXISTS gpt_suggestions")
cursor.execute("DROP TABLE IF EXISTS bottleneck_history")
cursor.execute("DROP TABLE IF EXISTS task_reassignments")
cursor.execute("DROP TABLE IF EXISTS improvement_log")

cursor.execute("""
CREATE TABLE tasks (
    task_id INTEGER PRIMARY KEY,
    task_name TEXT,
    assignee TEXT,
    status TEXT,
    priority TEXT,
    project TEXT,
    created_date TEXT,
    start_date TEXT,
    end_date TEXT,
    actual_duration REAL,
    task_duration REAL,
    is_delayed INTEGER,
    bottleneck_type TEXT,
    comments TEXT,
    reassignment_count INTEGER DEFAULT 0
)
""")

cursor.execute("""
CREATE TABLE ml_predictions (
    prediction_id INTEGER PRIMARY KEY,
    task_id INTEGER,
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
""")

cursor.execute("""
CREATE TABLE gpt_suggestions (
    suggestion_id INTEGER PRIMARY KEY,
    task_id INTEGER,
    suggestion_text TEXT,
    root_causes TEXT,
    recommendations TEXT,
    prompt_version TEXT,
    model_used TEXT,
    sentiment TEXT,
    urgency TEXT,
    quality_score REAL,
    applied INTEGER,
    applied_action TEXT,
    created_at TEXT,
    feedback_status TEXT,
    feedback_notes TEXT,
    feedback_date TEXT,
    was_helpful INTEGER,
    FOREIGN KEY(task_id) REFERENCES tasks(task_id)
)
""")

cursor.execute("""
CREATE TABLE bottleneck_history (
    bottleneck_id INTEGER PRIMARY KEY,
    task_id INTEGER,
    bottleneck_type TEXT,
    severity_score REAL,
    root_cause_suggestion TEXT,
    detected_date TEXT,
    resolution_date TEXT,
    FOREIGN KEY(task_id) REFERENCES tasks(task_id)
)
""")

cursor.execute("""
CREATE TABLE task_reassignments (
    reassignment_id INTEGER PRIMARY KEY,
    task_id INTEGER,
    from_assignee TEXT,
    to_assignee TEXT,
    reason TEXT,
    triggered_by TEXT,
    reassignment_date TEXT,
    was_delayed_before INTEGER,
    FOREIGN KEY(task_id) REFERENCES tasks(task_id)
)
""")

cursor.execute("""
CREATE TABLE improvement_log (
    improvement_id INTEGER PRIMARY KEY,
    task_id INTEGER,
    action_taken TEXT,
    owner TEXT,
    date_applied TEXT,
    impact_measured TEXT,
    improvement_score REAL,
    improvement_percentage REAL,
    FOREIGN KEY(task_id) REFERENCES tasks(task_id)
)
""")

conn.commit()
conn.close()

print("✅ Database schema created successfully.")
