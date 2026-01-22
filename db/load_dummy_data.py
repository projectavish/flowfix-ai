import sqlite3
import pandas as pd
import os
from datetime import datetime

DB_PATH = os.path.join("data", "workflow_data.db")
CSV_PATH = os.path.join("data", "dummy_tasks.csv")

def to_int(x, default=0):
    try:
        if x == "" or pd.isna(x):
            return default
        return int(float(x))
    except:
        return default

def to_float(x, default=None):
    try:
        if x == "" or pd.isna(x):
            return default
        return float(x)
    except:
        return default

def to_bool01(x, default=0):
    if x in [True, "True", "true", "1", 1]:
        return 1
    if x in [False, "False", "false", "0", 0, ""]:
        return 0
    try:
        return 1 if int(float(x)) != 0 else 0
    except:
        return default

# Connect
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Clear existing data
cursor.execute("DELETE FROM ml_predictions")
cursor.execute("DELETE FROM gpt_suggestions")
cursor.execute("DELETE FROM tasks")
conn.commit()
print("✅ Cleared tasks, ml_predictions, gpt_suggestions")

# Load CSV
df = pd.read_csv(CSV_PATH)
df = df.fillna("")

now_date = datetime.now().date().isoformat()

rows = []
for _, r in df.iterrows():
    task_id = str(r.get("task_id", "")).strip()
    if not task_id:
        continue

    task_name = str(r.get("task_name", "")).strip() or "Untitled Task"
    assignee = str(r.get("assignee", "")).strip() or "Unassigned"
    status = str(r.get("status", "")).strip() or "To Do"
    priority = str(r.get("priority", "")).strip() or "Low"
    project = str(r.get("project", "")).strip() or "General"

    comments = str(r.get("comments", "")).strip()
    bottleneck_type = str(r.get("bottleneck_type", "")).strip() or None

    start_date = str(r.get("start_date", "")).strip() or None
    end_date = str(r.get("end_date", "")).strip() or None

    planned_duration = to_int(r.get("planned_duration", ""), default=None)
    actual_duration = to_int(r.get("actual_duration", ""), default=None)

    is_delayed = to_bool01(r.get("is_delayed", ""), default=0)
    is_overdue = to_bool01(r.get("is_overdue", ""), default=0)

    reassignment_count = to_int(r.get("reassignment_count", ""), default=0)

    created_date = str(r.get("created_date", "")).strip() or now_date

    rows.append((
        task_id, task_name, assignee, status, created_date,
        start_date, end_date, priority, comments, project,
        planned_duration, actual_duration, is_delayed, is_overdue,
        bottleneck_type, reassignment_count
    ))

cursor.executemany("""
    INSERT INTO tasks (
        task_id, task_name, assignee, status, created_date,
        start_date, end_date, priority, comments, project,
        planned_duration, actual_duration, is_delayed, is_overdue,
        bottleneck_type, reassignment_count
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", rows)

conn.commit()
conn.close()

print(f"✅ Loaded {len(rows)} tasks into tasks table")
print("✅ Skipped inserting into ml_predictions and gpt_suggestions (not needed for training)")