"""
Database Initialization Script for FlowFix AI
Creates all tables with complete schema including all required columns
Run this once to set up a clean database from scratch
"""
import sqlite3
import os
import sys
from datetime import datetime

def init_database(db_path='data/workflow_data.db'):
    """Initialize complete database schema"""
    
    # Backup existing database if it exists
    if os.path.exists(db_path):
        backup_path = db_path.replace('.db', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db')
        print(f"Backing up existing database to: {backup_path}")
        import shutil
        shutil.copy2(db_path, backup_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("Creating complete database schema...")
    
    # 1. Tasks table - main workflow data
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tasks (
        task_id TEXT PRIMARY KEY,
        task_name TEXT NOT NULL,
        assignee TEXT,
        status TEXT,
        created_date TEXT,
        start_date TEXT,
        end_date TEXT,
        priority TEXT,
        comments TEXT,
        project TEXT,
        planned_duration FLOAT,
        actual_duration FLOAT,
        is_delayed INTEGER DEFAULT 0,
        is_overdue INTEGER DEFAULT 0,
        bottleneck_type TEXT,
        task_duration FLOAT,
        severity_score INTEGER DEFAULT 0,
        estimated_duration FLOAT,
        delay_days FLOAT,
        reassignment_count INTEGER DEFAULT 0
    )
    """)
    print("  [SUCCESS] tasks table created")
    
    # 2. Bottleneck history - detection records
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS bottleneck_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL,
        bottleneck_type TEXT,
        severity_score INTEGER,
        detected_date TEXT,
        resolution_date TEXT,
        resolution_action TEXT,
        assignee TEXT,
        impact_score INTEGER,
        notes TEXT,
        created_at TEXT,
        delay_days FLOAT,
        priority TEXT,
        root_cause_suggestion TEXT,
        FOREIGN KEY (task_id) REFERENCES tasks(task_id)
    )
    """)
    print("  [SUCCESS] bottleneck_history table created")
    
    # 3. GPT suggestions - AI recommendations
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS gpt_suggestions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL,
        suggestion_text TEXT,
        root_cause TEXT,
        priority_level TEXT,
        estimated_impact TEXT,
        created_at TEXT,
        model_version TEXT,
        confidence_score FLOAT,
        feedback_status TEXT DEFAULT 'pending',
        feedback_notes TEXT DEFAULT '',
        feedback_date TEXT DEFAULT '',
        was_helpful INTEGER DEFAULT NULL,
        actual_impact_score FLOAT DEFAULT NULL,
        applied_date TEXT DEFAULT NULL,
        FOREIGN KEY (task_id) REFERENCES tasks(task_id)
    )
    """)
    print("  [SUCCESS] gpt_suggestions table created")
    
    # 4. Task reassignments - workload balancing
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS task_reassignments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL,
        from_assignee TEXT,
        to_assignee TEXT,
        reassignment_date TEXT,
        reason TEXT,
        status_before TEXT,
        status_after TEXT,
        created_at TEXT,
        triggered_by TEXT DEFAULT 'manual',
        was_delayed_before INTEGER DEFAULT 0,
        duration_before_reassignment FLOAT DEFAULT NULL,
        FOREIGN KEY (task_id) REFERENCES tasks(task_id)
    )
    """)
    print("  [SUCCESS] task_reassignments table created")
    
    # 5. ML predictions - machine learning results
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ml_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL,
        prediction_type TEXT,
        predicted_value FLOAT,
        predicted_class TEXT,
        probability FLOAT,
        model_version TEXT,
        created_at TEXT,
        actual_value FLOAT,
        prediction_error FLOAT,
        FOREIGN KEY (task_id) REFERENCES tasks(task_id)
    )
    """)
    print("  [SUCCESS] ml_predictions table created")
    
    # 6. ML training log - model performance tracking
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ml_training_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_type TEXT,
        model_version TEXT,
        trained_at TEXT,
        metrics_json TEXT,
        feature_importance_json TEXT,
        dataset_size INTEGER,
        train_test_split FLOAT,
        notes TEXT
    )
    """)
    print("  [SUCCESS] ml_training_log table created")
    
    # 7. Feedback log - suggestion tracking
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feedback_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        suggestion_id INTEGER,
        task_id TEXT,
        feedback_type TEXT,
        feedback_text TEXT,
        rating INTEGER,
        created_at TEXT,
        FOREIGN KEY (suggestion_id) REFERENCES gpt_suggestions(id),
        FOREIGN KEY (task_id) REFERENCES tasks(task_id)
    )
    """)
    print("  [SUCCESS] feedback_log table created")
    
    # 8. Improvement log - before/after tracking
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS improvement_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        action_type TEXT,
        action_description TEXT,
        applied_date TEXT,
        metric_before FLOAT,
        metric_after FLOAT,
        improvement_score FLOAT DEFAULT 0,
        improvement_percentage FLOAT DEFAULT 0,
        affected_tasks INTEGER,
        notes TEXT,
        created_at TEXT
    )
    """)
    print("  [SUCCESS] improvement_log table created")
    
    # 9. Dashboard summary - cached KPIs
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dashboard_summary (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        metric_name TEXT UNIQUE,
        metric_value FLOAT,
        metric_unit TEXT,
        updated_at TEXT
    )
    """)
    print("  [SUCCESS] dashboard_summary table created")
    
    # 10. Ingestion log - data loading history
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ingestion_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT,
        source_type TEXT,
        ingestion_mode TEXT,
        rows_processed INTEGER,
        rows_inserted INTEGER,
        rows_updated INTEGER,
        rows_failed INTEGER,
        started_at TEXT,
        completed_at TEXT,
        status TEXT,
        error_message TEXT
    )
    """)
    print("  [SUCCESS] ingestion_log table created")
    
    # 11. Bottleneck summary - aggregated metrics
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS bottleneck_summary (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        bottleneck_type TEXT,
        count INTEGER,
        avg_severity FLOAT,
        affected_assignees INTEGER,
        total_delay_days FLOAT,
        generated_at TEXT
    )
    """)
    print("  [SUCCESS] bottleneck_summary table created")
    
    # 12. Pipeline runs - scheduled job tracking
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pipeline_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pipeline_name TEXT,
        run_type TEXT,
        started_at TEXT,
        completed_at TEXT,
        status TEXT,
        records_processed INTEGER,
        error_message TEXT,
        duration_seconds FLOAT
    )
    """)
    print("  [SUCCESS] pipeline_runs table created")
    
    # Create indexes for better query performance
    print("\nCreating indexes...")
    
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_tasks_assignee ON tasks(assignee)",
        "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)",
        "CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project)",
        "CREATE INDEX IF NOT EXISTS idx_tasks_is_delayed ON tasks(is_delayed)",
        "CREATE INDEX IF NOT EXISTS idx_bottleneck_history_task_id ON bottleneck_history(task_id)",
        "CREATE INDEX IF NOT EXISTS idx_bottleneck_history_detected_date ON bottleneck_history(detected_date)",
        "CREATE INDEX IF NOT EXISTS idx_gpt_suggestions_task_id ON gpt_suggestions(task_id)",
        "CREATE INDEX IF NOT EXISTS idx_gpt_suggestions_feedback_status ON gpt_suggestions(feedback_status)",
        "CREATE INDEX IF NOT EXISTS idx_task_reassignments_task_id ON task_reassignments(task_id)",
        "CREATE INDEX IF NOT EXISTS idx_ml_predictions_task_id ON ml_predictions(task_id)",
    ]
    
    for idx_sql in indexes:
        cursor.execute(idx_sql)
    print(f"  [SUCCESS] Created {len(indexes)} indexes")
    
    conn.commit()
    conn.close()
    
    print(f"\n[SUCCESS] Database initialized: {db_path}")
    print(f"Total tables: 12")
    print(f"Total indexes: {len(indexes)}")
    print("\nYou can now:")
    print("  1. Import data: python src/ingestion.py data/your_file.csv")
    print("  2. Detect bottlenecks: python src/bottleneck_detector.py")
    print("  3. Train ML models: python src/ml_predictor.py")
    print("  4. Launch dashboard: streamlit run dashboard/streamlit_app.py")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Initialize FlowFix AI database')
    parser.add_argument('--db-path', default='data/workflow_data.db', 
                       help='Database file path (default: data/workflow_data.db)')
    parser.add_argument('--force', action='store_true',
                       help='Skip backup and overwrite existing database')
    
    args = parser.parse_args()
    
    if args.force and os.path.exists(args.db_path):
        print(f"[WARNING] Force mode: deleting existing database")
        os.remove(args.db_path)
    
    init_database(args.db_path)
