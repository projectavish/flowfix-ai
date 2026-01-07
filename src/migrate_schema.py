"""
Database Schema Migration Utility
Ensures all tables and columns exist for production modules
"""
import sys
sys.path.insert(0, 'src')

from utils import get_engine
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def migrate_schema():
    """Run all schema migrations"""
    engine = get_engine()
    conn = engine.connect()
    
    logger.info("="*80)
    logger.info("DATABASE SCHEMA MIGRATION")
    logger.info("="*80)
    
    # Get existing tables
    result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
    existing_tables = [row[0] for row in result.fetchall()]
    
    migrations_applied = 0
    
    # 1. Add missing columns to tasks table
    logger.info("\n[1/5] Migrating tasks table...")
    if 'tasks' in existing_tables:
        result = conn.execute(text("PRAGMA table_info(tasks)"))
        task_columns = [row[1] for row in result.fetchall()]
        
        task_migrations = {
            'task_duration': 'FLOAT DEFAULT 0',
            'severity_score': 'INTEGER DEFAULT 0',
            'estimated_duration': 'FLOAT DEFAULT 0',
            'delay_days': 'FLOAT DEFAULT 0'
        }
        
        for col, definition in task_migrations.items():
            if col not in task_columns:
                try:
                    conn.execute(text(f"ALTER TABLE tasks ADD COLUMN {col} {definition}"))
                    logger.info(f"  ✓ Added column: {col}")
                    migrations_applied += 1
                except Exception as e:
                    logger.warning(f"  ✗ Could not add {col}: {e}")
    
    # 2. Add missing columns to gpt_suggestions table
    logger.info("\n[2/5] Migrating gpt_suggestions table...")
    if 'gpt_suggestions' in existing_tables:
        result = conn.execute(text("PRAGMA table_info(gpt_suggestions)"))
        gpt_columns = [row[1] for row in result.fetchall()]
        
        gpt_migrations = {
            'quality_score': 'INTEGER DEFAULT 0',
            'sentiment': "TEXT DEFAULT 'neutral'",
            'urgency': "TEXT DEFAULT 'medium'",
            'needs_manual_review': 'INTEGER DEFAULT 0',
            'prompt_version': "TEXT DEFAULT '1.0'",
            'model_used': "TEXT DEFAULT 'gpt-4'",
            'predicted_improvement': 'FLOAT DEFAULT 0'
        }
        
        for col, definition in gpt_migrations.items():
            if col not in gpt_columns:
                try:
                    conn.execute(text(f"ALTER TABLE gpt_suggestions ADD COLUMN {col} {definition}"))
                    logger.info(f"  ✓ Added column: {col}")
                    migrations_applied += 1
                except Exception as e:
                    logger.warning(f"  ✗ Could not add {col}: {e}")
    
    # 3. Create ml_predictions table if missing
    logger.info("\n[3/5] Creating ml_predictions table...")
    if 'ml_predictions' not in existing_tables:
        try:
            conn.execute(text("""
                CREATE TABLE ml_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    model_version TEXT,
                    prediction_date TEXT,
                    predicted_duration FLOAT,
                    predicted_delay_prob FLOAT,
                    actual_outcome TEXT,
                    prediction_correct INTEGER,
                    features_used TEXT,
                    confidence_score FLOAT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """))
            logger.info("  ✓ Created ml_predictions table")
            migrations_applied += 1
        except Exception as e:
            logger.warning(f"  ✗ Could not create table: {e}")
    else:
        logger.info("  - Table already exists")
    
    # 4. Create bottleneck_history table if missing
    logger.info("\n[4/5] Creating bottleneck_history table...")
    if 'bottleneck_history' not in existing_tables:
        try:
            conn.execute(text("""
                CREATE TABLE bottleneck_history (
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
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """))
            logger.info("  ✓ Created bottleneck_history table")
            migrations_applied += 1
        except Exception as e:
            logger.warning(f"  ✗ Could not create table: {e}")
    else:
        logger.info("  - Table already exists")
    
    # 5. Create ml_training_log table if missing
    logger.info("\n[5/5] Creating ml_training_log table...")
    if 'ml_training_log' not in existing_tables:
        try:
            conn.execute(text("""
                CREATE TABLE ml_training_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_type TEXT NOT NULL,
                    model_version TEXT,
                    training_date TEXT,
                    dataset_size INTEGER,
                    features_count INTEGER,
                    accuracy FLOAT,
                    precision_score FLOAT,
                    recall FLOAT,
                    f1_score FLOAT,
                    parameters TEXT,
                    model_path TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """))
            logger.info("  ✓ Created ml_training_log table")
            migrations_applied += 1
        except Exception as e:
            logger.warning(f"  ✗ Could not create table: {e}")
    else:
        logger.info("  - Table already exists")
    
    # 6. Create feedback_log table if missing
    logger.info("\n[6/6] Creating feedback_log table...")
    if 'feedback_log' not in existing_tables:
        try:
            conn.execute(text("""
                CREATE TABLE feedback_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    feedback_date TEXT,
                    impact_type TEXT,
                    impact_value FLOAT,
                    impact_score INTEGER,
                    notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """))
            logger.info("  ✓ Created feedback_log table")
            migrations_applied += 1
        except Exception as e:
            logger.warning(f"  ✗ Could not create table: {e}")
    else:
        logger.info("  - Table already exists")
    
    # 7. Create dashboard_summary table if missing
    logger.info("\n[7/7] Creating dashboard_summary table...")
    if 'dashboard_summary' not in existing_tables:
        try:
            conn.execute(text("""
                CREATE TABLE dashboard_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value FLOAT,
                    metric_category TEXT,
                    last_updated TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """))
            logger.info("  ✓ Created dashboard_summary table")
            migrations_applied += 1
        except Exception as e:
            logger.warning(f"  ✗ Could not create table: {e}")
    else:
        logger.info("  - Table already exists")
    
    # 8. Create ingestion_log table if missing
    logger.info("\n[8/8] Creating ingestion_log table...")
    if 'ingestion_log' not in existing_tables:
        try:
            conn.execute(text("""
                CREATE TABLE ingestion_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_file TEXT,
                    records_loaded INTEGER,
                    records_merged INTEGER,
                    records_skipped INTEGER,
                    merge_strategy TEXT,
                    status TEXT,
                    error_message TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """))
            logger.info("  ✓ Created ingestion_log table")
            migrations_applied += 1
        except Exception as e:
            logger.warning(f"  ✗ Could not create table: {e}")
    else:
        logger.info("  - Table already exists")
    
    # Update task_duration from planned_duration where needed
    logger.info("\n[9/9] Updating task_duration values...")
    try:
        conn.execute(text("""
            UPDATE tasks 
            SET task_duration = planned_duration 
            WHERE task_duration IS NULL OR task_duration = 0
        """))
        result = conn.execute(text("SELECT changes()"))
        changes = result.fetchone()[0]
        if changes > 0:
            logger.info(f"  ✓ Updated {changes} task_duration values")
            migrations_applied += 1
    except Exception as e:
        logger.warning(f"  ✗ Could not update task_duration: {e}")
    
    # Commit all changes
    conn.commit()
    conn.close()
    
    logger.info("\n" + "="*80)
    logger.info(f"✅ Migration complete: {migrations_applied} changes applied")
    logger.info("="*80)
    
    return migrations_applied

if __name__ == "__main__":
    try:
        count = migrate_schema()
        print(f"\n✅ Database schema is now up-to-date ({count} migrations applied)")
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        sys.exit(1)
