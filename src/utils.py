"""
Database utilities for FlowFix AI
Handles database connections, schema creation, and common queries

Schema Version: 2.0
Last Updated: 2026-01-05
"""
import os
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Schema version for migration tracking
SCHEMA_VERSION = "2.0"

# Get absolute path to database
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'workflow_data.db')
DB_PATH = os.path.abspath(DB_PATH).replace('\\', '/')  # Convert to absolute and use forward slashes

# Check if custom DATABASE_URL is set in env, otherwise use computed path
env_db_url = os.getenv('DATABASE_URL')
if env_db_url and not env_db_url.startswith('sqlite:///data/'):
    DATABASE_URL = env_db_url  # Use custom PostgreSQL or other DB
else:
    DATABASE_URL = f'sqlite:///{DB_PATH}'  # Use computed SQLite path

logger.info(f"Database path: {DB_PATH}")

# Auto-migration flag (set to False to disable automatic schema updates)
AUTO_MIGRATE = os.getenv('AUTO_MIGRATE', 'true').lower() == 'true'


def get_engine():
    """Create and return database engine"""
    try:
        return create_engine(DATABASE_URL, echo=False)
    except Exception as e:
        logger.error(f"Failed to create database engine: {str(e)}")
        raise


def get_session():
    """Create and return database session"""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def create_schema():
    """Create database schema for FlowFix AI with FK cascades"""
    # Ensure data directory exists
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    engine = get_engine()
    
    # Enable foreign key constraints
    with engine.connect() as conn:
        conn.execute(text("PRAGMA foreign_keys = ON"))
        conn.commit()
    
    schema_sql = """
    -- Tasks table
    CREATE TABLE IF NOT EXISTS tasks (
        task_id TEXT PRIMARY KEY,
        task_name TEXT NOT NULL,
        assignee TEXT NOT NULL,
        status TEXT NOT NULL,
        created_date DATE NOT NULL,
        start_date DATE,
        end_date DATE,
        priority TEXT NOT NULL,
        comments TEXT,
        project TEXT NOT NULL,
        planned_duration INTEGER,
        actual_duration INTEGER,
        is_delayed BOOLEAN DEFAULT 0,
        is_overdue BOOLEAN DEFAULT 0,
        bottleneck_type TEXT,
        reassignment_count INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- GPT Suggestions table with enhanced tracking
    CREATE TABLE IF NOT EXISTS gpt_suggestions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL,
        suggestion_text TEXT NOT NULL,
        root_causes TEXT,
        recommendations TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        applied BOOLEAN DEFAULT 0,
        suggested_by TEXT DEFAULT 'gpt-4o-mini',
        gpt_model_used TEXT,
        prompt_version TEXT DEFAULT '1.0',
        latency_ms INTEGER,
        response_score REAL,
        sentiment TEXT,
        urgency_level TEXT,
        token_length INTEGER,
        feedback_status TEXT DEFAULT 'pending',
        feedback_notes TEXT DEFAULT '',
        feedback_date TEXT DEFAULT '',
        FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
    );
    
    -- Bottleneck History table
    CREATE TABLE IF NOT EXISTS bottleneck_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL,
        bottleneck_type TEXT NOT NULL,
        severity_score REAL,
        delay_days INTEGER,
        priority TEXT,
        root_cause_suggestion TEXT,
        detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        resolved BOOLEAN DEFAULT 0,
        resolution_time_hours REAL,
        FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
    );
    
    -- Improvement Log table with scoring
    CREATE TABLE IF NOT EXISTS improvement_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL,
        action_taken TEXT NOT NULL,
        owner TEXT,
        date_applied DATE,
        impact_measured TEXT,
        improvement_score REAL,
        kpi_before TEXT,
        kpi_after TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
    );
    
    -- Bottleneck Summary table
    CREATE TABLE IF NOT EXISTS bottleneck_summary (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL,
        bottleneck_type TEXT NOT NULL,
        severity TEXT,
        detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        resolved BOOLEAN DEFAULT 0,
        FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
    );
    
    -- Pipeline Runs table
    CREATE TABLE IF NOT EXISTS pipeline_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_type TEXT NOT NULL,
        status TEXT NOT NULL,
        tasks_processed INTEGER,
        errors TEXT,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP
    );
    
    -- ML Predictions table
    CREATE TABLE IF NOT EXISTS ml_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL,
        predicted_duration REAL,
        delay_probability REAL,
        model_version TEXT,
        features_used TEXT,
        prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        actual_outcome TEXT,
        FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
    );
    
    -- ML Training Log table
    CREATE TABLE IF NOT EXISTS ml_training_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        model_type TEXT,
        model_version TEXT,
        accuracy REAL,
        precision_score REAL,
        recall_score REAL,
        f1_score REAL,
        training_samples INTEGER,
        features_used TEXT,
        hyperparameters TEXT
    );
    
    -- Reassignment Tracking table
    CREATE TABLE IF NOT EXISTS task_reassignments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL,
        from_assignee TEXT NOT NULL,
        to_assignee TEXT NOT NULL,
        reason TEXT,
        reassignment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        effectiveness_score REAL,
        completion_improved BOOLEAN,
        FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
    );
    
    -- Ingestion Log table
    CREATE TABLE IF NOT EXISTS ingestion_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        records_added INTEGER,
        records_failed INTEGER,
        ingestion_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        error_details TEXT,
        file_size_kb REAL,
        processing_time_sec REAL
    );
    
    -- Dashboard Summary KPIs table
    CREATE TABLE IF NOT EXISTS dashboard_summary (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        metric_name TEXT NOT NULL,
        metric_value REAL,
        metric_category TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    with engine.connect() as conn:
        for statement in schema_sql.split(';'):
            if statement.strip():
                try:
                    conn.execute(text(statement))
                except Exception as e:
                    logger.warning(f"Schema statement warning: {str(e)}")
        conn.commit()
    
    logger.info("[SUCCESS] Database schema created successfully")
    logger.info(f"   Schema Version: {SCHEMA_VERSION}")


def test_db_connection():
    """Test database connection and return status"""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("[SUCCESS] Database connection successful")
        return True
    except Exception as e:
        logger.error(f"[ERROR] Database connection failed: {str(e)}")
        return False


def execute_query(query, params=None):
    """Execute a SQL query and return results as DataFrame with error handling"""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            if params:
                result = conn.execute(text(query), params)
            else:
                result = conn.execute(text(query))
            return pd.DataFrame(result.fetchall(), columns=result.keys())
    except SQLAlchemyError as e:
        logger.error(f"Query execution failed: {str(e)}")
        logger.error(f"Query: {query}")
        raise


def get_task_statistics():
    """Get basic task statistics"""
    query = """
    SELECT 
        COUNT(*) as total_tasks,
        COUNT(CASE WHEN actual_duration IS NOT NULL THEN 1 END) as completed_tasks,
        COUNT(CASE WHEN status = 'Blocked' THEN 1 END) as blocked_tasks,
        COUNT(CASE WHEN is_delayed = 1 THEN 1 END) as delayed_tasks,
        AVG(actual_duration) as avg_duration,
        COUNT(DISTINCT assignee) as total_assignees,
        COUNT(DISTINCT project) as total_projects
    FROM tasks
    """
    return execute_query(query)


def get_assignee_workload():
    """Get workload by assignee"""
    query = """
    SELECT 
        assignee,
        COUNT(*) as total_tasks,
        COUNT(CASE WHEN status != 'Done' THEN 1 END) as active_tasks,
        AVG(actual_duration) as avg_duration,
        COUNT(CASE WHEN is_delayed = 1 THEN 1 END) as delayed_tasks,
        ROUND(COUNT(CASE WHEN is_delayed = 1 THEN 1 END) * 100.0 / COUNT(*), 2) as delay_percentage
    FROM tasks
    GROUP BY assignee
    ORDER BY delayed_tasks DESC
    """
    return execute_query(query)


def get_project_metrics():
    """Get metrics by project"""
    query = """
    SELECT 
        project,
        COUNT(*) as total_tasks,
        COUNT(CASE WHEN status = 'Done' THEN 1 END) as completed,
        COUNT(CASE WHEN status = 'Blocked' THEN 1 END) as blocked,
        AVG(actual_duration) as avg_duration,
        COUNT(CASE WHEN is_delayed = 1 THEN 1 END) as delayed_tasks
    FROM tasks
    GROUP BY project
    ORDER BY total_tasks DESC
    """
    return execute_query(query)


def get_overdue_tasks():
    """Get all overdue tasks"""
    query = """
    SELECT 
        task_id,
        task_name,
        assignee,
        status,
        start_date,
        end_date,
        priority,
        project,
        actual_duration,
        bottleneck_type
    FROM tasks
    WHERE is_overdue = 1
    ORDER BY start_date
    """
    return execute_query(query)


def get_tasks_by_bottleneck():
    """Get tasks grouped by bottleneck type"""
    query = """
    SELECT 
        bottleneck_type,
        COUNT(*) as count,
        AVG(actual_duration) as avg_duration,
        GROUP_CONCAT(DISTINCT assignee) as affected_assignees
    FROM tasks
    WHERE bottleneck_type IS NOT NULL
    GROUP BY bottleneck_type
    ORDER BY count DESC
    """
    return execute_query(query)


def get_tasks_with_predictions(limit: int = 1000):
    query = f"""
    SELECT
      t.task_id,
      t.task_name,
      t.assignee,
      t.status,
      t.priority,
      t.project,
      t.actual_duration,
      t.is_delayed,
      d.prediction_value AS predicted_duration,
      p.prediction_value AS predicted_delay_probability
    FROM tasks t
    LEFT JOIN ml_predictions d
      ON t.task_id = d.task_id AND d.model_type = 'duration'
    LEFT JOIN ml_predictions p
      ON t.task_id = p.task_id AND p.model_type = 'delay_probability'
    LIMIT {int(limit)}
    """
    return execute_query(query)


def get_summary_metrics():
    """Get comprehensive summary metrics for dashboard"""
    metrics = {}
    
    try:
        # Basic task metrics
        task_stats = get_task_statistics()
        if not task_stats.empty:
            metrics['total_tasks'] = int(task_stats['total_tasks'].iloc[0])
            metrics['completed_tasks'] = int(task_stats['completed_tasks'].iloc[0])
            metrics['blocked_tasks'] = int(task_stats['blocked_tasks'].iloc[0])
            metrics['delayed_tasks'] = int(task_stats['delayed_tasks'].iloc[0])
            metrics['avg_duration'] = float(task_stats['avg_duration'].iloc[0] or 0)
            metrics['total_assignees'] = int(task_stats['total_assignees'].iloc[0])
            metrics['total_projects'] = int(task_stats['total_projects'].iloc[0])
        
        # Bottleneck metrics
        bottleneck_query = """
        SELECT 
            COUNT(DISTINCT task_id) as bottleneck_count,
            COUNT(DISTINCT bottleneck_type) as bottleneck_types
        FROM tasks
        WHERE bottleneck_type IS NOT NULL AND bottleneck_type != ''
        """
        bottleneck_stats = execute_query(bottleneck_query)
        if not bottleneck_stats.empty:
            metrics['bottleneck_count'] = int(bottleneck_stats['bottleneck_count'].iloc[0] or 0)
            metrics['bottleneck_types'] = int(bottleneck_stats['bottleneck_types'].iloc[0] or 0)
        
        # GPT suggestions
        gpt_query = "SELECT COUNT(*) as suggestion_count FROM gpt_suggestions"
        gpt_stats = execute_query(gpt_query)
        if not gpt_stats.empty:
            metrics['gpt_suggestions'] = int(gpt_stats['suggestion_count'].iloc[0] or 0)
        
        # Improvement actions
        improvement_query = "SELECT COUNT(*) as improvement_count FROM improvement_log"
        improvement_stats = execute_query(improvement_query)
        if not improvement_stats.empty:
            metrics['improvements_logged'] = int(improvement_stats['improvement_count'].iloc[0] or 0)
        
        # Calculate rates
        if metrics.get('total_tasks', 0) > 0:
            metrics['delay_rate'] = round((metrics.get('delayed_tasks', 0) / metrics['total_tasks']) * 100, 2)
            metrics['completion_rate'] = round((metrics.get('completed_tasks', 0) / metrics['total_tasks']) * 100, 2)
            metrics['bottleneck_rate'] = round((metrics.get('bottleneck_count', 0) / metrics['total_tasks']) * 100, 2)
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error getting summary metrics: {str(e)}")
        return {}


def update_dashboard_kpis():
    """Update dashboard summary table with latest KPIs"""
    try:
        metrics = get_summary_metrics()
        engine = get_engine()
        
        with engine.connect() as conn:
            conn.execute(text("""
              CREATE TABLE IF NOT EXISTS dashboard_summary (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              metric_name TEXT NOT NULL,
              metric_value REAL,
              metric_category TEXT,
              updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""))
            # Clear existing metrics
            conn.execute(text("DELETE FROM dashboard_summary"))
            
            # Insert new metrics
            for metric_name, metric_value in metrics.items():
                count_keys = {
                    "total_tasks", "completed_tasks", "blocked_tasks", "delayed_tasks",
                    "total_assignees", "total_projects",
                    "bottleneck_count", "bottleneck_types",
                    "gpt_suggestions", "improvements_logged"
}
                category = "count" if metric_name in count_keys else "rate"
                conn.execute(text("""
                    INSERT INTO dashboard_summary (metric_name, metric_value, metric_category)
                    VALUES (:name, :value, :category)
                """), {
                    'name': metric_name,
                    'value': float(metric_value),
                    'category': category
                })
            
            conn.commit()
        
        logger.info(f"[SUCCESS] Updated {len(metrics)} KPIs in dashboard_summary table")
        return True
    
    except Exception as e:
        logger.error(f"Failed to update dashboard KPIs: {str(e)}")
        return False


if __name__ == "__main__":
    # Test database utilities
    print("\n" + "="*60)
    print("Testing Database Utilities")
    print("="*60 + "\n")
    
    # Test connection
    test_db_connection()
    
    # Create schema
    create_schema()
    
    # Get summary metrics
    metrics = get_summary_metrics()
    print("\n[STATS] Summary Metrics:")
    for key, value in metrics.items():
        print(f"   {key}: {value}")
    
    print("\n[SUCCESS] Database utilities module ready")
