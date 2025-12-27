"""
Database utilities for FlowFix AI
Handles database connections, schema creation, and common queries
"""
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

# Get absolute path to database
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'workflow_data.db')
DB_PATH = os.path.abspath(DB_PATH).replace('\\', '/')  # Convert to absolute and use forward slashes

# Check if custom DATABASE_URL is set in env, otherwise use computed path
env_db_url = os.getenv('DATABASE_URL')
if env_db_url and not env_db_url.startswith('sqlite:///data/'):
    DATABASE_URL = env_db_url  # Use custom PostgreSQL or other DB
else:
    DATABASE_URL = f'sqlite:///{DB_PATH}'  # Use computed SQLite path

print(f"Database path: {DB_PATH}")  # Debug print


def get_engine():
    """Create and return database engine"""
    return create_engine(DATABASE_URL, echo=False)


def get_session():
    """Create and return database session"""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def create_schema():
    """Create database schema for FlowFix AI"""
    # Ensure data directory exists
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    engine = get_engine()
    
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
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- GPT Suggestions table
    CREATE TABLE IF NOT EXISTS gpt_suggestions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL,
        suggestion_text TEXT NOT NULL,
        root_causes TEXT,
        recommendations TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        applied BOOLEAN DEFAULT 0,
        FOREIGN KEY (task_id) REFERENCES tasks(task_id)
    );
    
    -- Improvement Log table
    CREATE TABLE IF NOT EXISTS improvement_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL,
        action_taken TEXT NOT NULL,
        owner TEXT,
        date_applied DATE,
        impact_measured TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (task_id) REFERENCES tasks(task_id)
    );
    
    -- Bottleneck Summary table
    CREATE TABLE IF NOT EXISTS bottleneck_summary (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL,
        bottleneck_type TEXT NOT NULL,
        severity TEXT,
        detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        resolved BOOLEAN DEFAULT 0,
        FOREIGN KEY (task_id) REFERENCES tasks(task_id)
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
    """
    
    with engine.connect() as conn:
        for statement in schema_sql.split(';'):
            if statement.strip():
                conn.execute(text(statement))
        conn.commit()
    
    print("✅ Database schema created successfully")


def execute_query(query, params=None):
    """Execute a SQL query and return results as DataFrame"""
    engine = get_engine()
    with engine.connect() as conn:
        if params:
            result = conn.execute(text(query), params)
        else:
            result = conn.execute(text(query))
        return pd.DataFrame(result.fetchall(), columns=result.keys())


def get_task_statistics():
    """Get basic task statistics"""
    query = """
    SELECT 
        COUNT(*) as total_tasks,
        COUNT(CASE WHEN status = 'Done' THEN 1 END) as completed_tasks,
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


if __name__ == "__main__":
    # Test database utilities
    print("Testing database utilities...")
    create_schema()
    print("\n✅ Database utilities module ready")
