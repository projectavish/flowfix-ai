"""
Fix database data types and NULLs for Power BI compatibility
"""
import sqlite3
import os

# Get database path
db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'workflow_data.db')
db_path = db_path.replace('\\', '/')

print(f"Fixing database: {db_path}\n")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Fix NULL values and data types in tasks table
print("Fixing tasks table...")
cursor.execute("""
    UPDATE tasks 
    SET 
        task_name = COALESCE(task_name, ''),
        assignee = COALESCE(assignee, 'Unassigned'),
        status = COALESCE(status, 'Unknown'),
        priority = COALESCE(priority, 'Medium'),
        project = COALESCE(project, 'Unknown'),
        comments = COALESCE(comments, ''),
        bottleneck_type = COALESCE(bottleneck_type, '')
    WHERE task_name IS NULL 
       OR assignee IS NULL 
       OR status IS NULL 
       OR priority IS NULL 
       OR project IS NULL
""")
print(f"  Updated {cursor.rowcount} rows")

# Fix NULL values in gpt_suggestions
print("\nFixing gpt_suggestions table...")
cursor.execute("""
    UPDATE gpt_suggestions 
    SET 
        root_causes = COALESCE(root_causes, ''),
        recommendations = COALESCE(recommendations, '')
    WHERE root_causes IS NULL OR recommendations IS NULL
""")
print(f"  Updated {cursor.rowcount} rows")

# Fix NULL values in bottleneck_summary
print("\nFixing bottleneck_summary table...")
cursor.execute("""
    UPDATE bottleneck_summary 
    SET 
        bottleneck_type = COALESCE(bottleneck_type, 'Unknown')
    WHERE bottleneck_type IS NULL
""")
print(f"  Updated {cursor.rowcount} rows")

# Commit changes
conn.commit()

# Vacuum database
print("\nOptimizing database...")
cursor.execute("VACUUM")

# Verify tables
print("\nVerifying tables:")
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
for table in tables:
    cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
    count = cursor.fetchone()[0]
    print(f"  ✓ {table[0]}: {count} rows")

conn.close()
print("\n✅ Database fixed and ready for Power BI!")
