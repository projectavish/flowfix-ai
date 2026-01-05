"""
Export database tables to Excel for Power BI
Run this if ODBC connection fails
"""
import pandas as pd
from utils import execute_query

print("📊 Exporting data for Power BI...\n")

# Export directory
import os
export_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'exports')
os.makedirs(export_dir, exist_ok=True)

# 1. Tasks with all details
tasks_query = """
SELECT 
    task_id,
    task_name,
    assignee,
    status,
    priority,
    project,
    created_date,
    start_date,
    end_date,
    actual_duration,
    is_delayed,
    bottleneck_type,
    comments
FROM tasks
"""
tasks_df = execute_query(tasks_query)
excel_path = os.path.join(export_dir, 'powerbi_data.xlsx')

print(f"✓ Exporting {len(tasks_df)} tasks...")

# Create Excel file with multiple sheets
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    tasks_df.to_excel(writer, sheet_name='Tasks', index=False)
    
    # 2. Bottleneck summary
    bottleneck_query = """
    SELECT 
        assignee,
        project,
        bottleneck_type,
        COUNT(*) as count,
        AVG(actual_duration) as avg_duration
    FROM tasks
    WHERE bottleneck_type IS NOT NULL
    GROUP BY assignee, project, bottleneck_type
    """
    bottleneck_df = execute_query(bottleneck_query)
    bottleneck_df.to_excel(writer, sheet_name='Bottlenecks', index=False)
    print(f"✓ Exporting {len(bottleneck_df)} bottleneck records...")
    
    # 3. GPT suggestions
    gpt_query = """
    SELECT 
        task_id,
        root_cause,
        recommendation,
        created_at
    FROM gpt_suggestions
    """
    gpt_df = execute_query(gpt_query)
    gpt_df.to_excel(writer, sheet_name='GPT_Recommendations', index=False)
    print(f"✓ Exporting {len(gpt_df)} GPT recommendations...")
    
    # 4. Summary metrics
    metrics_query = """
    SELECT 
        COUNT(*) as total_tasks,
        AVG(actual_duration) as avg_duration,
        SUM(is_delayed) as delayed_count,
        CAST(SUM(is_delayed) AS FLOAT) / COUNT(*) * 100 as delay_percentage,
        COUNT(DISTINCT assignee) as total_assignees,
        COUNT(DISTINCT project) as total_projects
    FROM tasks
    """
    metrics_df = execute_query(metrics_query)
    metrics_df.to_excel(writer, sheet_name='Summary_Metrics', index=False)
    print(f"✓ Exporting summary metrics...")

print(f"\n✅ Excel file created: {excel_path}")
print("\n📌 Power BI Instructions:")
print("   1. Open Power BI Desktop")
print("   2. Click 'Get Data' → 'Excel'")
print(f"   3. Select file: {excel_path}")
print("   4. Check all sheets → Click 'Load'")
print("   5. Start creating visualizations!")
