"""
Task Reassignment Tracking for FlowFix AI
Tracks task reassignments and analyzes impact on performance
"""
from datetime import datetime
from utils import get_engine, execute_query
from sqlalchemy import text


def create_reassignment_table():
    """Create table to track task reassignments"""
    engine = get_engine()
    
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS task_reassignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                from_assignee TEXT NOT NULL,
                to_assignee TEXT NOT NULL,
                reassignment_date TEXT NOT NULL,
                reason TEXT,
                status_before TEXT,
                status_after TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (task_id) REFERENCES tasks(task_id)
            )
        """))
        
        conn.commit()
    
    print("✅ Reassignment tracking table created")


def record_reassignment(task_id, from_assignee, to_assignee, reason=''):
    """
    Record a task reassignment
    
    Args:
        task_id: The task being reassigned
        from_assignee: Original assignee
        to_assignee: New assignee
        reason: Reason for reassignment
    """
    engine = get_engine()
    
    with engine.connect() as conn:
        # Get current task status
        result = conn.execute(text("SELECT status FROM tasks WHERE task_id = :task_id"), 
                            {'task_id': task_id})
        row = result.fetchone()
        status_before = row[0] if row else 'Unknown'
        
        # Insert reassignment record
        conn.execute(text("""
            INSERT INTO task_reassignments 
            (task_id, from_assignee, to_assignee, reassignment_date, reason, status_before)
            VALUES (:task_id, :from_assignee, :to_assignee, :date, :reason, :status_before)
        """), {
            'task_id': task_id,
            'from_assignee': from_assignee,
            'to_assignee': to_assignee,
            'date': datetime.now().isoformat(),
            'reason': reason,
            'status_before': status_before
        })
        
        # Update task assignee
        conn.execute(text("""
            UPDATE tasks 
            SET assignee = :assignee
            WHERE task_id = :task_id
        """), {
            'assignee': to_assignee,
            'task_id': task_id
        })
        
        conn.commit()
    
    print(f"✅ Task {task_id} reassigned: {from_assignee} → {to_assignee}")


def get_reassignment_history(task_id=None):
    """Get reassignment history for a task or all tasks"""
    if task_id:
        query = """
        SELECT 
            task_id,
            from_assignee,
            to_assignee,
            reassignment_date,
            reason
        FROM task_reassignments
        WHERE task_id = ?
        ORDER BY reassignment_date DESC
        """
        df = execute_query(query, params=(task_id,))
    else:
        query = """
        SELECT 
            task_id,
            from_assignee,
            to_assignee,
            reassignment_date,
            reason
        FROM task_reassignments
        ORDER BY reassignment_date DESC
        """
        df = execute_query(query)
    
    return df


def get_reassignment_statistics():
    """Get overall reassignment statistics"""
    print("\n" + "="*60)
    print("TASK REASSIGNMENT STATISTICS")
    print("="*60 + "\n")
    
    # Total reassignments
    total_query = "SELECT COUNT(*) as total FROM task_reassignments"
    total = execute_query(total_query).iloc[0]['total']
    
    print(f"Total Reassignments: {int(total)}")
    
    if total == 0:
        print("\nNo reassignments recorded yet.")
        return
    
    # Reassignments by assignee (who received most reassignments)
    received_query = """
    SELECT 
        to_assignee,
        COUNT(*) as received_count
    FROM task_reassignments
    GROUP BY to_assignee
    ORDER BY received_count DESC
    LIMIT 5
    """
    received = execute_query(received_query)
    
    print("\nTop Assignees (Received Tasks):")
    for idx, row in received.iterrows():
        print(f"  {row['to_assignee']}: {int(row['received_count'])} tasks")
    
    # Reassignments by assignee (who gave away most tasks)
    given_query = """
    SELECT 
        from_assignee,
        COUNT(*) as given_count
    FROM task_reassignments
    GROUP BY from_assignee
    ORDER BY given_count DESC
    LIMIT 5
    """
    given = execute_query(given_query)
    
    print("\nTop Assignees (Gave Away Tasks):")
    for idx, row in given.iterrows():
        print(f"  {row['from_assignee']}: {int(row['given_count'])} tasks")
    
    # Reasons for reassignment
    reason_query = """
    SELECT 
        reason,
        COUNT(*) as count
    FROM task_reassignments
    WHERE reason != ''
    GROUP BY reason
    ORDER BY count DESC
    """
    reasons = execute_query(reason_query)
    
    if len(reasons) > 0:
        print("\nReassignment Reasons:")
        for idx, row in reasons.iterrows():
            print(f"  {row['reason']}: {int(row['count'])} times")


def simulate_reassignments_for_demo():
    """Simulate reassignments based on bottleneck data"""
    print("\n" + "="*60)
    print("SIMULATING REASSIGNMENTS (Demo Mode)")
    print("="*60 + "\n")
    
    # Create table
    create_reassignment_table()
    
    # Get bottleneck tasks that could benefit from reassignment
    query = """
    SELECT 
        task_id,
        assignee,
        bottleneck_type
    FROM tasks
    WHERE bottleneck_type != ''
    AND bottleneck_type LIKE '%Assignee_Bottleneck%'
    LIMIT 5
    """
    
    bottlenecks = execute_query(query)
    
    if len(bottlenecks) == 0:
        print("No assignee bottlenecks found for simulation.")
        return
    
    # Get assignees with lower workload
    low_workload_query = """
    SELECT 
        assignee,
        COUNT(*) as task_count
    FROM tasks
    WHERE status != 'Completed'
    GROUP BY assignee
    ORDER BY task_count ASC
    LIMIT 5
    """
    
    low_workload = execute_query(low_workload_query)
    
    if len(low_workload) == 0:
        print("No available assignees for reassignment.")
        return
    
    # Simulate reassignments
    reasons = [
        'Overloaded assignee - rebalancing workload',
        'Better expertise match',
        'Assignee bottleneck detected by ML model',
        'Improved task priority alignment',
        'Resource capacity optimization'
    ]
    
    for idx, row in bottlenecks.iterrows():
        if idx < len(low_workload):
            new_assignee = low_workload.iloc[idx]['assignee']
            reason = reasons[idx % len(reasons)]
            
            record_reassignment(
                task_id=row['task_id'],
                from_assignee=row['assignee'],
                to_assignee=new_assignee,
                reason=reason
            )
    
    print("\n✅ Reassignment simulation complete")


def analyze_reassignment_impact():
    """Analyze the impact of reassignments on task completion"""
    print("\n" + "="*60)
    print("REASSIGNMENT IMPACT ANALYSIS")
    print("="*60 + "\n")
    
    # Get tasks that were reassigned and later completed
    query = """
    SELECT 
        tr.task_id,
        tr.from_assignee,
        tr.to_assignee,
        t.status,
        t.actual_duration,
        t.is_delayed
    FROM task_reassignments tr
    JOIN tasks t ON tr.task_id = t.task_id
    """
    
    df = execute_query(query)
    
    if len(df) == 0:
        print("No reassignment data available for analysis.")
        return
    
    completed = len(df[df['status'] == 'Completed'])
    in_progress = len(df[df['status'] == 'In Progress'])
    delayed = df['is_delayed'].sum()
    
    print(f"Reassigned Tasks Status:")
    print(f"  Completed: {completed} ({completed/len(df)*100:.1f}%)")
    print(f"  In Progress: {in_progress} ({in_progress/len(df)*100:.1f}%)")
    print(f"  Delayed: {int(delayed)} ({delayed/len(df)*100:.1f}%)")
    
    if completed > 0:
        avg_duration = df[df['status'] == 'Completed']['actual_duration'].mean()
        print(f"\nAvg Duration (Completed): {avg_duration:.1f} days")


def generate_reassignment_report():
    """Generate comprehensive reassignment report"""
    print("\n" + "="*60)
    print("TASK REASSIGNMENT REPORT")
    print("="*60)
    
    get_reassignment_statistics()
    analyze_reassignment_impact()
    
    # Show recent reassignments
    print("\n" + "="*60)
    print("RECENT REASSIGNMENTS")
    print("="*60 + "\n")
    
    recent = get_reassignment_history()
    
    if len(recent) > 0:
        for idx, row in recent.head(10).iterrows():
            print(f"Task: {row['task_id']}")
            print(f"  {row['from_assignee']} → {row['to_assignee']}")
            print(f"  Date: {row['reassignment_date'][:10]}")
            if row['reason']:
                print(f"  Reason: {row['reason']}")
            print()
    else:
        print("No reassignments recorded yet.")


if __name__ == "__main__":
    # Initialize and simulate for demo
    simulate_reassignments_for_demo()
    
    # Generate report
    generate_reassignment_report()
