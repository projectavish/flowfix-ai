"""
Feedback Loop System for FlowFix AI
Tracks which suggestions are applied/rejected and measures impact
"""
from datetime import datetime
from utils import execute_query, get_engine
from sqlalchemy import text


def mark_suggestion_feedback(task_id, status='applied', notes=''):
    """
    Mark a GPT suggestion as applied or rejected
    
    Args:
        task_id: The task ID
        status: 'applied', 'rejected', 'pending'
        notes: Optional notes about the feedback
    """
    engine = get_engine()
    
    with engine.connect() as conn:
        # Update gpt_suggestions table
        query = text("""
            UPDATE gpt_suggestions
            SET feedback_status = :status,
                feedback_notes = :notes,
                feedback_date = :date
            WHERE task_id = :task_id
        """)
        
        conn.execute(query, {
            'status': status,
            'notes': notes,
            'date': datetime.now().isoformat(),
            'task_id': task_id
        })
        conn.commit()
    
    print(f"✅ Marked suggestion for task {task_id} as {status}")



def add_feedback_columns():
    """Add feedback tracking columns to gpt_suggestions table"""
    engine = get_engine()
    
    with engine.connect() as conn:
        # Check if columns already exist
        result = conn.execute(text("PRAGMA table_info(gpt_suggestions)"))
        columns = [row[1] for row in result.fetchall()]
        
        if 'feedback_status' not in columns:
            print("Adding feedback columns to database...")
            
            conn.execute(text("""
                ALTER TABLE gpt_suggestions 
                ADD COLUMN feedback_status TEXT DEFAULT 'pending'
            """))
            
            conn.execute(text("""
                ALTER TABLE gpt_suggestions 
                ADD COLUMN feedback_notes TEXT DEFAULT ''
            """))
            
            conn.execute(text("""
                ALTER TABLE gpt_suggestions 
                ADD COLUMN feedback_date TEXT DEFAULT ''
            """))
            
            conn.commit()
            print("✅ Feedback columns added")
        else:
            print("Feedback columns already exist")



def get_feedback_summary():
    """Get summary of suggestion feedback"""
    query = """
    SELECT 
        feedback_status,
        COUNT(*) as count
    FROM gpt_suggestions
    GROUP BY feedback_status
    """
    
    df = execute_query(query)
    
    print("\n" + "="*60)
    print("FEEDBACK SUMMARY")
    print("="*60)
    
    for idx, row in df.iterrows():
        print(f"{row['feedback_status'].title()}: {int(row['count'])} suggestions")
    
    return df


def get_applied_suggestions():
    """Get all applied suggestions with details"""
    query = """
    SELECT 
        gs.task_id,
        gs.recommendations,
        gs.feedback_date,
        gs.feedback_notes,
        t.assignee,
        t.priority,
        t.status
    FROM gpt_suggestions gs
    JOIN tasks t ON gs.task_id = t.task_id
    WHERE gs.feedback_status = 'applied'
    ORDER BY gs.feedback_date DESC
    """
    
    df = execute_query(query)
    
    if len(df) == 0:
        print("\nNo suggestions have been marked as applied yet.")
        return df
    
    print("\n" + "="*60)
    print("APPLIED SUGGESTIONS")
    print("="*60 + "\n")
    
    for idx, row in df.iterrows():
        print(f"Task: {row['task_id']}")
        print(f"  Assignee: {row['assignee']}")
        print(f"  Applied: {row['feedback_date'][:10]}")
        print(f"  Recommendation: {row['recommendations'][:100]}...")
        if row['feedback_notes']:
            print(f"  Notes: {row['feedback_notes']}")
        print()
    
    return df


def simulate_feedback_for_demo():
    """Simulate feedback for demo purposes"""
    print("\n" + "="*60)
    print("SIMULATING FEEDBACK FOR DEMO")
    print("="*60 + "\n")
    
    # Add columns if needed
    add_feedback_columns()
    
    # Get all suggestions
    suggestions_query = "SELECT task_id FROM gpt_suggestions ORDER BY id LIMIT 6"
    suggestions = execute_query(suggestions_query)
    
    if len(suggestions) == 0:
        print("No suggestions found. Run GPT suggester first.")
        return
    
    # Mark some as applied, some as pending
    statuses = ['applied', 'applied', 'pending', 'applied', 'pending', 'rejected']
    notes_list = [
        'Reassigned task, reduced duration by 3 days',
        'Cleared blocker, task completed',
        'Under review',
        'Increased team capacity',
        'Evaluating options',
        'Resource not available'
    ]
    
    for idx, row in suggestions.iterrows():
        if idx < len(statuses):
            status = statuses[idx]
            notes = notes_list[idx]
            mark_suggestion_feedback(row['task_id'], status, notes)
    
    print("\n✅ Feedback simulation complete")
    
    # Show summary
    get_feedback_summary()


def measure_feedback_impact():
    """Measure the impact of applied suggestions"""
    print("\n" + "="*60)
    print("MEASURING FEEDBACK IMPACT")
    print("="*60 + "\n")
    
    query = """
    SELECT 
        COUNT(*) as applied_count,
        AVG(CASE WHEN t.status = 'Completed' THEN 1 ELSE 0 END) as completion_rate
    FROM gpt_suggestions gs
    JOIN tasks t ON gs.task_id = t.task_id
    WHERE gs.feedback_status = 'applied'
    """
    
    result = execute_query(query)
    
    if len(result) == 0 or result.iloc[0]['applied_count'] == 0:
        print("Not enough applied suggestions to measure impact.")
        return
    
    applied = int(result.iloc[0]['applied_count'])
    completion = result.iloc[0]['completion_rate'] * 100
    
    print(f"Applied Suggestions: {applied}")
    print(f"Completion Rate: {completion:.1f}%")
    print(f"\nEstimated Impact: Suggestions have been applied to {applied} tasks")


def create_feedback_report():
    """Generate comprehensive feedback report"""
    print("\n" + "="*60)
    print("FEEDBACK LOOP REPORT")
    print("="*60 + "\n")
    
    # Summary
    get_feedback_summary()
    
    # Applied suggestions
    get_applied_suggestions()
    
    # Impact measurement
    measure_feedback_impact()


if __name__ == "__main__":
    # Initialize feedback system
    add_feedback_columns()
    
    # Simulate feedback for demo
    simulate_feedback_for_demo()
    
    # Generate report
    create_feedback_report()
