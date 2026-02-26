"""
Task Reassignment Tracking for FlowFix AI - Production Grade
Tracks reassignments with bottleneck/GPT/ML integration and effectiveness analysis
"""
import logging
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import text
<<<<<<< HEAD
from src.utils import get_engine, execute_query
=======
from utils import get_engine, execute_query
>>>>>>> 789db11de11bf607177a31557cbb9b376ebcdde5

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def record_reassignment(task_id, old_assignee, new_assignee, reason='', triggered_by='manual'):
    """
    Record a task reassignment with enhanced tracking
    
    Args:
        task_id: The task being reassigned
        old_assignee: Original assignee
        new_assignee: New assignee
        reason: Reason for reassignment
        triggered_by: What triggered this (manual, bottleneck, ml_prediction, gpt_suggestion)
    
    Returns:
        bool: Success status
    """
    engine = get_engine()
    
    try:
        with engine.connect() as conn:
            # Get current task details
            result = conn.execute(
                text("SELECT status, actual_duration, is_delayed FROM tasks WHERE task_id = :task_id"), 
                {'task_id': task_id}
            )
            row = result.fetchone()
            
            if not row:
                logger.error(f"Task {task_id} not found")
                return False
            
            status_before = row[0]
            duration_before = row[1]
            was_delayed = row[2]
            
            # Insert reassignment record
            conn.execute(text("""
                INSERT INTO task_reassignments 
                (task_id, from_assignee, to_assignee, reason, 
                 status_before, status_after, triggered_by, was_delayed_before, 
                 duration_before_reassignment, reassignment_date)
                VALUES (:task_id, :from_assignee, :to_assignee, :reason,
                        :status_before, :status_after, :triggered_by, :was_delayed, :duration_before, :reassignment_date)
            """), {
                'task_id': task_id,
                'from_assignee': old_assignee,
                'to_assignee': new_assignee,
                'reason': reason,
                'status_before': status_before,
                'status_after': '',
                'triggered_by': triggered_by,
                'was_delayed': was_delayed,
                'duration_before': duration_before,
                'reassignment_date': datetime.now().isoformat()
            })
                'duration_before': duration_before,
                'was_delayed': was_delayed,
                'triggered_by': triggered_by,
                'reassigned_at': datetime.now().isoformat()
            })
            
            # Update task assignee and increment reassignment count
            conn.execute(text("""
                UPDATE tasks 
                SET assignee = :assignee,
                    reassignment_count = COALESCE(reassignment_count, 0) + 1
                WHERE task_id = :task_id
            """), {
                'assignee': new_assignee,
                'task_id': task_id
            })
            
            conn.commit()
        
        logger.info(f"[SUCCESS] Task {task_id} reassigned: {old_assignee} → {new_assignee}")
        return True
    
    except Exception as e:
        logger.error(f"Error recording reassignment: {e}")
        return False


def suggest_reassignment_from_bottleneck(task_id, bottleneck_type, assignee):
    """
    Suggest reassignment when assignee bottleneck is detected
    
    Triggered by bottleneck_detector when Assignee_Bottleneck is found
    
    Args:
        task_id: Task with bottleneck
        bottleneck_type: Type of bottleneck detected
        assignee: Current assignee with bottleneck
        
    Returns:
        dict: Suggestion with new_assignee and reason
    """
    logger.info(f"[INFO] Analyzing reassignment options for {task_id} (bottleneck: {bottleneck_type})")
    
    # Find assignees with lower workload and better performance
    query = text("""
        SELECT 
            assignee,
            COUNT(*) as active_tasks,
            AVG(CASE WHEN is_delayed = 1 THEN 1 ELSE 0 END) as delay_rate,
            AVG(actual_duration) as avg_duration
        FROM tasks
        WHERE status != 'Completed' 
          AND status != 'Cancelled'
          AND assignee != :current_assignee
        GROUP BY assignee
        HAVING COUNT(*) < (
            SELECT AVG(task_count) 
            FROM (
                SELECT COUNT(*) as task_count 
                FROM tasks 
                WHERE status != 'Completed' 
                GROUP BY assignee
            )
        )
        ORDER BY delay_rate ASC, active_tasks ASC
        LIMIT 3
    """)
    
    try:
        df = execute_query(query, params={'current_assignee': assignee})
        
        if len(df) == 0:
            logger.warning("No suitable reassignment candidates found")
            return None
        
        # Select best candidate (lowest delay rate and workload)
        best_candidate = df.iloc[0]
        
        suggestion = {
            'task_id': task_id,
            'current_assignee': assignee,
            'suggested_assignee': best_candidate['assignee'],
            'reason': f"Assignee bottleneck detected. {best_candidate['assignee']} has lower workload ({int(best_candidate['active_tasks'])} active tasks) and better delay rate ({best_candidate['delay_rate']:.1%})",
            'confidence': 'high' if best_candidate['delay_rate'] < 0.3 else 'medium'
        }
        
        logger.info(f"[SUGGESTION] Suggestion: Reassign to {suggestion['suggested_assignee']} ({suggestion['confidence']} confidence)")
        
        return suggestion
    
    except Exception as e:
        logger.error(f"Error suggesting reassignment: {e}")
        return None


def auto_reassign_high_delay_predictions(threshold_probability=0.7):
    """
    Auto-trigger reassignment for tasks with high delay prediction
    
    Integrates with ML predictor - if delay probability > threshold, suggest reassignment
    
    Args:
        threshold_probability: Trigger reassignment if P(delay) > this value
        
    Returns:
        list: Tasks that were reassigned
    """
    logger.info(f"[ML] Checking ML predictions for high-risk tasks (threshold: {threshold_probability})")
    
    # Get recent predictions with high delay probability
    query = text("""
        SELECT 
            mp.task_id,
            t.assignee,
            t.task_name,
            t.priority,
            mp.prediction_value,
            mp.confidence_score
        FROM ml_predictions mp
        JOIN tasks t ON mp.task_id = t.task_id
        WHERE mp.model_type = 'delay'
          AND mp.prediction_value >= :threshold
          AND t.status != 'Completed'
          AND t.status != 'Cancelled'
          AND NOT EXISTS (
              SELECT 1 FROM task_reassignments tr
              WHERE tr.task_id = mp.task_id
                AND tr.triggered_by = 'ml_prediction'
                AND tr.reassigned_at > datetime('now', '-7 days')
          )
        ORDER BY mp.prediction_value DESC, t.priority DESC
        LIMIT 5
    """)
    
    try:
        df = execute_query(query, params={'threshold': threshold_probability})
        
        if len(df) == 0:
            logger.info("No high-risk tasks found for reassignment")
            return []
        
        logger.info(f"Found {len(df)} high-risk tasks for reassignment")
        
        reassigned = []
        
        for _, task in df.iterrows():
            # Get reassignment suggestion
            suggestion = suggest_reassignment_from_bottleneck(
                task['task_id'], 
                'ML_Predicted_Delay', 
                task['assignee']
            )
            
            if suggestion:
                # Record reassignment
                success = record_reassignment(
                    task_id=task['task_id'],
                    old_assignee=task['assignee'],
                    new_assignee=suggestion['suggested_assignee'],
                    reason=f"ML predicted {task['prediction_value']:.1%} delay probability. {suggestion['reason']}",
                    triggered_by='ml_prediction'
                )
                
                if success:
                    reassigned.append(task['task_id'])
        
        logger.info(f"[SUCCESS] Auto-reassigned {len(reassigned)} tasks based on ML predictions")
        return reassigned
    
    except Exception as e:
        logger.error(f"Error in auto-reassignment: {e}")
        return []


def get_reassignment_history(task_id=None, limit=50):
    """Get reassignment history with enhanced fields"""
    if task_id:
        query = text("""
            SELECT 
                tr.task_id,
                t.task_name,
                tr.old_assignee,
                tr.new_assignee,
                tr.reason,
                tr.triggered_by,
                tr.reassigned_at,
                tr.status_before_reassignment,
                tr.was_delayed_before,
                t.status as current_status,
                t.is_delayed as currently_delayed,
                t.actual_duration as current_duration
            FROM task_reassignments tr
            JOIN tasks t ON tr.task_id = t.task_id
            WHERE tr.task_id = :task_id
            ORDER BY tr.reassigned_at DESC
        """)
        return execute_query(query, params={'task_id': task_id})
    else:
        query = f"""
            SELECT 
                tr.task_id,
                t.task_name,
                tr.old_assignee,
                tr.new_assignee,
                tr.reason,
                tr.triggered_by,
                tr.reassigned_at,
                tr.status_before_reassignment,
                tr.was_delayed_before,
                t.status as current_status,
                t.is_delayed as currently_delayed
            FROM task_reassignments tr
            JOIN tasks t ON tr.task_id = t.task_id
            ORDER BY tr.reassigned_at DESC
            LIMIT {limit}
        """
        return execute_query(query)


def weekly_workload_rebalancing():
    """
    Weekly script to rebalance workload across assignees
    
    Identifies overloaded assignees and suggests redistributing tasks
    Can be run as CRON job or scheduled task
    """
    logger.info("\n" + "="*60)
    logger.info("[INFO] WEEKLY WORKLOAD REBALANCING")
    logger.info("="*60 + "\n")
    
    # Get current workload distribution
    query = """
        SELECT 
            assignee,
            COUNT(*) as active_tasks,
            AVG(actual_duration) as avg_duration,
            SUM(CASE WHEN is_delayed = 1 THEN 1 ELSE 0 END) as delayed_count,
            AVG(CASE WHEN is_delayed = 1 THEN 1.0 ELSE 0.0 END) as delay_rate
        FROM tasks
        WHERE status != 'Completed' AND status != 'Cancelled'
        GROUP BY assignee
        ORDER BY active_tasks DESC
    """
    
    workload = execute_query(query)
    
    if len(workload) < 2:
        logger.info("Not enough assignees for rebalancing")
        return
    
    # Calculate statistics
    avg_tasks = workload['active_tasks'].mean()
    std_tasks = workload['active_tasks'].std()
    
    logger.info(f"Current Workload Statistics:")
    logger.info(f"   Avg tasks per assignee: {avg_tasks:.1f}")
    logger.info(f"   Std deviation: {std_tasks:.1f}\n")
    
    # Identify overloaded assignees (> mean + 1 std)
    threshold = avg_tasks + std_tasks
    overloaded = workload[workload['active_tasks'] > threshold]
    
    if len(overloaded) == 0:
        logger.info("[SUCCESS] Workload is well balanced")
        return
    
    logger.info(f"Overloaded Assignees (>{threshold:.1f} tasks):")
    for _, row in overloaded.iterrows():
        logger.info(f"   {row['assignee']}: {int(row['active_tasks'])} tasks (delay rate: {row['delay_rate']:.1%})")
    
    # Get underloaded assignees
    underloaded = workload[workload['active_tasks'] < avg_tasks]
    
    if len(underloaded) == 0:
        logger.info("\nNo available assignees for task redistribution")
        return
    
    logger.info(f"\nUnderloaded Assignees (<{avg_tasks:.1f} tasks):")
    for _, row in underloaded.iterrows():
        logger.info(f"   {row['assignee']}: {int(row['active_tasks'])} tasks")
    
    # Suggest reassignments
    logger.info("\n[INFO] Reassignment Suggestions:")
    
    reassignment_count = 0
    
    for _, overloaded_person in overloaded.iterrows():
        # Get their non-critical tasks
        tasks_query = text("""
            SELECT task_id, task_name, priority, status
            FROM tasks
            WHERE assignee = :assignee
              AND status != 'Completed'
              AND status != 'Cancelled'
              AND priority != 'High'
            ORDER BY 
                CASE priority 
                    WHEN 'Low' THEN 1 
                    WHEN 'Medium' THEN 2 
                    ELSE 3 
                END
            LIMIT 3
        """)
        
        tasks_to_move = execute_query(tasks_query, params={'assignee': overloaded_person['assignee']})
        
        if len(tasks_to_move) == 0:
            continue
        
        # Assign to least loaded person
        target_assignee = underloaded.iloc[0]['assignee']
        
        for _, task in tasks_to_move.iterrows():
            logger.info(f"   Task {task['task_id']}: {overloaded_person['assignee']} → {target_assignee}")
            logger.info(f"      Priority: {task['priority']}, Status: {task['status']}")
            
            # Record suggestion (not auto-executing for safety)
            reassignment_count += 1
    
    logger.info(f"\n[SUGGESTION] Total suggestions: {reassignment_count}")
    logger.info("[WARNING] Review and approve manually before applying\n")


def calculate_reassignment_effectiveness():
    """
    Calculate effectiveness of reassignments
    
    Compares task performance before/after reassignment
    
    Returns:
        dict: Effectiveness metrics
    """
    logger.info("\n" + "="*60)
    logger.info("[STATS] REASSIGNMENT EFFECTIVENESS ANALYSIS")
    logger.info("="*60 + "\n")
    
    # Get completed tasks that were reassigned
    query = """
        SELECT 
            tr.task_id,
            tr.was_delayed_before,
            t.is_delayed as is_delayed_after,
            tr.duration_before_reassignment,
            t.actual_duration as duration_after,
            tr.triggered_by
        FROM task_reassignments tr
        JOIN tasks t ON tr.task_id = t.task_id
        WHERE t.status = 'Completed'
          AND tr.duration_before_reassignment IS NOT NULL
    """
    
    df = execute_query(query)
    
    if len(df) == 0:
        logger.info("No completed reassigned tasks to analyze")
        return None
    
    # Calculate metrics
    total_reassignments = len(df)
    
    # Delay improvement
    was_delayed_before = df['was_delayed_before'].sum()
    is_delayed_after = df['is_delayed_after'].sum()
    delay_improvement = was_delayed_before - is_delayed_after
    
    # Duration improvement
    duration_before_avg = df['duration_before_reassignment'].mean()
    duration_after_avg = df['duration_after'].mean()
    duration_change = duration_after_avg - duration_before_avg
    
    logger.info(f"Total Completed Reassignments: {total_reassignments}")
    logger.info(f"\nDelay Status:")
    logger.info(f"   Before: {int(was_delayed_before)} delayed ({was_delayed_before/total_reassignments*100:.1f}%)")
    logger.info(f"   After: {int(is_delayed_after)} delayed ({is_delayed_after/total_reassignments*100:.1f}%)")
    logger.info(f"   Improvement: {int(delay_improvement)} fewer delays")
    
    logger.info(f"\nDuration:")
    logger.info(f"   Before: {duration_before_avg:.1f} days")
    logger.info(f"   After: {duration_after_avg:.1f} days")
    logger.info(f"   Change: {duration_change:+.1f} days")
    
    # Effectiveness by trigger type
    logger.info(f"\nBy Trigger Type:")
    by_trigger = df.groupby('triggered_by').agg({
        'task_id': 'count',
        'is_delayed_after': 'sum'
    }).rename(columns={'task_id': 'count', 'is_delayed_after': 'delayed_count'})
    
    for trigger, row in by_trigger.iterrows():
        delay_rate = row['delayed_count'] / row['count']
        logger.info(f"   {trigger}: {int(row['count'])} tasks, {delay_rate:.1%} delay rate")
    
    metrics = {
        'total_reassignments': int(total_reassignments),
        'delay_improvement': int(delay_improvement),
        'duration_change_days': float(duration_change),
        'effectiveness_score': float((delay_improvement / was_delayed_before * 100) if was_delayed_before > 0 else 0)
    }
    
    logger.info(f"\n[SUCCESS] Overall Effectiveness Score: {metrics['effectiveness_score']:.1f}%\n")
    
    return metrics


def get_reassignment_statistics():
    """Get comprehensive reassignment statistics"""
    logger.info("\n" + "="*60)
    logger.info("[STATS] REASSIGNMENT STATISTICS")
    logger.info("="*60 + "\n")
    
    # Total reassignments
    total_query = "SELECT COUNT(*) as total FROM task_reassignments"
    total = execute_query(total_query).iloc[0]['total']
    
    logger.info(f"Total Reassignments: {int(total)}")
    
    if total == 0:
        logger.info("\nNo reassignments recorded yet.")
        return
    
    # By trigger type
    trigger_query = """
        SELECT 
            triggered_by,
            COUNT(*) as count
        FROM task_reassignments
        GROUP BY triggered_by
        ORDER BY count DESC
    """
    triggers = execute_query(trigger_query)
    
    logger.info("\nBy Trigger Type:")
    for _, row in triggers.iterrows():
        logger.info(f"   {row['triggered_by']}: {int(row['count'])} ({row['count']/total*100:.1f}%)")
    
    # Most reassigned tasks
    most_reassigned_query = """
        SELECT 
            t.task_id,
            t.task_name,
            COUNT(*) as reassignment_count
        FROM task_reassignments tr
        JOIN tasks t ON tr.task_id = t.task_id
        GROUP BY t.task_id
        HAVING COUNT(*) > 1
        ORDER BY reassignment_count DESC
        LIMIT 5
    """
    most_reassigned = execute_query(most_reassigned_query)
    
    if len(most_reassigned) > 0:
        logger.info("\nMost Reassigned Tasks:")
        for _, row in most_reassigned.iterrows():
            logger.info(f"   {row['task_id']}: {int(row['reassignment_count'])} times")
    
    # Top assignees receiving tasks
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
    
    logger.info("\nTop Assignees (Received Tasks):")
    for _, row in received.iterrows():
        logger.info(f"   {row['to_assignee']}: {int(row['received_count'])} tasks")
    
    # Top assignees giving away tasks
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
    
    logger.info("\nTop Assignees (Gave Away Tasks):")
    for _, row in given.iterrows():
        logger.info(f"   {row['from_assignee']}: {int(row['given_count'])} tasks")


def generate_reassignment_report():
    """Generate comprehensive reassignment report"""
    logger.info("\n" + "="*70)
    logger.info("TASK REASSIGNMENT COMPREHENSIVE REPORT")
    logger.info("="*70)
    
    get_reassignment_statistics()
    calculate_reassignment_effectiveness()
    
    # Show recent reassignments
    logger.info("\n" + "="*60)
    logger.info("[INFO] RECENT REASSIGNMENTS (Last 10)")
    logger.info("="*60 + "\n")
    
    recent = get_reassignment_history(limit=10)
    
    if len(recent) > 0:
        for _, row in recent.iterrows():
            logger.info(f"Task: {row['task_id']} - {row['task_name'][:40]}")
            logger.info(f"   {row['old_assignee']} → {row['new_assignee']}")
            logger.info(f"   Triggered by: {row['triggered_by']}")
            logger.info(f"   Date: {row['reassigned_at'][:19]}")
            if row['reason']:
                logger.info(f"   Reason: {row['reason'][:80]}")
            logger.info(f"   Current Status: {row['current_status']}")
            logger.info("")
    else:
        logger.info("No reassignments recorded yet.")


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'report':
            generate_reassignment_report()
        
        elif command == 'rebalance':
            weekly_workload_rebalancing()
        
        elif command == 'auto-reassign':
            threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
            auto_reassign_high_delay_predictions(threshold)
        
        elif command == 'effectiveness':
            calculate_reassignment_effectiveness()
        
        else:
            print("Usage: python reassignment_tracker.py [report|rebalance|auto-reassign|effectiveness]")
    
    else:
        # Default: show full report
        generate_reassignment_report()
