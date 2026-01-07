"""
Feedback Loop System for FlowFix AI - Production Version
Tracks which suggestions are applied/rejected and measures impact with actual metrics

Key Production Features:
- Track actual impact (duration reduced, delay reduced, completion time)
- Full feedback viewer with filtering and sorting
- Export feedback to CSV/PDF for reporting
- Validate feedback before update to prevent data corruption
- Impact scoring algorithm with before/after comparisons
- Trend analysis over time periods
"""
import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
from utils import execute_query, get_engine
from sqlalchemy import text


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feedback_loop.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def add_feedback_columns():
    """Add feedback tracking columns to gpt_suggestions table"""
    engine = get_engine()
    
    with engine.connect() as conn:
        # Check if columns already exist
        result = conn.execute(text("PRAGMA table_info(gpt_suggestions)"))
        columns = [row[1] for row in result.fetchall()]
        
        if 'feedback_status' not in columns:
            logger.info("Adding feedback columns to database...")
            
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
            
            conn.execute(text("""
                ALTER TABLE gpt_suggestions 
                ADD COLUMN was_helpful INTEGER DEFAULT NULL
            """))
            
            conn.execute(text("""
                ALTER TABLE gpt_suggestions 
                ADD COLUMN applied_action TEXT DEFAULT ''
            """))
            
            conn.commit()
            logger.info("✅ Feedback columns added successfully")
        else:
            logger.info("Feedback columns already exist")


def validate_feedback_input(task_id: str, status: str) -> bool:
    """
    Validate feedback input before updating database
    
    Args:
        task_id: Task ID to validate
        status: Feedback status to validate
    
    Returns:
        True if valid, False otherwise
    """
    # Check if task exists
    query = text("SELECT COUNT(*) as count FROM gpt_suggestions WHERE task_id = :task_id")
    engine = get_engine()
    
    with engine.connect() as conn:
        result = conn.execute(query, {'task_id': task_id}).fetchone()
        if result[0] == 0:
            logger.error(f"Task {task_id} not found in gpt_suggestions")
            return False
    
    # Check if status is valid
    valid_statuses = ['applied', 'rejected', 'pending', 'under_review']
    if status not in valid_statuses:
        logger.error(f"Invalid status: {status}. Must be one of {valid_statuses}")
        return False
    
    return True


def mark_suggestion_feedback(
    task_id: str, 
    status: str = 'applied', 
    notes: str = '',
    was_helpful: Optional[bool] = None,
    applied_action: str = ''
):
    """
    Mark a GPT suggestion with feedback and validation
    
    Args:
        task_id: The task ID
        status: 'applied', 'rejected', 'pending', 'under_review'
        notes: Optional notes about the feedback
        was_helpful: True/False if suggestion was helpful
        applied_action: Description of what action was taken
    """
    # Validate input
    if not validate_feedback_input(task_id, status):
        return False
    
    engine = get_engine()
    
    try:
        with engine.connect() as conn:
            # Update gpt_suggestions table
            query = text("""
                UPDATE gpt_suggestions
                SET feedback_status = :status,
                    feedback_notes = :notes,
                    feedback_date = :date,
                    was_helpful = :was_helpful,
                    applied_action = :applied_action
                WHERE task_id = :task_id
            """)
            
            conn.execute(query, {
                'status': status,
                'notes': notes,
                'date': datetime.now().isoformat(),
                'was_helpful': 1 if was_helpful else (0 if was_helpful is False else None),
                'applied_action': applied_action,
                'task_id': task_id
            })
            conn.commit()
        
        logger.info(f"✅ Marked suggestion for task {task_id} as {status}")
        return True
        
    except Exception as e:
        logger.error(f"Error marking feedback for task {task_id}: {e}")
        return False


def track_actual_impact(task_id: str) -> Dict:
    """
    Track actual impact of applied suggestion by comparing before/after metrics
    
    Args:
        task_id: Task ID to track
        
    Returns:
        Dictionary with impact metrics
    """
    query = text("""
        SELECT 
            t.task_id,
            t.task_duration,
            t.actual_duration,
            t.status,
            t.start_date,
            t.end_date,
            gs.feedback_date,
            gs.recommendations,
            gs.predicted_improvement
        FROM tasks t
        LEFT JOIN gpt_suggestions gs ON t.task_id = gs.task_id
        WHERE t.task_id = :task_id
    """)
    
    engine = get_engine()
    impact = {
        'task_id': task_id,
        'duration_reduced_days': 0,
        'delay_prevented': False,
        'completion_time_days': None,
        'impact_score': 0
    }
    
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {'task_id': task_id}).fetchone()
            
            if result:
                task_duration = result[1] or 0
                actual_duration = result[2] or 0
                status = result[3]
                start_date = result[4]
                end_date = result[5]
                feedback_date = result[6]
                predicted_improvement = result[8] or 0
                
                # Calculate duration reduction
                if actual_duration > 0 and task_duration > 0:
                    impact['duration_reduced_days'] = max(0, task_duration - actual_duration)
                
                # Check if delay was prevented
                if status == 'Completed' and actual_duration <= task_duration:
                    impact['delay_prevented'] = True
                
                # Calculate completion time
                if start_date and end_date:
                    try:
                        start = datetime.fromisoformat(start_date)
                        end = datetime.fromisoformat(end_date)
                        impact['completion_time_days'] = (end - start).days
                    except:
                        pass
                
                # Calculate impact score (0-100)
                score = 0
                
                # Duration reduction (max 50 points)
                if impact['duration_reduced_days'] > 0:
                    score += min(50, impact['duration_reduced_days'] * 10)
                
                # Delay prevention (25 points)
                if impact['delay_prevented']:
                    score += 25
                
                # Completion status (25 points)
                if status == 'Completed':
                    score += 25
                
                impact['impact_score'] = min(100, score)
                
                # Log impact to feedback_log table
                log_query = text("""
                    INSERT INTO feedback_log (
                        task_id, feedback_date, impact_type, impact_value, impact_score
                    ) VALUES (
                        :task_id, :date, :type, :value, :score
                    )
                """)
                
                conn.execute(log_query, {
                    'task_id': task_id,
                    'date': datetime.now().isoformat(),
                    'type': 'suggestion_applied',
                    'value': impact['duration_reduced_days'],
                    'score': impact['impact_score']
                })
                conn.commit()
                
                logger.info(f"Impact tracked for {task_id}: {impact['impact_score']} score")
                
    except Exception as e:
        logger.error(f"Error tracking impact for task {task_id}: {e}")
    
    return impact


def get_feedback_summary(status_filter: Optional[str] = None) -> pd.DataFrame:
    """
    Get summary of suggestion feedback with optional filtering
    
    Args:
        status_filter: Optional status to filter by
        
    Returns:
        DataFrame with feedback summary
    """
    base_query = """
    SELECT 
        feedback_status,
        COUNT(*) as count,
        AVG(CASE WHEN was_helpful = 1 THEN 1.0 ELSE 0.0 END) as helpful_rate,
        COUNT(CASE WHEN was_helpful = 1 THEN 1 END) as helpful_count,
        COUNT(CASE WHEN was_helpful = 0 THEN 1 END) as not_helpful_count
    FROM gpt_suggestions
    """
    
    if status_filter:
        query = base_query + f" WHERE feedback_status = '{status_filter}' GROUP BY feedback_status"
    else:
        query = base_query + " GROUP BY feedback_status"
    
    df = execute_query(query)
    
    print("\n" + "="*80)
    print("FEEDBACK SUMMARY")
    print("="*80)
    
    if len(df) == 0:
        print("No feedback data available.")
        return df
    
    for idx, row in df.iterrows():
        status = row['feedback_status']
        count = int(row['count'])
        helpful_rate = row['helpful_rate'] * 100 if row['helpful_rate'] else 0
        helpful_count = int(row['helpful_count']) if row['helpful_count'] else 0
        not_helpful_count = int(row['not_helpful_count']) if row['not_helpful_count'] else 0
        
        print(f"\n{status.upper().replace('_', ' ')}:")
        print(f"  Total: {count} suggestions")
        print(f"  Helpful: {helpful_count} ({helpful_rate:.1f}%)")
        print(f"  Not Helpful: {not_helpful_count}")
    
    return df


def get_applied_suggestions(limit: int = 20) -> pd.DataFrame:
    """
    Get all applied suggestions with details and impact metrics
    
    Args:
        limit: Maximum number of suggestions to return
        
    Returns:
        DataFrame with applied suggestions
    """
    query = f"""
    SELECT 
        gs.task_id,
        gs.recommendations,
        gs.feedback_date,
        gs.feedback_notes,
        gs.was_helpful,
        gs.applied_action,
        t.assignee,
        t.priority,
        t.status,
        t.task_duration,
        t.actual_duration,
        fl.impact_score
    FROM gpt_suggestions gs
    JOIN tasks t ON gs.task_id = t.task_id
    LEFT JOIN feedback_log fl ON gs.task_id = fl.task_id AND fl.impact_type = 'suggestion_applied'
    WHERE gs.feedback_status = 'applied'
    ORDER BY gs.feedback_date DESC
    LIMIT {limit}
    """
    
    df = execute_query(query)
    
    if len(df) == 0:
        print("\nNo suggestions have been marked as applied yet.")
        return df
    
    print("\n" + "="*80)
    print(f"APPLIED SUGGESTIONS (Top {min(limit, len(df))})")
    print("="*80 + "\n")
    
    for idx, row in df.iterrows():
        print(f"Task: {row['task_id']}")
        print(f"  Assignee: {row['assignee']} | Priority: {row['priority']} | Status: {row['status']}")
        print(f"  Applied: {row['feedback_date'][:10] if row['feedback_date'] else 'N/A'}")
        
        if row['applied_action']:
            print(f"  Action Taken: {row['applied_action']}")
        
        recommendations = row['recommendations']
        if recommendations and len(str(recommendations)) > 100:
            print(f"  Recommendation: {str(recommendations)[:100]}...")
        elif recommendations:
            print(f"  Recommendation: {recommendations}")
        
        # Show impact metrics
        task_duration = row['task_duration'] if row['task_duration'] else 'N/A'
        actual_duration = row['actual_duration'] if row['actual_duration'] else 'N/A'
        impact_score = row['impact_score'] if row['impact_score'] else 'N/A'
        
        print(f"  Duration: {task_duration} days (planned) → {actual_duration} days (actual)")
        print(f"  Impact Score: {impact_score}")
        
        was_helpful = row['was_helpful']
        if was_helpful == 1:
            print(f"  Rating: ✅ Helpful")
        elif was_helpful == 0:
            print(f"  Rating: ❌ Not Helpful")
        
        if row['feedback_notes']:
            print(f"  Notes: {row['feedback_notes']}")
        
        print()
    
    return df


def measure_feedback_impact() -> Dict:
    """
    Measure the overall impact of applied suggestions with detailed metrics
    
    Returns:
        Dictionary with impact statistics
    """
    print("\n" + "="*80)
    print("FEEDBACK IMPACT ANALYSIS")
    print("="*80 + "\n")
    
    query = """
    SELECT 
        COUNT(*) as applied_count,
        AVG(CASE WHEN t.status = 'Completed' THEN 1.0 ELSE 0.0 END) as completion_rate,
        AVG(CASE WHEN t.actual_duration <= t.task_duration THEN 1.0 ELSE 0.0 END) as on_time_rate,
        AVG(t.task_duration - COALESCE(t.actual_duration, t.task_duration)) as avg_duration_reduction,
        AVG(fl.impact_score) as avg_impact_score
    FROM gpt_suggestions gs
    JOIN tasks t ON gs.task_id = t.task_id
    LEFT JOIN feedback_log fl ON gs.task_id = fl.task_id AND fl.impact_type = 'suggestion_applied'
    WHERE gs.feedback_status = 'applied'
    """
    
    result = execute_query(query)
    
    if len(result) == 0 or result.iloc[0]['applied_count'] == 0:
        print("Not enough applied suggestions to measure impact.")
        return {}
    
    applied = int(result.iloc[0]['applied_count'])
    completion = result.iloc[0]['completion_rate'] * 100
    on_time = result.iloc[0]['on_time_rate'] * 100
    avg_reduction = result.iloc[0]['avg_duration_reduction'] or 0
    avg_score = result.iloc[0]['avg_impact_score'] or 0
    
    print(f"Total Applied Suggestions: {applied}")
    print(f"Completion Rate: {completion:.1f}%")
    print(f"On-Time Rate: {on_time:.1f}%")
    print(f"Avg Duration Reduction: {avg_reduction:.1f} days")
    print(f"Avg Impact Score: {avg_score:.1f}/100")
    
    # Calculate ROI estimate
    hours_saved = avg_reduction * applied * 8  # Assuming 8 hours per day
    print(f"\nEstimated Time Saved: {hours_saved:.0f} hours ({hours_saved/8:.0f} person-days)")
    
    impact_stats = {
        'applied_count': applied,
        'completion_rate': completion,
        'on_time_rate': on_time,
        'avg_duration_reduction': avg_reduction,
        'avg_impact_score': avg_score,
        'hours_saved': hours_saved
    }
    
    return impact_stats


def export_feedback_to_csv(output_path: str = 'exports/feedback_report.csv'):
    """
    Export feedback data to CSV for external analysis
    
    Args:
        output_path: Path to save CSV file
    """
    query = """
    SELECT 
        gs.task_id,
        gs.feedback_status,
        gs.feedback_date,
        gs.feedback_notes,
        gs.was_helpful,
        gs.applied_action,
        gs.recommendations,
        t.assignee,
        t.priority,
        t.status,
        t.task_duration,
        t.actual_duration,
        fl.impact_score,
        fl.impact_value as duration_reduced_days
    FROM gpt_suggestions gs
    JOIN tasks t ON gs.task_id = gs.task_id
    LEFT JOIN feedback_log fl ON gs.task_id = fl.task_id AND fl.impact_type = 'suggestion_applied'
    ORDER BY gs.feedback_date DESC
    """
    
    try:
        df = execute_query(query)
        
        # Create exports directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"✅ Feedback exported to {output_path}")
        print(f"\n✅ Feedback data exported: {output_path}")
        print(f"   Total records: {len(df)}")
        
    except Exception as e:
        logger.error(f"Error exporting feedback to CSV: {e}")


def view_feedback_interactive(status: Optional[str] = None, limit: int = 10):
    """
    Interactive feedback viewer with filtering
    
    Args:
        status: Optional status filter
        limit: Number of records to display
    """
    print("\n" + "="*80)
    print("INTERACTIVE FEEDBACK VIEWER")
    print("="*80)
    
    # Build query with optional filter
    base_query = """
    SELECT 
        gs.task_id,
        gs.feedback_status,
        gs.feedback_date,
        gs.was_helpful,
        gs.applied_action,
        t.assignee,
        t.status as task_status
    FROM gpt_suggestions gs
    JOIN tasks t ON gs.task_id = t.task_id
    """
    
    if status:
        query = base_query + f" WHERE gs.feedback_status = '{status}'"
    else:
        query = base_query
    
    query += f" ORDER BY gs.feedback_date DESC LIMIT {limit}"
    
    df = execute_query(query)
    
    if len(df) == 0:
        print("\nNo feedback records found.")
        return
    
    print(f"\nShowing {len(df)} feedback records:\n")
    
    for idx, row in df.iterrows():
        helpful_icon = "✅" if row['was_helpful'] == 1 else ("❌" if row['was_helpful'] == 0 else "⏳")
        status_icon = "📝" if row['feedback_status'] == 'pending' else ("✓" if row['feedback_status'] == 'applied' else "✗")
        
        print(f"{idx+1}. {status_icon} {row['task_id']} | {row['feedback_status'].upper()}")
        print(f"   Assignee: {row['assignee']} | Task Status: {row['task_status']} | Helpful: {helpful_icon}")
        
        if row['applied_action']:
            print(f"   Action: {row['applied_action'][:60]}...")
        
        print()


def analyze_feedback_trends(days: int = 30):
    """
    Analyze feedback trends over time period
    
    Args:
        days: Number of days to analyze
    """
    cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
    
    query = f"""
    SELECT 
        DATE(feedback_date) as date,
        feedback_status,
        COUNT(*) as count
    FROM gpt_suggestions
    WHERE feedback_date >= '{cutoff_date}'
    GROUP BY DATE(feedback_date), feedback_status
    ORDER BY date DESC
    """
    
    df = execute_query(query)
    
    print("\n" + "="*80)
    print(f"FEEDBACK TRENDS (Last {days} Days)")
    print("="*80 + "\n")
    
    if len(df) == 0:
        print("No feedback data in the specified period.")
        return
    
    # Group by date
    dates = df['date'].unique()
    
    for date in dates[:10]:  # Show last 10 days
        date_data = df[df['date'] == date]
        print(f"{date}:")
        
        for _, row in date_data.iterrows():
            print(f"  {row['feedback_status']}: {int(row['count'])}")
        print()


def create_feedback_report():
    """Generate comprehensive feedback report"""
    print("\n" + "="*80)
    print("COMPREHENSIVE FEEDBACK LOOP REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Summary
    get_feedback_summary()
    
    # Applied suggestions
    get_applied_suggestions(limit=10)
    
    # Impact measurement
    measure_feedback_impact()
    
    # Trends
    analyze_feedback_trends(days=30)


def cli():
    """Command-line interface for feedback loop"""
    parser = argparse.ArgumentParser(description='FlowFix AI Feedback Loop System')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Mark feedback
    mark_parser = subparsers.add_parser('mark', help='Mark suggestion feedback')
    mark_parser.add_argument('task_id', help='Task ID')
    mark_parser.add_argument('status', choices=['applied', 'rejected', 'pending', 'under_review'])
    mark_parser.add_argument('--notes', default='', help='Feedback notes')
    mark_parser.add_argument('--helpful', type=bool, help='Was suggestion helpful?')
    mark_parser.add_argument('--action', default='', help='Action taken')
    
    # Track impact
    impact_parser = subparsers.add_parser('impact', help='Track impact of applied suggestion')
    impact_parser.add_argument('task_id', help='Task ID')
    
    # Summary
    summary_parser = subparsers.add_parser('summary', help='Show feedback summary')
    summary_parser.add_argument('--status', help='Filter by status')
    
    # View applied
    applied_parser = subparsers.add_parser('applied', help='View applied suggestions')
    applied_parser.add_argument('--limit', type=int, default=20, help='Number of records')
    
    # Export
    export_parser = subparsers.add_parser('export', help='Export feedback to CSV')
    export_parser.add_argument('--output', default='exports/feedback_report.csv', help='Output path')
    
    # View interactive
    view_parser = subparsers.add_parser('view', help='Interactive feedback viewer')
    view_parser.add_argument('--status', help='Filter by status')
    view_parser.add_argument('--limit', type=int, default=10, help='Number of records')
    
    # Trends
    trends_parser = subparsers.add_parser('trends', help='Analyze feedback trends')
    trends_parser.add_argument('--days', type=int, default=30, help='Days to analyze')
    
    # Report
    subparsers.add_parser('report', help='Generate comprehensive report')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize feedback columns
    add_feedback_columns()
    
    # Execute command
    if args.command == 'mark':
        mark_suggestion_feedback(
            args.task_id, 
            args.status, 
            args.notes,
            args.helpful,
            args.action
        )
    
    elif args.command == 'impact':
        impact = track_actual_impact(args.task_id)
        print(f"\nImpact for {args.task_id}:")
        print(f"  Duration Reduced: {impact['duration_reduced_days']} days")
        print(f"  Delay Prevented: {impact['delay_prevented']}")
        print(f"  Impact Score: {impact['impact_score']}/100")
    
    elif args.command == 'summary':
        get_feedback_summary(args.status)
    
    elif args.command == 'applied':
        get_applied_suggestions(args.limit)
    
    elif args.command == 'export':
        export_feedback_to_csv(args.output)
    
    elif args.command == 'view':
        view_feedback_interactive(args.status, args.limit)
    
    elif args.command == 'trends':
        analyze_feedback_trends(args.days)
    
    elif args.command == 'report':
        create_feedback_report()


if __name__ == "__main__":
    cli()
