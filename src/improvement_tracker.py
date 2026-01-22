"""
Improvement Tracking Module for FlowFix AI - Production Grade
Enhanced with CLI, score fields, KPI push, and API routes
"""
import argparse
import logging
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import pandas as pd
from sqlalchemy import text
from src.utils import get_engine, execute_query, update_dashboard_kpis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_baseline_metrics():
    """Calculate current baseline metrics"""
    logger.info("[STATS] Calculating baseline metrics...")
    
    query = """
    SELECT 
        COUNT(*) as total_tasks,
        AVG(actual_duration) as avg_duration,
        SUM(CASE WHEN is_delayed = 1 THEN 1 ELSE 0 END) as delayed_tasks,
        COUNT(DISTINCT assignee) as total_assignees,
        COUNT(CASE WHEN bottleneck_type != '' AND bottleneck_type IS NOT NULL THEN 1 END) as bottleneck_count,
        AVG(CASE WHEN status = 'Completed' THEN actual_duration END) as avg_completed_duration,
        COUNT(CASE WHEN status = 'Completed' THEN 1 END) as completed_count
    FROM tasks
    WHERE actual_duration IS NOT NULL
    """
    
    result = execute_query(query).iloc[0]
    
    total_tasks = int(result['total_tasks'])
    delayed_tasks = int(result['delayed_tasks'])
    bottleneck_count = int(result['bottleneck_count'])
    
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'total_tasks': total_tasks,
        'avg_duration': float(result['avg_duration']),
        'delayed_tasks': delayed_tasks,
        'delay_rate': (delayed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
        'bottleneck_count': bottleneck_count,
        'bottleneck_rate': (bottleneck_count / total_tasks * 100) if total_tasks > 0 else 0,
        'total_assignees': int(result['total_assignees']),
        'avg_completed_duration': float(result['avg_completed_duration']) if result['avg_completed_duration'] else 0,
        'completed_count': int(result['completed_count'])
    }
    
    logger.info(f"[SUCCESS] Baseline captured:")
    logger.info(f"   Total Tasks: {metrics['total_tasks']}")
    logger.info(f"   Avg Duration: {metrics['avg_duration']:.2f} days")
    logger.info(f"   Delay Rate: {metrics['delay_rate']:.1f}%")
    logger.info(f"   Bottleneck Rate: {metrics['bottleneck_rate']:.1f}%")
    
    return metrics


def calculate_improvement_score(metrics_before: Dict, metrics_after: Dict) -> float:
    """
    Calculate improvement score (0-100) based on multiple factors
    
    Factors:
    - Duration improvement (40%)
    - Delay rate improvement (30%)
    - Bottleneck rate improvement (30%)
    """
    score = 50  # Base score
    
    # Duration improvement (up to 40 points)
    if metrics_before.get('avg_duration', 0) > 0:
        duration_improvement = (
            (metrics_before['avg_duration'] - metrics_after['avg_duration']) / 
            metrics_before['avg_duration'] * 100
        )
        # Each 10% improvement = 10 points (max 40)
        score += min(duration_improvement * 4, 40)
    
    # Delay rate improvement (up to 30 points)
    delay_improvement = metrics_before.get('delay_rate', 0) - metrics_after.get('delay_rate', 0)
    # Each 5pp improvement = 10 points (max 30)
    score += min(delay_improvement * 2, 30)
    
    # Bottleneck rate improvement (up to 30 points)
    bottleneck_improvement = metrics_before.get('bottleneck_rate', 0) - metrics_after.get('bottleneck_rate', 0)
    # Each 5pp improvement = 10 points (max 30)
    score += min(bottleneck_improvement * 2, 30)
    
    return max(0, min(score, 100))


def save_improvement_log(
    task_id: str, 
    action_taken: str, 
    impact_description: str = '',
    improvement_score: Optional[float] = None,
    metrics_before: Optional[Dict] = None, 
    metrics_after: Optional[Dict] = None, 
    owner: str = 'System'
) -> bool:
    """
    Log an improvement action with enhanced tracking
    
    Args:
        task_id: Task ID
        action_taken: Description of action
        impact_description: Impact details
        improvement_score: Calculated score (0-100)
        metrics_before: Baseline metrics dict
        metrics_after: Post-improvement metrics dict
        owner: Person responsible
        
    Returns:
        bool: Success status
    """
    engine = get_engine()
    
    # Calculate improvement score if metrics provided
    if improvement_score is None and metrics_before and metrics_after:
        improvement_score = calculate_improvement_score(metrics_before, metrics_after)
    
    # Calculate improvement percentage if possible
    improvement_pct = None
    if metrics_after and metrics_before:
        if metrics_before.get('delay_rate', 0) > 0:
            improvement_pct = (
                (metrics_before['delay_rate'] - metrics_after['delay_rate']) / 
                metrics_before['delay_rate'] * 100
            )
    
    try:
        with engine.connect() as conn:
            query = text("""
                INSERT INTO improvement_log 
                (task_id, action_taken, owner, date_applied, impact_measured, 
                 improvement_score, improvement_percentage)
                VALUES (:task_id, :action_taken, :owner, :date_applied, :impact_measured,
                        :improvement_score, :improvement_pct)
            """)
            
            conn.execute(query, {
                'task_id': task_id,
                'action_taken': action_taken,
                'owner': owner,
                'date_applied': datetime.now().date().isoformat(),
                'impact_measured': impact_description,
                'improvement_score': improvement_score,
                'improvement_pct': improvement_pct
            })
            conn.commit()
        
        logger.info(f"[SUCCESS] Improvement logged for task {task_id}")
        if improvement_score:
            logger.info(f"   Improvement Score: {improvement_score:.1f}/100")
        if improvement_pct:
            logger.info(f"   Improvement: {improvement_pct:.1f}%")
        
        # Push updated KPIs to dashboard
        update_dashboard_kpis()
        
        return True
    
    except Exception as e:
        logger.error(f"Error saving improvement log: {e}")
        return False


def mark_suggestion_applied(task_id: str, action_description: str) -> bool:
    """
    Mark a GPT suggestion as applied
    
    Args:
        task_id: Task ID
        action_description: What was done
        
    Returns:
        bool: Success status
    """
    engine = get_engine()
    
    try:
        with engine.connect() as conn:
            # Update applied flag in gpt_suggestions
            query = text("""
                UPDATE gpt_suggestions 
                SET applied = 1,
                    applied_date = :applied_date,
                    applied_action = :action
                WHERE task_id = :task_id
            """)
            
            conn.execute(query, {
                'task_id': task_id,
                'applied_date': datetime.now().isoformat(),
                'action': action_description
            })
            conn.commit()
        
        logger.info(f"[SUCCESS] Marked suggestion for task {task_id} as applied")
        return True
    
    except Exception as e:
        logger.error(f"Error marking suggestion applied: {e}")
        return False


def compare_metrics(before: Dict, after: Dict) -> Dict:
    """
    Compare before and after metrics with detailed analysis
    
    Args:
        before: Baseline metrics
        after: Current metrics
        
    Returns:
        dict: Comparison results
    """
    logger.info("\n" + "="*60)
    logger.info("[STATS] IMPROVEMENT ANALYSIS")
    logger.info("="*60 + "\n")
    
    # Duration comparison
    duration_change = after['avg_duration'] - before['avg_duration']
    duration_pct = (duration_change / before['avg_duration'] * 100) if before['avg_duration'] > 0 else 0
    
    logger.info("Before vs After:")
    logger.info(f"   Avg Duration: {before['avg_duration']:.2f} → {after['avg_duration']:.2f} days "
          f"({duration_pct:+.1f}%)")
    
    # Delay rate comparison
    delay_change = after['delay_rate'] - before['delay_rate']
    logger.info(f"   Delay Rate: {before['delay_rate']:.1f}% → {after['delay_rate']:.1f}% "
          f"({delay_change:+.1f}pp)")
    
    # Bottleneck rate comparison
    bottleneck_change = after['bottleneck_rate'] - before['bottleneck_rate']
    logger.info(f"   Bottleneck Rate: {before['bottleneck_rate']:.1f}% → {after['bottleneck_rate']:.1f}% "
          f"({bottleneck_change:+.1f}pp)")
    
    # Calculate overall improvement score
    improvement_score = calculate_improvement_score(before, after)
    
    logger.info(f"\n   Overall Improvement Score: {improvement_score:.1f}/100")
    
    if improvement_score >= 80:
        logger.info("   Status: [SUCCESS] Excellent Improvement!")
    elif improvement_score >= 60:
        logger.info("   Status: [SUCCESS] Significant Improvement!")
    elif improvement_score >= 50:
        logger.info("   Status: [INFO] Positive Improvement")
    else:
        logger.info("   Status: [WARNING] Needs More Action")
    
    results = {
        'duration_change_days': duration_change,
        'duration_change_pct': duration_pct,
        'delay_change_pp': delay_change,
        'bottleneck_change_pp': bottleneck_change,
        'improvement_score': improvement_score
    }
    
    return results


def get_improvement_history(limit: int = 50) -> pd.DataFrame:
    """
    Get history of improvements with enhanced fields
    
    Args:
        limit: Maximum number of records to return
        
    Returns:
        DataFrame: Improvement history
    """
    query = f"""
    SELECT 
        il.task_id,
        t.task_name,
        il.action_taken,
        il.owner,
        il.date_applied,
        il.impact_measured,
        il.improvement_score,
        il.improvement_percentage,
        t.status as current_status,
        t.is_delayed as currently_delayed
    FROM improvement_log il
    LEFT JOIN tasks t ON il.task_id = t.task_id
    ORDER BY il.date_applied DESC
    LIMIT {limit}
    """
    
    df = execute_query(query)
    
    if len(df) == 0:
        logger.info("[INFO] No improvements logged yet")
        return df
    
    logger.info(f"\n[INFO] Improvement History ({len(df)} actions):")
    for _, row in df.head(10).iterrows():
        logger.info(f"\n   {row['date_applied']}")
        logger.info(f"   Task: {row['task_id']} - {row['task_name'][:40] if row['task_name'] else 'N/A'}")
        logger.info(f"   Action: {row['action_taken']}")
        logger.info(f"   Owner: {row['owner']}")
        if row['improvement_score']:
            logger.info(f"   Score: {row['improvement_score']:.1f}/100")
        if row['improvement_percentage']:
            logger.info(f"   Improvement: {row['improvement_percentage']:.1f}%")
    
    return df


def get_improvement_kpis() -> Dict:
    """
    Get key performance indicators for improvements
    
    Returns:
        dict: KPI metrics
    """
    query = """
    SELECT 
        COUNT(*) as total_improvements,
        AVG(improvement_score) as avg_improvement_score,
        AVG(improvement_percentage) as avg_improvement_pct,
        COUNT(CASE WHEN improvement_score >= 80 THEN 1 END) as excellent_count,
        COUNT(CASE WHEN improvement_score >= 60 THEN 1 END) as significant_count
    FROM improvement_log
    WHERE improvement_score IS NOT NULL
    """
    
    result = execute_query(query)
    
    if len(result) == 0 or result.iloc[0]['total_improvements'] == 0:
        return {
            'total_improvements': 0,
            'avg_improvement_score': 0,
            'avg_improvement_pct': 0,
            'excellent_count': 0,
            'significant_count': 0
        }
    
    row = result.iloc[0]
    
    kpis = {
        'total_improvements': int(row['total_improvements']),
        'avg_improvement_score': float(row['avg_improvement_score']) if row['avg_improvement_score'] else 0,
        'avg_improvement_pct': float(row['avg_improvement_pct']) if row['avg_improvement_pct'] else 0,
        'excellent_count': int(row['excellent_count']),
        'significant_count': int(row['significant_count'])
    }
    
    return kpis


def generate_improvement_report() -> Dict:
    """"Generate comprehensive improvement report
    
    Returns:
        dict: Report data
    """
    logger.info("\n" + "="*60)
    logger.info("[STATS] IMPROVEMENT TRACKING REPORT")
    logger.info("="*60 + "\n")
    
    # Get current metrics
    current = get_baseline_metrics()
    
    # Get improvement KPIs
    kpis = get_improvement_kpis()
    
    # Get improvement history
    history = get_improvement_history(limit=10)
    
    # Get applied suggestions count
    query = """
    SELECT 
        COUNT(*) as total_suggestions,
        COUNT(CASE WHEN applied = 1 THEN 1 END) as applied_count
    FROM gpt_suggestions
    """
    
    suggestions = execute_query(query).iloc[0]
    
    logger.info(f"\n[STATS] Improvement Summary:")
    logger.info(f"   Total Actions: {kpis['total_improvements']}")
    logger.info(f"   Avg Improvement Score: {kpis['avg_improvement_score']:.1f}/100")
    logger.info(f"   Excellent Improvements: {kpis['excellent_count']}")
    logger.info(f"   Significant Improvements: {kpis['significant_count']}")
    
    logger.info(f"\n[AI] GPT Suggestions:")
    logger.info(f"   Total Generated: {int(suggestions['total_suggestions'])}")
    logger.info(f"   Applied: {int(suggestions['applied_count'])}")
    if suggestions['total_suggestions'] > 0:
        apply_rate = suggestions['applied_count'] / suggestions['total_suggestions'] * 100
        logger.info(f"   Apply Rate: {apply_rate:.1f}%")
    
    logger.info(f"\n[STATS] Current Metrics:")
    logger.info(f"   Delay Rate: {current['delay_rate']:.1f}%")
    logger.info(f"   Bottleneck Rate: {current['bottleneck_rate']:.1f}%")
    logger.info(f"   Avg Duration: {current['avg_duration']:.1f} days")
    
    return {
        'current_metrics': current,
        'kpis': kpis,
        'improvement_history': history.to_dict('records') if len(history) > 0 else [],
        'applied_suggestions': int(suggestions['applied_count']),
        'total_suggestions': int(suggestions['total_suggestions'])
    }


# ============================================================================
# CLI Interface
# ============================================================================

def cli_log_improvement():
    """CLI command to log an improvement"""
    parser = argparse.ArgumentParser(description='Log an improvement action')
    parser.add_argument('task_id', help='Task ID')
    parser.add_argument('action', help='Action taken')
    parser.add_argument('--owner', default='Manual', help='Person responsible')
    parser.add_argument('--impact', default='', help='Impact description')
    parser.add_argument('--score', type=float, help='Improvement score (0-100)')
    
    args = parser.parse_args()
    
    success = save_improvement_log(
        task_id=args.task_id,
        action_taken=args.action,
        impact_description=args.impact,
        improvement_score=args.score,
        owner=args.owner
    )
    
    if success:
        print(f"\n[SUCCESS] Improvement logged successfully!")
    else:
        print(f"\n[ERROR] Failed to log improvement")


def cli_mark_applied():
    """CLI command to mark suggestion as applied"""
    parser = argparse.ArgumentParser(description='Mark GPT suggestion as applied')
    parser.add_argument('task_id', help='Task ID')
    parser.add_argument('action', help='Action taken')
    
    args = parser.parse_args()
    
    success = mark_suggestion_applied(args.task_id, args.action)
    
    if success:
        print(f"\n[SUCCESS] Suggestion marked as applied!")
        # Also log to improvement_log
        save_improvement_log(
            task_id=args.task_id,
            action_taken=f"Applied GPT suggestion: {args.action}",
            owner='User'
        )
    else:
        print(f"\n[ERROR] Failed to mark suggestion")


def cli_compare():
    """CLI command to compare metrics"""
    print("\n[STATS] Comparing baseline metrics...\n")
    
    # For demo, get metrics from 7 days ago vs now
    # In production, you'd store historical snapshots
    current = get_baseline_metrics()
    
    # Simulate "before" metrics (for demo)
    before = current.copy()
    before['delay_rate'] = current['delay_rate'] * 1.2
    before['bottleneck_rate'] = current['bottleneck_rate'] * 1.15
    before['avg_duration'] = current['avg_duration'] * 1.1
    
    compare_metrics(before, current)


def cli_report():
    """CLI command to generate report"""
    generate_improvement_report()


def cli_kpis():
    """CLI command to show KPIs"""
    kpis = get_improvement_kpis()
    
    print("\n[STATS] Improvement KPIs:")
    print(f"   Total Improvements: {kpis['total_improvements']}")
    print(f"   Avg Score: {kpis['avg_improvement_score']:.1f}/100")
    print(f"   Excellent (≥80): {kpis['excellent_count']}")
    print(f"   Significant (≥60): {kpis['significant_count']}")
    print()


# ============================================================================
# API Routes (for future web integration)
# ============================================================================

def api_track_improvement(data: Dict) -> Dict:
    """
    API endpoint to track improvement
    
    POST /track_improvement
    Body: {
        "task_id": "TASK-123",
        "action": "Reassigned to expert",
        "owner": "Manager",
        "impact": "Reduced delay",
        "score": 75.5
    }
    
    Returns:
        dict: Response with success status
    """
    required_fields = ['task_id', 'action']
    
    # Validate input
    for field in required_fields:
        if field not in data:
            return {
                'success': False,
                'error': f'Missing required field: {field}'
            }
    
    success = save_improvement_log(
        task_id=data['task_id'],
        action_taken=data['action'],
        owner=data.get('owner', 'API'),
        impact_description=data.get('impact', ''),
        improvement_score=data.get('score')
    )
    
    return {
        'success': success,
        'task_id': data['task_id'],
        'timestamp': datetime.now().isoformat()
    }


def api_get_improvement_stats() -> Dict:
    """
    API endpoint to get improvement statistics
    
    GET /improvement_stats
    
    Returns:
        dict: Current statistics
    """
    kpis = get_improvement_kpis()
    current = get_baseline_metrics()
    
    return {
        'success': True,
        'timestamp': datetime.now().isoformat(),
        'kpis': kpis,
        'current_metrics': current
    }


def api_mark_applied(data: Dict) -> Dict:
    """
    API endpoint to mark suggestion as applied
    
    POST /mark_applied
    Body: {
        "task_id": "TASK-123",
        "action": "Implemented recommendation"
    }
    
    Returns:
        dict: Response with success status
    """
    if 'task_id' not in data or 'action' not in data:
        return {
            'success': False,
            'error': 'Missing required fields: task_id, action'
        }
    
    success = mark_suggestion_applied(data['task_id'], data['action'])
    
    return {
        'success': success,
        'task_id': data['task_id'],
        'timestamp': datetime.now().isoformat()
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'log':
            cli_log_improvement()
        elif command == 'mark-applied':
            cli_mark_applied()
        elif command == 'compare':
            cli_compare()
        elif command == 'report':
            cli_report()
        elif command == 'kpis':
            cli_kpis()
        elif command == 'history':
            get_improvement_history(limit=20)
        else:
            print("Usage: python improvement_tracker.py [log|mark-applied|compare|report|kpis|history]")
            print("\nCommands:")
            print("  log <task_id> <action> [--owner NAME] [--score SCORE]")
            print("  mark-applied <task_id> <action>")
            print("  compare  - Compare before/after metrics")
            print("  report   - Generate full report")
            print("  kpis     - Show improvement KPIs")
            print("  history  - Show improvement history")
    else:
        # Default: generate report
        generate_improvement_report()
