"""
Improvement Tracking Module for FlowFix AI
Tracks before/after metrics and measures impact of applied suggestions
"""
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import text
from utils import get_engine, execute_query


def get_baseline_metrics():
    """Calculate current baseline metrics"""
    print("📊 Calculating baseline metrics...")
    
    query = """
    SELECT 
        COUNT(*) as total_tasks,
        AVG(actual_duration) as avg_duration,
        SUM(CASE WHEN is_delayed = 1 THEN 1 ELSE 0 END) as delayed_tasks,
        COUNT(DISTINCT assignee) as total_assignees,
        COUNT(CASE WHEN bottleneck_type != '' THEN 1 END) as bottleneck_count
    FROM tasks
    WHERE actual_duration IS NOT NULL
    """
    
    result = execute_query(query).iloc[0]
    
    metrics = {
        'timestamp': datetime.now(),
        'total_tasks': int(result['total_tasks']),
        'avg_duration': float(result['avg_duration']),
        'delayed_tasks': int(result['delayed_tasks']),
        'delay_rate': float(result['delayed_tasks']) / float(result['total_tasks']) * 100,
        'bottleneck_count': int(result['bottleneck_count']),
        'bottleneck_rate': float(result['bottleneck_count']) / float(result['total_tasks']) * 100,
        'total_assignees': int(result['total_assignees'])
    }
    
    print(f"✅ Baseline captured:")
    print(f"   Total Tasks: {metrics['total_tasks']}")
    print(f"   Avg Duration: {metrics['avg_duration']:.2f} days")
    print(f"   Delay Rate: {metrics['delay_rate']:.1f}%")
    print(f"   Bottleneck Rate: {metrics['bottleneck_rate']:.1f}%")
    
    return metrics


def save_improvement_log(task_id, action_taken, impact_description, metrics_before=None, metrics_after=None, owner=None):
    """Log an improvement action and its impact"""
    engine = get_engine()
    
    # Calculate improvement if after metrics provided
    improvement_pct = None
    if metrics_after and metrics_before:
        if 'delay_rate' in metrics_before and 'delay_rate' in metrics_after:
            improvement_pct = ((metrics_before['delay_rate'] - metrics_after['delay_rate']) / 
                              metrics_before['delay_rate'] * 100)
    
    with engine.connect() as conn:
        query = text("""
            INSERT INTO improvement_log 
            (task_id, action_taken, owner, date_applied, impact_measured)
            VALUES (:task_id, :action_taken, :owner, :date_applied, :impact_measured)
        """)
        
        conn.execute(query, {
            'task_id': task_id,
            'action_taken': action_taken,
            'owner': owner or 'System',
            'date_applied': datetime.now().date(),
            'impact_measured': impact_description
        })
        conn.commit()
    
    print(f"✅ Improvement logged for task {task_id}")
    if improvement_pct:
        print(f"   Improvement: {improvement_pct:.1f}%")


def mark_suggestion_applied(task_id, action_description):
    """Mark a GPT suggestion as applied"""
    engine = get_engine()
    
    with engine.connect() as conn:
        # Update the gpt_suggestions table to mark as applied
        query = text("""
            UPDATE gpt_suggestions 
            SET suggestion_text = suggestion_text || '\n[APPLIED: ' || :action || ']'
            WHERE task_id = :task_id
        """)
        
        conn.execute(query, {
            'task_id': task_id,
            'action': action_description
        })
        conn.commit()
    
    print(f"✅ Marked suggestion for task {task_id} as applied")


def compare_metrics(before, after):
    """Compare before and after metrics"""
    print("\n" + "="*60)
    print("📈 IMPROVEMENT ANALYSIS")
    print("="*60 + "\n")
    
    print("Before vs After:")
    print(f"   Avg Duration: {before['avg_duration']:.2f} → {after['avg_duration']:.2f} days "
          f"({((after['avg_duration'] - before['avg_duration'])/before['avg_duration']*100):+.1f}%)")
    
    print(f"   Delay Rate: {before['delay_rate']:.1f}% → {after['delay_rate']:.1f}% "
          f"({(after['delay_rate'] - before['delay_rate']):+.1f}pp)")
    
    print(f"   Bottleneck Rate: {before['bottleneck_rate']:.1f}% → {after['bottleneck_rate']:.1f}% "
          f"({(after['bottleneck_rate'] - before['bottleneck_rate']):+.1f}pp)")
    
    # Calculate overall improvement score
    duration_improvement = (before['avg_duration'] - after['avg_duration']) / before['avg_duration'] * 100
    delay_improvement = (before['delay_rate'] - after['delay_rate'])
    bottleneck_improvement = (before['bottleneck_rate'] - after['bottleneck_rate'])
    
    overall_score = (duration_improvement * 0.4 + delay_improvement * 0.3 + bottleneck_improvement * 0.3)
    
    print(f"\n   Overall Improvement Score: {overall_score:+.1f}%")
    
    if overall_score > 5:
        print("   Status: ✅ Significant Improvement!")
    elif overall_score > 0:
        print("   Status: ✓ Positive Improvement")
    else:
        print("   Status: ⚠️ Needs More Action")
    
    return {
        'duration_improvement': duration_improvement,
        'delay_improvement': delay_improvement,
        'bottleneck_improvement': bottleneck_improvement,
        'overall_score': overall_score
    }


def get_improvement_history():
    """Get history of improvements"""
    query = """
    SELECT 
        task_id,
        action_taken,
        owner,
        date_applied,
        impact_measured
    FROM improvement_log
    ORDER BY date_applied DESC
    """
    
    df = execute_query(query)
    
    if len(df) == 0:
        print("📋 No improvements logged yet")
        return df
    
    print(f"\n📋 Improvement History ({len(df)} actions):")
    for idx, row in df.iterrows():
        print(f"\n   {row['date_applied']}")
        print(f"   Task: {row['task_id']}")
        print(f"   Action: {row['action_taken']}")
        print(f"   Owner: {row['owner']}")
        if row['impact_measured']:
            print(f"   Impact: {row['impact_measured']}")
        if row['improvement_percentage']:
            print(f"   Improvement: {row['improvement_percentage']:.1f}%")
    
    return df


def simulate_improvement(task_ids, improvement_factor=0.8):
    """Simulate improvement by reducing durations of specific tasks (for demo)"""
    print(f"\n🔧 Simulating improvement for {len(task_ids)} tasks...")
    
    engine = get_engine()
    
    with engine.connect() as conn:
        for task_id in task_ids:
            query = text("""
                UPDATE tasks 
                SET actual_duration = actual_duration * :factor,
                    is_delayed = 0,
                    bottleneck_type = ''
                WHERE task_id = :task_id
            """)
            
            conn.execute(query, {
                'task_id': task_id,
                'factor': improvement_factor
            })
        
        conn.commit()
    
    print(f"✅ Simulated improvement applied (reduced duration by {(1-improvement_factor)*100:.0f}%)")


def generate_improvement_report():
    """Generate comprehensive improvement report"""
    print("\n" + "="*60)
    print("📊 IMPROVEMENT TRACKING REPORT")
    print("="*60 + "\n")
    
    # Get current metrics
    current = get_baseline_metrics()
    
    # Get improvement history
    history = get_improvement_history()
    
    # Get applied suggestions
    query = """
    SELECT COUNT(*) as applied_count
    FROM gpt_suggestions
    WHERE suggestion_text LIKE '%[APPLIED:%'
    """
    
    applied_count = execute_query(query).iloc[0]['applied_count']
    
    print(f"\n📈 Summary:")
    print(f"   Actions Applied: {applied_count}")
    print(f"   Current Delay Rate: {current['delay_rate']:.1f}%")
    print(f"   Current Bottleneck Rate: {current['bottleneck_rate']:.1f}%")
    
    return {
        'current_metrics': current,
        'improvement_history': history,
        'applied_suggestions': applied_count
    }


if __name__ == "__main__":
    # Example usage
    print("🚀 FlowFix AI - Improvement Tracking\n")
    
    # Get baseline
    baseline = get_baseline_metrics()
    
    # Show improvement history
    get_improvement_history()
    
    # Generate report
    report = generate_improvement_report()
