"""
Production-Grade Bottleneck Detection Engine for FlowFix AI
Version: 2.0
Features: Severity scoring, history logging, root cause suggestions, ML estimator, auto-reports
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import text
import os
import logging
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
<<<<<<< HEAD
from src.utils import get_engine, execute_query
=======
from utils import get_engine, execute_query
>>>>>>> 789db11de11bf607177a31557cbb9b376ebcdde5

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

DELAY_THRESHOLD_MULTIPLIER = float(os.getenv('DELAY_THRESHOLD_MULTIPLIER', 1.5))
START_DELAY_DAYS = int(os.getenv('START_DELAY_DAYS', 3))
REASSIGNMENT_THRESHOLD = int(os.getenv('REASSIGNMENT_THRESHOLD', 2))


def calculate_severity_score(delay_days, priority, status):
    """
    Calculate severity score (0-100) for a bottleneck
    Based on delay days, priority level, and current status
    """
    score = 0.0
    
    # Delay component (0-50 points)
    if delay_days > 0:
        score += min(delay_days * 3, 50)  # Cap at 50
    
    # Priority component (0-30 points)
    priority_scores = {
        'Critical': 30,
        'High': 20,
        'Medium': 10,
        'Low': 5
    }
    score += priority_scores.get(priority, 10)
    
    # Status component (0-20 points)
    status_scores = {
        'Blocked': 20,
        'On Hold': 15,
        'In Progress': 10,
        'To Do': 5,
        'Done': 0
    }
    score += status_scores.get(status, 5)
    
    return min(score, 100)  # Cap at 100


def suggest_root_cause(row, bottleneck_type):
    """
    Suggest likely root cause based on task attributes and bottleneck type
    """
    suggestions = []
    
    if bottleneck_type == 'Blocked':
        if pd.isna(row.get('comments')) or str(row.get('comments')).strip() == '':
            suggestions.append("No blocking details in comments - follow up with assignee")
        else:
            suggestions.append("Task explicitly marked as blocked - check external dependencies")
    
    elif bottleneck_type == 'Duration_Anomaly':
        if row.get('reassignment_count', 0) > 1:
            suggestions.append("Multiple reassignments may have caused delays")
        if row.get('priority') == 'Low':
            suggestions.append("Low priority tasks often deprioritized - consider adjusting priority")
        else:
            suggestions.append("Task complexity may be underestimated - review scope")
    
    elif bottleneck_type == 'Start_Delay':
        suggestions.append("Task sat in backlog too long - improve sprint planning")
        suggestions.append("Consider automating task assignment for faster pickup")
    
    elif bottleneck_type == 'Assignee_Bottleneck':
        suggestions.append(f"Assignee {row.get('assignee')} may be overloaded - check workload balance")
        suggestions.append("Consider task reassignment or additional resources")
    
    elif bottleneck_type == 'Missing_Closure':
        suggestions.append("Task marked done but no end date - enforce completion tracking")
    
    else:
        suggestions.append("Review task workflow and identify blockers")
    
    return " | ".join(suggestions)


def log_bottleneck_history(task_id, bottleneck_type, severity_score, delay_days, priority, root_cause):
    """Log bottleneck to history table for tracking"""
    engine = get_engine()
    
    try:
        with engine.connect() as conn:
            query = text("""
                INSERT INTO bottleneck_history 
                (task_id, bottleneck_type, severity_score, delay_days, priority, root_cause_suggestion)
                VALUES (:task_id, :bottleneck_type, :severity_score, :delay_days, :priority, :root_cause)
            """)
            
            conn.execute(query, {
                'task_id': task_id,
                'bottleneck_type': bottleneck_type,
                'severity_score': severity_score,
                'delay_days': delay_days,
                'priority': priority,
                'root_cause': root_cause
            })
            conn.commit()
    
    except Exception as e:
        logger.error(f"Failed to log bottleneck history: {str(e)}")


def train_ml_resolution_estimator(df):
    """
    Train ML model to predict resolution time for bottlenecks
    Returns trained model and feature names
    """
    logger.info("[ML] Training ML resolution time estimator...")
    
    # Filter resolved bottlenecks
    query = """
    SELECT 
        bh.*,
        t.actual_duration,
        t.status,
        (julianday(CURRENT_TIMESTAMP) - julianday(bh.detected_date)) * 24 as hours_to_resolve
    FROM bottleneck_history bh
    JOIN tasks t ON bh.task_id = t.task_id
    WHERE bh.resolution_date IS NOT NULL
    """
    
    try:
        training_data = execute_query(query)
        
        if len(training_data) < 10:
            logger.warning("Not enough resolved bottlenecks for ML training (need 10+)")
            return None, None
        
        # Prepare features
        features = pd.DataFrame({
            'severity_score': training_data['severity_score'],
            'delay_days': training_data['delay_days'],
            'priority_encoded': training_data['priority'].map({'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4})
        })
        
        target = training_data['hours_to_resolve']
        
        # Train simple linear regression
        model = LinearRegression()
        model.fit(features, target)
        
        logger.info(f"[SUCCESS] ML model trained on {len(training_data)} resolved bottlenecks")
        logger.info(f"   Model R² score: {model.score(features, target):.3f}")
        
        return model, features.columns.tolist()
    
    except Exception as e:
        logger.error(f"ML training failed: {str(e)}")
        return None, None


def predict_resolution_time(model, feature_names, severity_score, delay_days, priority):
    """Predict resolution time using trained ML model"""
    if model is None:
        return None
    
    try:
        priority_encoded = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}.get(priority, 2)
        features = pd.DataFrame([[severity_score, delay_days, priority_encoded]], 
                               columns=feature_names)
        prediction = model.predict(features)[0]
        return max(prediction, 1)  # Minimum 1 hour
    except:
        return None


def load_tasks():
    """Load all tasks from database"""
    query = "SELECT * FROM tasks"
    df = execute_query(query)
    
    # Convert date strings back to datetime
    date_columns = ['created_date', 'start_date', 'end_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df


def calculate_baseline_metrics(df):
    """Calculate baseline duration metrics"""
    logger.info("[STATS] Calculating baseline metrics...")
    
    completed_tasks = df[df['actual_duration'].notna()]
    
    baseline = {
        'overall_mean': completed_tasks['actual_duration'].mean(),
        'overall_std': completed_tasks['actual_duration'].std(),
        'overall_median': completed_tasks['actual_duration'].median(),
        'by_assignee': {},
        'by_project': {},
        'by_priority': {}
    }
    
    # By assignee
    for assignee in df['assignee'].unique():
        assignee_tasks = completed_tasks[completed_tasks['assignee'] == assignee]
        if len(assignee_tasks) > 0:
            baseline['by_assignee'][assignee] = {
                'mean': assignee_tasks['actual_duration'].mean(),
                'std': assignee_tasks['actual_duration'].std(),
                'count': len(assignee_tasks)
            }
    
    # By project
    for project in df['project'].unique():
        project_tasks = completed_tasks[completed_tasks['project'] == project]
        if len(project_tasks) > 0:
            baseline['by_project'][project] = {
                'mean': project_tasks['actual_duration'].mean(),
                'std': project_tasks['actual_duration'].std(),
                'count': len(project_tasks)
            }
    
    logger.info(f"[SUCCESS] Baseline: mean={baseline['overall_mean']:.1f}d, median={baseline['overall_median']:.1f}d")
    
    return baseline


def detect_all_bottlenecks(df, baseline, ml_model=None, ml_features=None):
    """Main bottleneck detection with severity scoring and root cause"""
    logger.info("\n[INFO] Running comprehensive bottleneck detection...")
    
    bottlenecks = []
    
    for idx, row in df.iterrows():
        task_bottlenecks = []
        
        # Check for blocked status
        if row['status'] == 'Blocked':
            task_bottlenecks.append('Blocked')
        
        # Check duration anomaly
        if pd.notna(row['actual_duration']) and row['actual_duration'] > 0:
            assignee_baseline = baseline['by_assignee'].get(row['assignee'])
            
            if assignee_baseline and assignee_baseline['count'] >= 3:
                mean = assignee_baseline['mean']
                std = assignee_baseline['std']
            else:
                mean = baseline['overall_mean']
                std = baseline['overall_std']
            
            threshold = mean + (DELAY_THRESHOLD_MULTIPLIER * std)
            if row['actual_duration'] > threshold:
                task_bottlenecks.append('Duration_Anomaly')
        
        # Check start delay
        if pd.notna(row['created_date']) and pd.notna(row['start_date']):
            gap = (row['start_date'] - row['created_date']).days
            if gap > START_DELAY_DAYS:
                task_bottlenecks.append('Start_Delay')
        
        # Check assignee overload
        if row.get('reassignment_count', 0) >= REASSIGNMENT_THRESHOLD:
            task_bottlenecks.append('Assignee_Bottleneck')
        
        # Check missing closure
        if row['status'] == 'Done' and pd.isna(row['end_date']):
            task_bottlenecks.append('Missing_Closure')
        
        # If bottlenecks found, calculate severity and log
        if task_bottlenecks:
            bottleneck_type = ', '.join(task_bottlenecks)
            delay_days = max(0, row.get('actual_duration', 0) - baseline['overall_mean'])
            severity_score = calculate_severity_score(delay_days, row['priority'], row['status'])
            root_cause = suggest_root_cause(row, task_bottlenecks[0])
            
            # Predict resolution time if ML model available
            resolution_est = None
            if ml_model:
                resolution_est = predict_resolution_time(ml_model, ml_features, 
                                                        severity_score, delay_days, row['priority'])
            
            bottlenecks.append({
                'task_id': row['task_id'],
                'task_name': row['task_name'],
                'assignee': row['assignee'],
                'project': row['project'],
                'bottleneck_type': bottleneck_type,
                'severity_score': severity_score,
                'delay_days': delay_days,
                'priority': row['priority'],
                'status': row['status'],
                'root_cause': root_cause,
                'estimated_resolution_hours': resolution_est
            })
            
            # Log to history
            log_bottleneck_history(row['task_id'], bottleneck_type, severity_score, 
                                  delay_days, row['priority'], root_cause)
    
    logger.info(f"[SUCCESS] Detected {len(bottlenecks)} bottlenecks")
    
    return pd.DataFrame(bottlenecks)


def update_tasks_with_bottlenecks(bottlenecks_df):
    """Update tasks table with bottleneck information"""
    engine = get_engine()
    
    try:
        with engine.connect() as conn:
            for _, row in bottlenecks_df.iterrows():
                query = text("""
                    UPDATE tasks 
                    SET bottleneck_type = :bottleneck_type,
                        is_delayed = 1
                    WHERE task_id = :task_id
                """)
                
                conn.execute(query, {
                    'task_id': row['task_id'],
                    'bottleneck_type': row['bottleneck_type']
                })
            
            conn.commit()
        
        logger.info(f"[SUCCESS] Updated {len(bottlenecks_df)} tasks with bottleneck flags")
    
    except Exception as e:
        logger.error(f"Failed to update tasks: {str(e)}")


def generate_auto_summary_report(bottlenecks_df, output_path=None):
    """Generate automatic text summary report of bottlenecks"""
    if output_path is None:
        reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'exports')
        os.makedirs(reports_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(reports_dir, f'bottleneck_summary_{timestamp}.md')
    
    with open(output_path, 'w') as f:
        f.write("# Bottleneck Detection Summary Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Overall stats
        f.write("## Overall Statistics\n\n")
        f.write(f"- **Total Bottlenecks Detected:** {len(bottlenecks_df)}\n")
        
        if len(bottlenecks_df) > 0:
            f.write(f"- **Average Severity Score:** {bottlenecks_df['severity_score'].mean():.1f}/100\n")
            f.write(f"- **High Severity (>70):** {len(bottlenecks_df[bottlenecks_df['severity_score'] > 70])}\n")
            f.write(f"- **Average Delay:** {bottlenecks_df['delay_days'].mean():.1f} days\n\n")
            
            # By type
            f.write("## Bottlenecks by Type\n\n")
            type_counts = bottlenecks_df['bottleneck_type'].value_counts()
            for btype, count in type_counts.items():
                f.write(f"- **{btype}:** {count}\n")
            f.write("\n")
            
            # By assignee
            f.write("## Bottlenecks by Assignee\n\n")
            assignee_counts = bottlenecks_df['assignee'].value_counts().head(10)
            for assignee, count in assignee_counts.items():
                avg_severity = bottlenecks_df[bottlenecks_df['assignee'] == assignee]['severity_score'].mean()
                f.write(f"- **{assignee}:** {count} bottlenecks (avg severity: {avg_severity:.1f})\n")
            f.write("\n")
            
            # Top delays
            f.write("## Top 10 Delays by Severity\n\n")
            top_delays = bottlenecks_df.nlargest(10, 'severity_score')
            for _, row in top_delays.iterrows():
                f.write(f"### {row['task_name']}\n")
                f.write(f"- **Task ID:** {row['task_id']}\n")
                f.write(f"- **Assignee:** {row['assignee']}\n")
                f.write(f"- **Project:** {row['project']}\n")
                f.write(f"- **Type:** {row['bottleneck_type']}\n")
                f.write(f"- **Severity:** {row['severity_score']:.0f}/100\n")
                f.write(f"- **Delay:** {row['delay_days']:.1f} days\n")
                f.write(f"- **Root Cause:** {row['root_cause']}\n")
                if pd.notna(row.get('estimated_resolution_hours')):
                    f.write(f"- **Est. Resolution:** {row['estimated_resolution_hours']:.1f} hours\n")
                f.write("\n")
            
            # Recommendations
            f.write("## Fix Recommendations\n\n")
            if len(bottlenecks_df[bottlenecks_df['bottleneck_type'].str.contains('Blocked')]) > 0:
                f.write("- **Blocked Tasks:** Schedule daily unblock meetings\n")
            if len(bottlenecks_df[bottlenecks_df['bottleneck_type'].str.contains('Assignee')]) > 0:
                f.write("- **Assignee Bottlenecks:** Review workload distribution and consider reassignments\n")
            if len(bottlenecks_df[bottlenecks_df['bottleneck_type'].str.contains('Start_Delay')]) > 0:
                f.write("- **Start Delays:** Improve sprint planning and task prioritization\n")
            if len(bottlenecks_df[bottlenecks_df['bottleneck_type'].str.contains('Duration')]) > 0:
                f.write("- **Duration Anomalies:** Review task complexity estimation process\n")
    
    logger.info(f"[REPORT] Auto-summary report saved: {output_path}")
    return output_path


def run_bottleneck_detection(save_report=True):
    """Main execution function"""
    logger.info("\n" + "="*70)
    logger.info("[INFO] BOTTLENECK DETECTION ENGINE v2.0")
    logger.info("="*70 + "\n")
    
    # Load data
    df = load_tasks()
    logger.info(f"[STATS] Loaded {len(df)} tasks")
    
    # Calculate baselines
    baseline = calculate_baseline_metrics(df)
    
    # Train ML model
    ml_model, ml_features = train_ml_resolution_estimator(df)
    
    # Detect bottlenecks
    bottlenecks_df = detect_all_bottlenecks(df, baseline, ml_model, ml_features)
    
    if len(bottlenecks_df) > 0:
        # Update database
        update_tasks_with_bottlenecks(bottlenecks_df)
        
        # Generate report
        if save_report:
            generate_auto_summary_report(bottlenecks_df)
        
        # Print summary
        logger.info("\n[STATS] SUMMARY")
        logger.info("="*70)
        logger.info(f"Total Bottlenecks: {len(bottlenecks_df)}")
        logger.info(f"Avg Severity: {bottlenecks_df['severity_score'].mean():.1f}/100")
        logger.info(f"High Severity (>70): {len(bottlenecks_df[bottlenecks_df['severity_score'] > 70])}")
        logger.info("\nTop Bottleneck Types:")
        for btype, count in bottlenecks_df['bottleneck_type'].value_counts().head(5).items():
            logger.info(f"  - {btype}: {count}")
    else:
        logger.info("[SUCCESS] No bottlenecks detected!")
    
    logger.info("\n" + "="*70 + "\n")
    
    return bottlenecks_df


if __name__ == "__main__":
    run_bottleneck_detection()
