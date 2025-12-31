"""
Bottleneck Detection Engine for FlowFix AI
Identifies workflow bottlenecks and classifies them by type
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import text
import os
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils import get_engine, execute_query

# Load environment variables
load_dotenv()

DELAY_THRESHOLD_MULTIPLIER = float(os.getenv('DELAY_THRESHOLD_MULTIPLIER', 1.5))
START_DELAY_DAYS = int(os.getenv('START_DELAY_DAYS', 3))
REASSIGNMENT_THRESHOLD = int(os.getenv('REASSIGNMENT_THRESHOLD', 2))


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
    """Calculate baseline duration metrics by assignee and project"""
    print("📊 Calculating baseline metrics...")
    
    # Overall baseline
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
    
    # By priority
    for priority in df['priority'].unique():
        priority_tasks = completed_tasks[completed_tasks['priority'] == priority]
        if len(priority_tasks) > 0:
            baseline['by_priority'][priority] = {
                'mean': priority_tasks['actual_duration'].mean(),
                'std': priority_tasks['actual_duration'].std(),
                'count': len(priority_tasks)
            }
    
    print(f"✅ Baseline metrics calculated")
    print(f"   Overall mean duration: {baseline['overall_mean']:.1f} days")
    print(f"   Overall median duration: {baseline['overall_median']:.1f} days")
    
    return baseline


def detect_duration_delays(df, baseline):
    """Detect tasks with abnormally long durations"""
    print("\n🔍 Detecting duration delays...")
    
    delay_flags = []
    
    for idx, row in df.iterrows():
        if pd.isna(row['actual_duration']) or row['actual_duration'] < 0:
            continue
        
        # Get appropriate baseline
        assignee_baseline = baseline['by_assignee'].get(row['assignee'])
        project_baseline = baseline['by_project'].get(row['project'])
        
        # Use assignee baseline if available, otherwise project or overall
        if assignee_baseline and assignee_baseline['count'] >= 3:
            mean = assignee_baseline['mean']
            std = assignee_baseline['std']
        elif project_baseline and project_baseline['count'] >= 3:
            mean = project_baseline['mean']
            std = project_baseline['std']
        else:
            mean = baseline['overall_mean']
            std = baseline['overall_std']
        
        # Check if delayed
        threshold = mean + (DELAY_THRESHOLD_MULTIPLIER * std)
        if row['actual_duration'] > threshold:
            delay_flags.append({
                'task_id': row['task_id'],
                'is_delayed': True,
                'expected_duration': mean,
                'actual_duration': row['actual_duration'],
                'delay_days': row['actual_duration'] - mean
            })
    
    print(f"✅ Found {len(delay_flags)} delayed tasks")
    return delay_flags


def detect_start_delays(df):
    """Detect tasks with significant gap between creation and start"""
    print("\n🔍 Detecting start delays...")
    
    start_delays = []
    
    for idx, row in df.iterrows():
        if pd.notna(row['created_date']) and pd.notna(row['start_date']):
            gap = (row['start_date'] - row['created_date']).days
            
            if gap > START_DELAY_DAYS:
                start_delays.append({
                    'task_id': row['task_id'],
                    'start_delay_days': gap
                })
    
    print(f"✅ Found {len(start_delays)} tasks with start delays")
    return start_delays


def detect_missing_closure(df):
    """Detect tasks without end dates that should be completed"""
    print("\n🔍 Detecting missing closure...")
    
    current_date = datetime.now()
    missing_closure = []
    
    for idx, row in df.iterrows():
        # Task has start date but no end date, and has been running for a while
        if pd.notna(row['start_date']) and pd.isna(row['end_date']):
            days_running = (current_date - row['start_date']).days
            
            # If running longer than 30 days or marked as Done but no end date
            if days_running > 30 or row['status'] == 'Done':
                missing_closure.append({
                    'task_id': row['task_id'],
                    'days_running': days_running,
                    'status': row['status']
                })
    
    print(f"✅ Found {len(missing_closure)} tasks missing closure")
    return missing_closure


def detect_blocked_tasks(df):
    """Identify tasks marked as blocked"""
    print("\n🔍 Detecting blocked tasks...")
    
    blocked = df[df['status'] == 'Blocked']
    
    blocked_list = []
    for idx, row in blocked.iterrows():
        blocked_list.append({
            'task_id': row['task_id'],
            'assignee': row['assignee'],
            'project': row['project'],
            'comments': row['comments']
        })
    
    print(f"✅ Found {len(blocked_list)} blocked tasks")
    return blocked_list


def classify_bottleneck(row, delay_info, start_delays, missing_closure, blocked_tasks):
    """Classify the type of bottleneck for a task"""
    task_id = row['task_id']
    
    # Check various bottleneck indicators
    is_delayed = any(d['task_id'] == task_id for d in delay_info)
    has_start_delay = any(d['task_id'] == task_id for d in start_delays)
    has_missing_closure = any(d['task_id'] == task_id for d in missing_closure)
    is_blocked = any(d['task_id'] == task_id for d in blocked_tasks)
    
    # Classification logic
    if is_blocked:
        return 'Blocked'
    elif has_missing_closure and row['status'] == 'Done':
        return 'Administrative'
    elif has_missing_closure and row['status'] == 'In_Review':
        return 'Review_Bottleneck'
    elif has_missing_closure:
        return 'Stalled'
    elif is_delayed and row['status'] == 'In_Review':
        return 'Review_Bottleneck'
    elif is_delayed:
        return 'Assignee_Bottleneck'
    elif has_start_delay:
        return 'Resource_Availability'
    else:
        return None


def update_bottleneck_flags(df, delay_info, start_delays, missing_closure, blocked_tasks):
    """Update database with bottleneck classifications"""
    print("\n🏷️  Classifying bottlenecks...")
    
    engine = get_engine()
    updates = []
    
    for idx, row in df.iterrows():
        task_id = row['task_id']
        
        # Classify bottleneck
        bottleneck_type = classify_bottleneck(
            row, delay_info, start_delays, missing_closure, blocked_tasks
        )
        
        # Check if delayed
        is_delayed = any(d['task_id'] == task_id for d in delay_info)
        
        if bottleneck_type or is_delayed:
            updates.append({
                'task_id': task_id,
                'bottleneck_type': bottleneck_type if bottleneck_type else '',  # Empty string for Power BI compatibility
                'is_delayed': 1 if is_delayed else 0
            })
    
    # Batch update database
    if updates:
        with engine.connect() as conn:
            for update in updates:
                query = text("""
                    UPDATE tasks 
                    SET bottleneck_type = :bottleneck_type,
                        is_delayed = :is_delayed
                    WHERE task_id = :task_id
                """)
                conn.execute(query, update)
            conn.commit()
    
    print(f"✅ Updated {len(updates)} tasks with bottleneck classifications")
    
    # Summary by type
    bottleneck_summary = {}
    for update in updates:
        btype = update['bottleneck_type']
        if btype:
            bottleneck_summary[btype] = bottleneck_summary.get(btype, 0) + 1
    
    if bottleneck_summary:
        print("\n📈 Bottleneck Summary:")
        for btype, count in sorted(bottleneck_summary.items(), key=lambda x: x[1], reverse=True):
            print(f"   {btype}: {count}")
    
    return updates


def save_bottleneck_summary(updates):
    """Save bottleneck summary to database"""
    engine = get_engine()
    
    with engine.connect() as conn:
        # Clear existing summary
        conn.execute(text("DELETE FROM bottleneck_summary"))
        
        # Insert new records
        for update in updates:
            if update['bottleneck_type']:
                query = text("""
                    INSERT INTO bottleneck_summary (task_id, bottleneck_type, severity)
                    VALUES (:task_id, :bottleneck_type, :severity)
                """)
                
                # Determine severity based on type
                severity = 'High' if update['bottleneck_type'] == 'Blocked' else 'Medium'
                
                conn.execute(query, {
                    'task_id': update['task_id'],
                    'bottleneck_type': update['bottleneck_type'],
                    'severity': severity
                })
        
        conn.commit()
    
    print(f"✅ Saved bottleneck summary")


def cluster_bottlenecks_ml(df):
    """Apply ML clustering to find bottleneck patterns"""
    print("\n🤖 Applying ML clustering to bottleneck patterns...")
    
    # Filter to only tasks with bottlenecks
    bottleneck_df = df[df['bottleneck_type'] != ''].copy()
    
    if len(bottleneck_df) < 10:
        print("⚠️ Not enough bottlenecks for clustering (need at least 10)")
        return None
    
    # Prepare features for clustering
    features = []
    
    # Numeric features
    if 'actual_duration' in bottleneck_df.columns:
        features.append(bottleneck_df['actual_duration'].fillna(0))
    
    # Delay flag
    if 'is_delayed' in bottleneck_df.columns:
        features.append(bottleneck_df['is_delayed'])
    
    # Status encoding
    if 'status' in bottleneck_df.columns:
        status_map = {'To_Do': 0, 'In_Progress': 1, 'In_Review': 2, 'Done': 3, 'Blocked': 4}
        features.append(bottleneck_df['status'].map(status_map).fillna(0))
    
    # Priority encoding
    if 'priority' in bottleneck_df.columns:
        priority_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
        features.append(bottleneck_df['priority'].map(priority_map).fillna(1))
    
    # Stack features
    X = np.column_stack(features)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal number of clusters
    n_clusters = min(4, len(bottleneck_df) // 10)
    n_clusters = max(2, n_clusters)  # At least 2 clusters
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels
    bottleneck_df['cluster'] = clusters
    
    # Analyze clusters
    print(f"\n📊 Identified {n_clusters} bottleneck patterns:")
    
    for i in range(n_clusters):
        cluster_tasks = bottleneck_df[bottleneck_df['cluster'] == i]
        
        print(f"\n   Pattern {i+1}:")
        print(f"      Size: {len(cluster_tasks)} tasks")
        print(f"      Avg Duration: {cluster_tasks['actual_duration'].mean():.1f} days")
        print(f"      Delay Rate: {cluster_tasks['is_delayed'].mean()*100:.1f}%")
        print(f"      Common Type: {cluster_tasks['bottleneck_type'].mode()[0] if len(cluster_tasks) > 0 else 'Unknown'}")
    
    print("\n✅ ML clustering complete")
    return bottleneck_df


def analyze_bottlenecks():
    """Main function to run complete bottleneck analysis"""
    print("\n" + "="*60)
    print("🔍 BOTTLENECK DETECTION ENGINE")
    print("="*60 + "\n")
    
    # Load tasks
    df = load_tasks()
    print(f"📊 Loaded {len(df)} tasks from database\n")
    
    # Calculate baselines
    baseline = calculate_baseline_metrics(df)
    
    # Run detection algorithms
    delay_info = detect_duration_delays(df, baseline)
    start_delays = detect_start_delays(df)
    missing_closure = detect_missing_closure(df)
    blocked_tasks = detect_blocked_tasks(df)
    
    # Update flags and classifications
    updates = update_bottleneck_flags(df, delay_info, start_delays, missing_closure, blocked_tasks)
    
    # Save summary
    save_bottleneck_summary(updates)
    
    # Apply ML clustering (optional advanced analysis)
    cluster_bottlenecks_ml(df)
    
    print("\n" + "="*60)
    print("✅ BOTTLENECK ANALYSIS COMPLETE")
    print("="*60 + "\n")
    
    return {
        'total_bottlenecks': len(updates),
        'delayed_tasks': len(delay_info),
        'start_delays': len(start_delays),
        'missing_closure': len(missing_closure),
        'blocked_tasks': len(blocked_tasks)
    }


if __name__ == "__main__":
    result = analyze_bottlenecks()
    
    print("\n📋 Final Summary:")
    print(f"   Total Bottlenecks Detected: {result['total_bottlenecks']}")
    print(f"   Duration Delays: {result['delayed_tasks']}")
    print(f"   Start Delays: {result['start_delays']}")
    print(f"   Missing Closure: {result['missing_closure']}")
    print(f"   Blocked Tasks: {result['blocked_tasks']}")
