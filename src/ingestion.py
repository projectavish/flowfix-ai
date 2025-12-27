"""
Data Ingestion Module for FlowFix AI
Handles CSV/Excel import, data cleaning, and loading into database
"""
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine
import os
import sys
from utils import get_engine, create_schema

# Column mapping for different data sources
COLUMN_MAPPINGS = {
    'trello': {
        'Card Name': 'task_name',
        'Member': 'assignee',
        'List': 'status',
        'Due Date': 'end_date',
        'Description': 'comments',
        'Labels': 'priority'
    },
    'jira': {
        'Issue key': 'task_id',
        'Summary': 'task_name',
        'Assignee': 'assignee',
        'Status': 'status',
        'Created': 'created_date',
        'Updated': 'start_date',
        'Resolved': 'end_date',
        'Priority': 'priority',
        'Project name': 'project'
    }
}


def detect_date_format(date_string):
    """Detect and parse various date formats"""
    if pd.isna(date_string) or date_string == '':
        return None
    
    date_formats = [
        '%Y-%m-%d',
        '%m/%d/%Y',
        '%d/%m/%Y',
        '%Y/%m/%d',
        '%d-%m-%Y',
        '%Y-%m-%d %H:%M:%S',
        '%m/%d/%Y %H:%M:%S'
    ]
    
    for fmt in date_formats:
        try:
            return datetime.strptime(str(date_string), fmt)
        except ValueError:
            continue
    
    # Try pandas parser as fallback
    try:
        return pd.to_datetime(date_string)
    except:
        return None


def clean_data(df):
    """Clean and normalize the imported data"""
    print("🧹 Cleaning data...")
    
    # Make a copy
    df = df.copy()
    
    # Normalize column names
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    
    # Rename to standard schema if needed
    column_map = {
        'task_id': 'task_id',
        'task_name': 'task_name',
        'assignee': 'assignee',
        'status': 'status',
        'created_date': 'created_date',
        'start_date': 'start_date',
        'end_date': 'end_date',
        'priority': 'priority',
        'comments': 'comments',
        'project': 'project'
    }
    
    # Handle date columns
    date_columns = ['created_date', 'start_date', 'end_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = df[col].apply(detect_date_format)
    
    # Clean text columns with Power BI compatible defaults
    text_columns = {
        'task_name': 'Unnamed Task',
        'assignee': 'Unassigned',
        'status': 'Unknown',
        'priority': 'Medium',
        'project': 'Unknown',
        'comments': ''
    }
    for col, default_value in text_columns.items():
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(['nan', 'None', ''], default_value)
            df[col] = df[col].fillna(default_value)
    
    # Standardize status values
    status_map = {
        'In Progress': 'In_Progress',
        'In Review': 'In_Review',
        'To Do': 'To_Do',
        'Done': 'Done',
        'Blocked': 'Blocked'
    }
    if 'status' in df.columns:
        df['status'] = df['status'].replace(status_map)
    
    # Standardize priority
    if 'priority' in df.columns:
        df['priority'] = df['priority'].str.title()
    
    # Calculate durations
    df['planned_duration'] = np.nan
    df['actual_duration'] = np.nan
    
    if 'start_date' in df.columns and 'end_date' in df.columns:
        mask = df['start_date'].notna() & df['end_date'].notna()
        df.loc[mask, 'actual_duration'] = (
            df.loc[mask, 'end_date'] - df.loc[mask, 'start_date']
        ).dt.days
    
    # Initialize flags with Power BI compatible defaults
    df['is_delayed'] = 0
    df['is_overdue'] = 0
    df['bottleneck_type'] = ''  # Empty string instead of NULL for Power BI
    
    # Mark overdue tasks (end date is null and past start date)
    if 'start_date' in df.columns:
        current_date = datetime.now()
        overdue_mask = (
            (df['end_date'].isna()) & 
            (df['start_date'].notna()) &
            (df['status'] != 'Done') &
            ((current_date - df['start_date']).dt.days > 14)
        )
        df.loc[overdue_mask, 'is_overdue'] = 1
    
    print(f"✅ Cleaned {len(df)} records")
    return df


def validate_data(df):
    """Validate required columns and data quality"""
    print("✔️  Validating data...")
    
    required_columns = ['task_id', 'task_name', 'assignee', 'status', 'priority', 'project']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for null values in critical columns
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        print(f"⚠️  Warning: Found null values:\n{null_counts[null_counts > 0]}")
    
    # Check date logic
    if 'start_date' in df.columns and 'end_date' in df.columns:
        invalid_dates = df[
            (df['start_date'].notna()) & 
            (df['end_date'].notna()) & 
            (df['end_date'] < df['start_date'])
        ]
        if len(invalid_dates) > 0:
            print(f"⚠️  Warning: {len(invalid_dates)} tasks have end_date before start_date")
    
    print(f"✅ Validation complete: {len(df)} valid records")
    return True


def load_to_database(df, table_name='tasks', if_exists='replace'):
    """Load cleaned data into database"""
    print(f"💾 Loading data to database table '{table_name}'...")
    
    engine = get_engine()
    
    # Convert datetime columns to string for SQLite compatibility
    date_columns = ['created_date', 'start_date', 'end_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = df[col].dt.strftime('%Y-%m-%d').where(df[col].notna(), None)
    
    # Load to database
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)
    
    print(f"✅ Loaded {len(df)} records to database")
    return True


def ingest_csv(file_path, source_type='standard'):
    """
    Main ingestion function
    
    Args:
        file_path: Path to CSV/Excel file
        source_type: 'standard', 'trello', or 'jira'
    """
    print(f"\n{'='*60}")
    print(f"🚀 Starting data ingestion from: {file_path}")
    print(f"{'='*60}\n")
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Read file
    print("📖 Reading file...")
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel.")
    
    print(f"✅ Read {len(df)} records with {len(df.columns)} columns")
    print(f"Columns: {', '.join(df.columns.tolist())}\n")
    
    # Apply column mapping if needed
    if source_type in COLUMN_MAPPINGS:
        df = df.rename(columns=COLUMN_MAPPINGS[source_type])
    
    # Clean data
    df = clean_data(df)
    
    # Validate data
    validate_data(df)
    
    # Create schema if not exists
    create_schema()
    
    # Load to database
    load_to_database(df)
    
    print(f"\n{'='*60}")
    print("✅ Data ingestion completed successfully!")
    print(f"{'='*60}\n")
    
    return df


def get_ingestion_summary(df):
    """Generate summary statistics of ingested data"""
    summary = {
        'total_records': len(df),
        'total_assignees': df['assignee'].nunique(),
        'total_projects': df['project'].nunique(),
        'status_distribution': df['status'].value_counts().to_dict(),
        'priority_distribution': df['priority'].value_counts().to_dict(),
        'date_range': {
            'earliest': df['created_date'].min(),
            'latest': df['created_date'].max()
        },
        'missing_end_dates': df['end_date'].isna().sum(),
        'tasks_with_comments': df['comments'].notna().sum()
    }
    
    return summary


if __name__ == "__main__":
    # Main execution
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Default to sample data
        file_path = "../data/FlowFixAI_FinalTaskData_1000.csv"
    
    # Ingest data
    df = ingest_csv(file_path)
    
    # Print summary
    summary = get_ingestion_summary(df)
    print("\n📊 Ingestion Summary:")
    print(f"Total Records: {summary['total_records']}")
    print(f"Total Assignees: {summary['total_assignees']}")
    print(f"Total Projects: {summary['total_projects']}")
    print(f"\nStatus Distribution:")
    for status, count in summary['status_distribution'].items():
        print(f"  {status}: {count}")
    print(f"\nMissing End Dates: {summary['missing_end_dates']}")
