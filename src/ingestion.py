"""
Production-Grade Data Ingestion Module for FlowFix AI
Handles CSV/Excel import with merge logic, validation, dry-run, and error tracking
Version: 2.0
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import uuid
import yaml
import logging
from sqlalchemy import text
from utils import get_engine, create_schema, execute_query

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_column_mapping():
    """Load column mapping from YAML config file"""
    config_path = os.path.join(os.path.dirname(__file__), 'column_mapping.yaml')
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("column_mapping.yaml not found, using default mapping")
        return {'generic': {}}


def generate_task_id():
    """Generate unique task ID using UUID4"""
    return f"TASK-{uuid.uuid4().hex[:8].upper()}"


def normalize_date(date_value):
    """Normalize dates to YYYY-MM-DD format"""
    if pd.isna(date_value) or date_value == '':
        return None
    
    date_formats = [
        '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d',
        '%d-%m-%Y', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S',
        '%d/%m/%Y %H:%M:%S', '%Y/%m/%d %H:%M:%S'
    ]
    
    for fmt in date_formats:
        try:
            parsed = datetime.strptime(str(date_value), fmt)
            return parsed.strftime('%Y-%m-%d')
        except ValueError:
            continue
    
    # Try pandas parser
    try:
        parsed = pd.to_datetime(date_value)
        return parsed.strftime('%Y-%m-%d')
    except:
        logger.warning(f"Could not parse date: {date_value}")
        return None


def validate_row(row, row_num):
    """Validate a single row and return validation errors"""
    errors = []
    
    # Required fields
    required_fields = ['task_name', 'assignee', 'status', 'priority', 'project']
    for field in required_fields:
        if pd.isna(row.get(field)) or str(row.get(field)).strip() == '':
            errors.append(f"Missing required field: {field}")
    
    # Validate status
    valid_statuses = ['To Do', 'In Progress', 'Done', 'Blocked', 'On Hold']
    if not pd.isna(row.get('status')) and row['status'] not in valid_statuses:
        errors.append(f"Invalid status: {row['status']}")
    
    # Validate priority
    valid_priorities = ['Low', 'Medium', 'High', 'Critical']
    if not pd.isna(row.get('priority')) and row['priority'] not in valid_priorities:
        errors.append(f"Invalid priority: {row['priority']}")
    
    # Validate durations
    if not pd.isna(row.get('actual_duration')) and row['actual_duration'] < 0:
        errors.append(f"Negative duration: {row['actual_duration']}")
    
    return errors


def clean_and_validate_data(df, source_type='generic'):
    """Clean, normalize, and validate imported data"""
    logger.info("[INFO] Cleaning and validating data...")
    
    # Load column mapping
    column_mappings = load_column_mapping()
    mapping = column_mappings.get(source_type, column_mappings.get('generic', {}))
    
    # Apply column mapping if needed
    if mapping:
        df = df.rename(columns=mapping)
    
    # Normalize column names
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    
    # Generate task_id if missing
    if 'task_id' not in df.columns or df['task_id'].isna().any():
        logger.info("Generating missing task IDs...")
        df.loc[df['task_id'].isna(), 'task_id'] = [generate_task_id() for _ in range(df['task_id'].isna().sum())]
    
    # Normalize dates globally
    date_columns = ['created_date', 'start_date', 'end_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = df[col].apply(normalize_date)
    
    # Set defaults for missing values
    defaults = {
        'task_name': 'Unnamed Task',
        'assignee': 'Unassigned',
        'status': 'To Do',
        'priority': 'Medium',
        'comments': '',
        'project': 'Default Project',
        'planned_duration': 0,
        'actual_duration': 0,
        'is_delayed': 0,
        'is_overdue': 0,
        'bottleneck_type': '',
        'reassignment_count': 0
    }
    
    for col, default_val in defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default_val)
    
    # Calculate durations if not present
    if 'actual_duration' in df.columns and 'start_date' in df.columns and 'end_date' in df.columns:
        for idx, row in df.iterrows():
            if pd.isna(row['actual_duration']) or row['actual_duration'] == 0:
                if not pd.isna(row['start_date']) and not pd.isna(row['end_date']):
                    try:
                        start = pd.to_datetime(row['start_date'])
                        end = pd.to_datetime(row['end_date'])
                        df.at[idx, 'actual_duration'] = (end - start).days
                    except:
                        pass
    
    # Validate all rows and collect failures
    failed_rows = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        errors = validate_row(row, idx)
        if errors:
            failed_rows.append({
                'row_number': idx + 2,  # +2 for header and 0-indexing
                'task_id': row.get('task_id', 'N/A'),
                'errors': '; '.join(errors),
                'raw_data': row.to_dict()
            })
        else:
            valid_indices.append(idx)
    
    # Separate valid and failed data
    valid_df = df.loc[valid_indices].copy()
    
    logger.info(f"[SUCCESS] Valid records: {len(valid_df)}")
    if failed_rows:
        logger.warning(f"[WARNING] Failed records: {len(failed_rows)}")
    
    return valid_df, failed_rows


def save_failed_rows(failed_rows, filename):
    """Save failed rows to CSV for review"""
    if not failed_rows:
        return
    
    failed_df = pd.DataFrame(failed_rows)
    exports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'exports')
    os.makedirs(exports_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(exports_dir, f'failed_rows_{timestamp}.csv')
    failed_df.to_csv(output_path, index=False)
    
    logger.info(f"[INFO] Failed rows saved to: {output_path}")
    return output_path


def merge_or_append_data(df, mode='merge', dry_run=False):
    """Merge or append data to database"""
    engine = get_engine()
    
    if dry_run:
        logger.info("[INFO] DRY RUN MODE - No data will be written")
        logger.info(f"   Would process {len(df)} records")
        logger.info(f"   Mode: {mode}")
        return 0
    
    records_added = 0
    
    with engine.connect() as conn:
        if mode == 'replace':
            logger.warning("[WARNING] REPLACE mode - clearing existing data")
            conn.execute(text("DELETE FROM tasks"))
            conn.commit()
        
        for idx, row in df.iterrows():
            try:
                if mode == 'merge':
                    # Check if task exists
                    result = conn.execute(
                        text("SELECT task_id FROM tasks WHERE task_id = :task_id"),
                        {'task_id': row['task_id']}
                    )
                    
                    if result.fetchone():
                        # Update existing
                        update_query = text("""
                            UPDATE tasks SET
                                task_name = :task_name,
                                assignee = :assignee,
                                status = :status,
                                created_date = :created_date,
                                start_date = :start_date,
                                end_date = :end_date,
                                priority = :priority,
                                comments = :comments,
                                project = :project,
                                planned_duration = :planned_duration,
                                actual_duration = :actual_duration,
                                is_delayed = :is_delayed,
                                is_overdue = :is_overdue,
                                bottleneck_type = :bottleneck_type,
                                reassignment_count = :reassignment_count
                            WHERE task_id = :task_id
                        """)
                        conn.execute(update_query, row.to_dict())
                    else:
                        # Insert new
                        insert_query = text("""
                            INSERT INTO tasks (task_id, task_name, assignee, status, created_date,
                                             start_date, end_date, priority, comments, project,
                                             planned_duration, actual_duration, is_delayed,
                                             is_overdue, bottleneck_type, reassignment_count)
                            VALUES (:task_id, :task_name, :assignee, :status, :created_date,
                                   :start_date, :end_date, :priority, :comments, :project,
                                   :planned_duration, :actual_duration, :is_delayed,
                                   :is_overdue, :bottleneck_type, :reassignment_count)
                        """)
                        conn.execute(insert_query, row.to_dict())
                        records_added += 1
                else:  # append mode
                    insert_query = text("""
                        INSERT INTO tasks (task_id, task_name, assignee, status, created_date,
                                         start_date, end_date, priority, comments, project,
                                         planned_duration, actual_duration, is_delayed,
                                         is_overdue, bottleneck_type, reassignment_count)
                        VALUES (:task_id, :task_name, :assignee, :status, :created_date,
                               :start_date, :end_date, :priority, :comments, :project,
                               :planned_duration, :actual_duration, :is_delayed,
                               :is_overdue, :bottleneck_type, :reassignment_count)
                    """)
                    conn.execute(insert_query, row.to_dict())
                    records_added += 1
            
            except Exception as e:
                logger.error(f"Error processing row {idx}: {str(e)}")
                continue
        
        conn.commit()
    
    return records_added


def log_ingestion(filename, records_added, records_failed, file_size_kb, processing_time_sec, errors=''):
    """Log ingestion metrics to database"""
    engine = get_engine()
    
    try:
        with engine.connect() as conn:
            query = text("""
                INSERT INTO ingestion_log 
                (filename, records_added, records_failed, file_size_kb, 
                 processing_time_sec, error_details)
                VALUES (:filename, :records_added, :records_failed, :file_size_kb,
                       :processing_time_sec, :error_details)
            """)
            
            conn.execute(query, {
                'filename': filename,
                'records_added': records_added,
                'records_failed': records_failed,
                'file_size_kb': file_size_kb,
                'processing_time_sec': processing_time_sec,
                'error_details': errors
            })
            conn.commit()
        
        logger.info("[SUCCESS] Ingestion logged to database")
    except Exception as e:
        logger.error(f"Failed to log ingestion: {str(e)}")


def ingest_file(filepath, source_type='generic', mode='merge', dry_run=False):
    """
    Main ingestion function with production features
    
    Args:
        filepath: Path to CSV or Excel file
        source_type: Type of source system (generic, jira, trello, etc.)
        mode: 'merge' (update existing), 'append' (add new), or 'replace' (clear all)
        dry_run: If True, simulate ingestion without writing
    """
    start_time = datetime.now()
    
    logger.info("\n" + "="*70)
    logger.info("[INFO] PRODUCTION DATA INGESTION")
    logger.info("="*70)
    logger.info(f"File: {filepath}")
    logger.info(f"Source Type: {source_type}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Dry Run: {dry_run}")
    logger.info("")
    
    # Ensure schema exists
    create_schema()
    
    # Load file
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")
        
        file_size_kb = os.path.getsize(filepath) / 1024
        logger.info(f"[STATS] Loaded {len(df)} records ({file_size_kb:.2f} KB)")
    
    except Exception as e:
        logger.error(f"[ERROR] Failed to load file: {str(e)}")
        return False
    
    # Clean and validate
    valid_df, failed_rows = clean_and_validate_data(df, source_type)
    
    # Save failed rows
    if failed_rows:
        save_failed_rows(failed_rows, os.path.basename(filepath))
    
    # Merge or append data
    records_added = merge_or_append_data(valid_df, mode, dry_run)
    
    # Calculate processing time
    processing_time = (datetime.now() - start_time).total_seconds()
    
    # Log ingestion
    if not dry_run:
        log_ingestion(
            filename=os.path.basename(filepath),
            records_added=records_added,
            records_failed=len(failed_rows),
            file_size_kb=file_size_kb,
            processing_time_sec=processing_time
        )
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("[STATS] INGESTION SUMMARY")
    logger.info("="*70)
    logger.info(f"[SUCCESS] Valid records processed: {len(valid_df)}")
    logger.info(f"[INFO] Records added/updated: {records_added}")
    logger.info(f"[ERROR] Failed records: {len(failed_rows)}")
    logger.info(f"[INFO] Processing time: {processing_time:.2f}s")
    logger.info(f"[STATS] Rate: {len(valid_df)/processing_time:.0f} records/sec")
    logger.info("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='FlowFix AI Data Ingestion')
    parser.add_argument('filepath', help='Path to CSV or Excel file')
    parser.add_argument('--source', default='generic', help='Source type (generic, jira, trello, etc.)')
    parser.add_argument('--mode', default='merge', choices=['merge', 'append', 'replace'], 
                       help='Ingestion mode')
    parser.add_argument('--dry-run', action='store_true', help='Simulate ingestion without writing')
    
    args = parser.parse_args()
    
    ingest_file(args.filepath, args.source, args.mode, args.dry_run)
