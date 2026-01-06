"""
Fix database data types and NULLs for Power BI compatibility - Production Grade
Enhanced with data type standardization, orphan cleanup, and CLI options
"""
import os
import sys
import logging
import argparse
from datetime import datetime
from sqlalchemy import text
from utils import get_engine, execute_query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fix_null_values(dry_run: bool = False, verbose: bool = False) -> int:
    """
    Replace NULL values with appropriate defaults
    
    Args:
        dry_run: If True, only report what would be changed
        verbose: Show detailed information
        
    Returns:
        int: Total rows updated
    """
    logger.info("🔧 Fixing NULL values...")
    
    engine = get_engine()
    total_updates = 0
    
    fixes = [
        {
            'table': 'tasks',
            'updates': {
                'task_name': "''",
                'assignee': "'Unassigned'",
                'status': "'Unknown'",
                'priority': "'Medium'",
                'project': "'Unknown'",
                'comments': "''",
                'bottleneck_type': "''",
                'reassignment_count': '0'
            },
            'condition': """
                task_name IS NULL OR assignee IS NULL OR status IS NULL 
                OR priority IS NULL OR project IS NULL
            """
        },
        {
            'table': 'gpt_suggestions',
            'updates': {
                'root_causes': "'[]'",
                'recommendations': "'[]'",
                'suggestion_text': "''",
                'sentiment': "'neutral'",
                'urgency_level': "'medium'"
            },
            'condition': "root_causes IS NULL OR recommendations IS NULL"
        },
        {
            'table': 'bottleneck_history',
            'updates': {
                'root_cause_suggestion': "'Not specified'"
            },
            'condition': "root_cause_suggestion IS NULL"
        }
    ]
    
    with engine.connect() as conn:
        for fix in fixes:
            # Count rows needing fix
            count_query = f"SELECT COUNT(*) as count FROM {fix['table']} WHERE {fix['condition']}"
            result = conn.execute(text(count_query))
            count = result.fetchone()[0]
            
            if count > 0:
                logger.info(f"   {fix['table']}: {count} rows need fixing")
                
                if verbose:
                    for col, val in fix['updates'].items():
                        logger.info(f"      - {col} → {val}")
                
                if not dry_run:
                    # Build SET clause
                    set_clause = ', '.join([f"{col} = COALESCE({col}, {val})" 
                                          for col, val in fix['updates'].items()])
                    
                    update_query = f"UPDATE {fix['table']} SET {set_clause} WHERE {fix['condition']}"
                    conn.execute(text(update_query))
                    total_updates += count
                    logger.info(f"   ✓ Updated {count} rows in {fix['table']}")
                else:
                    logger.info(f"   [DRY RUN] Would update {count} rows")
            else:
                logger.info(f"   {fix['table']}: No NULL values found ✓")
        
        if not dry_run:
            conn.commit()
    
    return total_updates


def standardize_data_types(dry_run: bool = False, verbose: bool = False) -> int:
    """
    Standardize data types for Power BI compatibility
    - Dates → TEXT (YYYY-MM-DD format)
    - Durations → INTEGER
    - Booleans → INTEGER (0/1)
    
    Args:
        dry_run: If True, only report what would be changed
        verbose: Show detailed information
        
    Returns:
        int: Total rows updated
    """
    logger.info("📊 Standardizing data types...")
    
    engine = get_engine()
    total_updates = 0
    
    with engine.connect() as conn:
        # 1. Fix date formats (ensure YYYY-MM-DD)
        date_fixes = [
            ("tasks", "created_date"),
            ("tasks", "start_date"),
            ("tasks", "end_date")
        ]
        
        for table, col in date_fixes:
            # Check for non-standard date formats
            check_query = f"""
                SELECT COUNT(*) as count FROM {table} 
                WHERE {col} IS NOT NULL 
                  AND {col} != '' 
                  AND LENGTH({col}) != 10
            """
            result = conn.execute(text(check_query))
            count = result.fetchone()[0]
            
            if count > 0:
                logger.info(f"   {table}.{col}: {count} dates need standardization")
                
                if not dry_run:
                    # Attempt to standardize (this is simplified - may need more complex logic)
                    update_query = f"""
                        UPDATE {table} 
                        SET {col} = SUBSTR({col}, 1, 10)
                        WHERE {col} IS NOT NULL AND LENGTH({col}) > 10
                    """
                    conn.execute(text(update_query))
                    total_updates += count
                    logger.info(f"   ✓ Standardized {count} dates")
                else:
                    logger.info(f"   [DRY RUN] Would standardize {count} dates")
        
        # 2. Ensure durations are integers
        duration_query = """
            SELECT COUNT(*) as count FROM tasks 
            WHERE actual_duration IS NOT NULL 
              AND TYPEOF(actual_duration) != 'integer'
        """
        result = conn.execute(text(duration_query))
        count = result.fetchone()[0]
        
        if count > 0:
            logger.info(f"   tasks.actual_duration: {count} non-integer values")
            
            if not dry_run:
                conn.execute(text("""
                    UPDATE tasks 
                    SET actual_duration = CAST(actual_duration AS INTEGER)
                    WHERE TYPEOF(actual_duration) != 'integer'
                """))
                total_updates += count
                logger.info(f"   ✓ Converted {count} durations to INTEGER")
            else:
                logger.info(f"   [DRY RUN] Would convert {count} durations")
        
        if not dry_run:
            conn.commit()
    
    return total_updates


def normalize_casing(dry_run: bool = False, verbose: bool = False) -> int:
    """
    Normalize text casing for consistency
    - Priority: High/Medium/Low (title case)
    - Status: In Progress/Completed/etc (title case)
    
    Args:
        dry_run: If True, only report what would be changed
        verbose: Show detailed information
        
    Returns:
        int: Total rows updated
    """
    logger.info("🔤 Normalizing text casing...")
    
    engine = get_engine()
    total_updates = 0
    
    normalizations = [
        {
            'table': 'tasks',
            'column': 'priority',
            'values': {
                'high': 'High',
                'HIGH': 'High',
                'medium': 'Medium',
                'MEDIUM': 'Medium',
                'low': 'Low',
                'LOW': 'Low'
            }
        },
        {
            'table': 'tasks',
            'column': 'status',
            'values': {
                'in progress': 'In Progress',
                'IN PROGRESS': 'In Progress',
                'completed': 'Completed',
                'COMPLETED': 'Completed',
                'pending': 'Pending',
                'PENDING': 'Pending',
                'cancelled': 'Cancelled',
                'CANCELLED': 'Cancelled'
            }
        }
    ]
    
    with engine.connect() as conn:
        for norm in normalizations:
            for old_val, new_val in norm['values'].items():
                check_query = f"""
                    SELECT COUNT(*) as count FROM {norm['table']} 
                    WHERE {norm['column']} = :old_val
                """
                result = conn.execute(text(check_query), {'old_val': old_val})
                count = result.fetchone()[0]
                
                if count > 0:
                    if verbose:
                        logger.info(f"   {norm['table']}.{norm['column']}: '{old_val}' → '{new_val}' ({count} rows)")
                    
                    if not dry_run:
                        update_query = f"""
                            UPDATE {norm['table']} 
                            SET {norm['column']} = :new_val
                            WHERE {norm['column']} = :old_val
                        """
                        conn.execute(text(update_query), {'old_val': old_val, 'new_val': new_val})
                        total_updates += count
        
        if not dry_run and total_updates > 0:
            conn.commit()
            logger.info(f"   ✓ Normalized {total_updates} values")
        elif dry_run and total_updates > 0:
            logger.info(f"   [DRY RUN] Would normalize {total_updates} values")
        else:
            logger.info(f"   No casing issues found ✓")
    
    return total_updates


def remove_orphan_records(dry_run: bool = False, verbose: bool = False) -> int:
    """
    Remove orphan records (referencing non-existent tasks)
    
    Args:
        dry_run: If True, only report what would be deleted
        verbose: Show detailed information
        
    Returns:
        int: Total rows deleted
    """
    logger.info("🗑️  Checking for orphan records...")
    
    engine = get_engine()
    total_deleted = 0
    
    # Tables with FK to tasks
    dependent_tables = [
        'gpt_suggestions',
        'bottleneck_history',
        'task_reassignments',
        'ml_predictions',
        'improvement_log'
    ]
    
    with engine.connect() as conn:
        for table in dependent_tables:
            try:
                # Check for orphans
                check_query = f"""
                    SELECT COUNT(*) as count FROM {table} 
                    WHERE task_id NOT IN (SELECT task_id FROM tasks)
                """
                result = conn.execute(text(check_query))
                count = result.fetchone()[0]
                
                if count > 0:
                    logger.warning(f"   {table}: {count} orphan records found")
                    
                    if not dry_run:
                        delete_query = f"""
                            DELETE FROM {table} 
                            WHERE task_id NOT IN (SELECT task_id FROM tasks)
                        """
                        conn.execute(text(delete_query))
                        total_deleted += count
                        logger.info(f"   ✓ Deleted {count} orphan records")
                    else:
                        logger.info(f"   [DRY RUN] Would delete {count} orphans")
                else:
                    logger.info(f"   {table}: No orphans ✓")
            
            except Exception as e:
                if verbose:
                    logger.warning(f"   {table}: Could not check (table may not exist)")
        
        if not dry_run:
            conn.commit()
    
    return total_deleted


def vacuum_database():
    """Optimize database by reclaiming space"""
    logger.info("🧹 Optimizing database...")
    
    engine = get_engine()
    
    with engine.connect() as conn:
        # Note: VACUUM cannot run inside a transaction in SQLite
        conn.execute(text("VACUUM"))
        logger.info("   ✓ Database optimized")


def verify_tables(verbose: bool = False):
    """Verify all tables and show statistics"""
    logger.info("\n📋 Database verification:")
    
    engine = get_engine()
    
    with engine.connect() as conn:
        # Get all tables
        result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"))
        tables = result.fetchall()
        
        for (table_name,) in tables:
            try:
                count_result = conn.execute(text(f"SELECT COUNT(*) as count FROM {table_name}"))
                count = count_result.fetchone()[0]
                
                if verbose and count > 0:
                    # Show null counts for key columns
                    null_result = conn.execute(text(f"PRAGMA table_info({table_name})"))
                    columns = null_result.fetchall()
                    
                    logger.info(f"   {table_name}: {count} rows")
                    
                    # Check for nulls in each column
                    for col in columns[:5]:  # Limit to first 5 columns
                        col_name = col[1]
                        null_check = conn.execute(text(f"""
                            SELECT COUNT(*) FROM {table_name} WHERE {col_name} IS NULL
                        """))
                        null_count = null_check.fetchone()[0]
                        if null_count > 0:
                            logger.info(f"      - {col_name}: {null_count} NULLs")
                else:
                    logger.info(f"   {table_name}: {count} rows")
            
            except Exception as e:
                logger.error(f"   {table_name}: Error - {e}")


def fix_all(dry_run: bool = False, verbose: bool = False):
    """Run all fixes"""
    logger.info("\n" + "="*60)
    logger.info("🔧 DATABASE CLEANUP FOR POWER BI")
    if dry_run:
        logger.info("   [DRY RUN MODE - No changes will be made]")
    logger.info("="*60 + "\n")
    
    total_changes = 0
    
    # Run all fixes
    total_changes += fix_null_values(dry_run, verbose)
    total_changes += standardize_data_types(dry_run, verbose)
    total_changes += normalize_casing(dry_run, verbose)
    total_changes += remove_orphan_records(dry_run, verbose)
    
    if not dry_run:
        vacuum_database()
    
    verify_tables(verbose)
    
    logger.info("\n" + "="*60)
    if dry_run:
        logger.info(f"✅ ANALYSIS COMPLETE - {total_changes} changes needed")
    else:
        logger.info(f"✅ DATABASE FIXED - {total_changes} total changes")
    logger.info("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fix database for Power BI compatibility')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be changed without making changes')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed information')
    parser.add_argument('--nulls-only', action='store_true',
                       help='Only fix NULL values')
    parser.add_argument('--types-only', action='store_true',
                       help='Only standardize data types')
    parser.add_argument('--casing-only', action='store_true',
                       help='Only normalize casing')
    parser.add_argument('--orphans-only', action='store_true',
                       help='Only remove orphan records')
    
    args = parser.parse_args()
    
    # Run specific fix or all fixes
    if args.nulls_only:
        fix_null_values(args.dry_run, args.verbose)
    elif args.types_only:
        standardize_data_types(args.dry_run, args.verbose)
    elif args.casing_only:
        normalize_casing(args.dry_run, args.verbose)
    elif args.orphans_only:
        remove_orphan_records(args.dry_run, args.verbose)
    else:
        fix_all(args.dry_run, args.verbose)
