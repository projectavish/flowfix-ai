"""
Export database tables to Excel for Power BI - Production Grade
Enhanced with correct column names, feedback fields, and error handling
"""
import os
import logging
from datetime import datetime
from typing import Optional
import pandas as pd
from utils import execute_query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_export_filename(base_name: str = 'powerbi_data', with_timestamp: bool = True) -> str:
    """
    Generate export filename with optional timestamp
    
    Args:
        base_name: Base name for the file
        with_timestamp: Whether to include timestamp in filename
        
    Returns:
        str: Full file path
    """
    export_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'exports')
    os.makedirs(export_dir, exist_ok=True)
    
    if with_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{base_name}_{timestamp}.xlsx"
    else:
        filename = f"{base_name}.xlsx"
    
    return os.path.join(export_dir, filename)


def export_tasks_sheet() -> Optional[pd.DataFrame]:
    """Export tasks with all relevant fields"""
    logger.info("📋 Exporting Tasks sheet...")
    
    query = """
    SELECT 
        task_id,
        task_name,
        assignee,
        status,
        priority,
        project,
        created_date,
        start_date,
        end_date,
        actual_duration,
        is_delayed,
        bottleneck_type,
        comments,
        reassignment_count
    FROM tasks
    """
    
    try:
        df = execute_query(query)
        logger.info(f"   ✓ Exported {len(df)} tasks")
        return df
    except Exception as e:
        logger.error(f"   ✗ Error exporting tasks: {e}")
        return None


def export_bottlenecks_sheet() -> Optional[pd.DataFrame]:
    """Export bottleneck analysis"""
    logger.info("🚨 Exporting Bottlenecks sheet...")
    
    query = """
    SELECT 
        bh.task_id,
        t.task_name,
        t.assignee,
        t.project,
        t.priority,
        bh.bottleneck_type,
        bh.severity_score,
        bh.root_cause_suggestion,
        bh.detected_at,
        t.actual_duration,
        t.is_delayed
    FROM bottleneck_history bh
    JOIN tasks t ON bh.task_id = t.task_id
    ORDER BY bh.detected_at DESC
    """
    
    try:
        df = execute_query(query)
        logger.info(f"   ✓ Exported {len(df)} bottleneck records")
        return df
    except Exception as e:
        logger.error(f"   ✗ Error exporting bottlenecks: {e}")
        return None


def export_gpt_suggestions_sheet() -> Optional[pd.DataFrame]:
    """Export GPT suggestions with feedback fields"""
    logger.info("🤖 Exporting GPT Suggestions sheet...")
    
    # Fix: Use correct column names (root_causes, recommendations plural)
    query = """
    SELECT 
        g.task_id,
        t.task_name,
        t.assignee,
        t.project,
        t.priority,
        g.suggestion_text,
        g.root_causes,
        g.recommendations,
        g.prompt_version,
        g.gpt_model_used,
        g.sentiment,
        g.urgency_level,
        g.quality_score,
        g.applied,
        g.applied_date,
        g.applied_action,
        g.created_at,
        fb.feedback_status,
        fb.feedback_comment,
        fb.feedback_date,
        fb.was_helpful
    FROM gpt_suggestions g
    JOIN tasks t ON g.task_id = t.task_id
    LEFT JOIN feedback_log fb ON g.task_id = fb.task_id AND fb.feedback_type = 'gpt_suggestion'
    ORDER BY g.created_at DESC
    """
    
    try:
        df = execute_query(query)
        logger.info(f"   ✓ Exported {len(df)} GPT suggestions")
        return df
    except Exception as e:
        logger.error(f"   ✗ Error exporting GPT suggestions: {e}")
        # Try fallback query without feedback fields
        try:
            fallback_query = """
            SELECT 
                g.task_id,
                t.task_name,
                t.assignee,
                t.project,
                g.suggestion_text,
                g.root_causes,
                g.recommendations,
                g.applied,
                g.created_at
            FROM gpt_suggestions g
            JOIN tasks t ON g.task_id = t.task_id
            """
            df = execute_query(fallback_query)
            logger.warning(f"   ⚠ Used fallback query, exported {len(df)} suggestions")
            return df
        except Exception as e2:
            logger.error(f"   ✗ Fallback also failed: {e2}")
            return None


def export_improvements_sheet() -> Optional[pd.DataFrame]:
    """Export improvement tracking data"""
    logger.info("📈 Exporting Improvements sheet...")
    
    query = """
    SELECT 
        il.task_id,
        t.task_name,
        t.assignee,
        t.project,
        il.action_taken,
        il.owner,
        il.date_applied,
        il.impact_measured,
        il.improvement_score,
        il.improvement_percentage,
        t.status as current_status,
        t.is_delayed as currently_delayed
    FROM improvement_log il
    JOIN tasks t ON il.task_id = t.task_id
    ORDER BY il.date_applied DESC
    """
    
    try:
        df = execute_query(query)
        logger.info(f"   ✓ Exported {len(df)} improvement records")
        return df
    except Exception as e:
        logger.error(f"   ✗ Error exporting improvements: {e}")
        return None


def export_reassignments_sheet() -> Optional[pd.DataFrame]:
    """Export task reassignment history"""
    logger.info("🔄 Exporting Reassignments sheet...")
    
    query = """
    SELECT 
        tr.task_id,
        t.task_name,
        t.project,
        tr.old_assignee,
        tr.new_assignee,
        tr.reason,
        tr.triggered_by,
        tr.reassigned_at,
        tr.was_delayed_before,
        t.is_delayed as is_delayed_after,
        t.status as current_status
    FROM task_reassignments tr
    JOIN tasks t ON tr.task_id = t.task_id
    ORDER BY tr.reassigned_at DESC
    """
    
    try:
        df = execute_query(query)
        logger.info(f"   ✓ Exported {len(df)} reassignment records")
        return df
    except Exception as e:
        logger.error(f"   ✗ Error exporting reassignments: {e}")
        return None


def export_ml_predictions_sheet() -> Optional[pd.DataFrame]:
    """Export ML predictions"""
    logger.info("🤖 Exporting ML Predictions sheet...")
    
    query = """
    SELECT 
        mp.task_id,
        t.task_name,
        t.assignee,
        t.project,
        mp.model_type,
        mp.prediction_value,
        mp.confidence_score,
        mp.model_version,
        mp.predicted_at,
        t.actual_duration,
        t.is_delayed
    FROM ml_predictions mp
    JOIN tasks t ON mp.task_id = t.task_id
    ORDER BY mp.predicted_at DESC
    """
    
    try:
        df = execute_query(query)
        logger.info(f"   ✓ Exported {len(df)} ML predictions")
        return df
    except Exception as e:
        logger.error(f"   ✗ Error exporting ML predictions: {e}")
        return None


def export_summary_metrics_sheet() -> Optional[pd.DataFrame]:
    """Export summary metrics"""
    logger.info("📊 Exporting Summary Metrics sheet...")
    
    query = """
    SELECT 
        COUNT(*) as total_tasks,
        AVG(actual_duration) as avg_duration_days,
        SUM(CASE WHEN is_delayed = 1 THEN 1 ELSE 0 END) as delayed_count,
        CAST(SUM(CASE WHEN is_delayed = 1 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100 as delay_percentage,
        COUNT(CASE WHEN bottleneck_type IS NOT NULL AND bottleneck_type != '' THEN 1 END) as bottleneck_count,
        COUNT(DISTINCT assignee) as total_assignees,
        COUNT(DISTINCT project) as total_projects,
        COUNT(CASE WHEN status = 'Completed' THEN 1 END) as completed_tasks,
        COUNT(CASE WHEN status = 'In Progress' THEN 1 END) as in_progress_tasks
    FROM tasks
    """
    
    try:
        df = execute_query(query)
        
        # Add timestamp
        df['export_timestamp'] = datetime.now().isoformat()
        
        logger.info(f"   ✓ Exported summary metrics")
        return df
    except Exception as e:
        logger.error(f"   ✗ Error exporting summary metrics: {e}")
        return None


def export_to_powerbi(filename: Optional[str] = None, include_timestamp: bool = True) -> bool:
    """
    Main export function - exports all tables to Excel for Power BI
    
    Args:
        filename: Custom filename (without .xlsx extension)
        include_timestamp: Whether to include timestamp in filename
        
    Returns:
        bool: Success status
    """
    logger.info("\n" + "="*60)
    logger.info("📊 POWER BI DATA EXPORT - Starting")
    logger.info("="*60 + "\n")
    
    # Generate filename
    if filename:
        excel_path = get_export_filename(filename, include_timestamp)
    else:
        excel_path = get_export_filename('powerbi_data', include_timestamp)
    
    sheets_exported = 0
    sheets_failed = 0
    
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            
            # Export each sheet with error handling
            
            # 1. Tasks
            df = export_tasks_sheet()
            if df is not None and len(df) > 0:
                df.to_excel(writer, sheet_name='Tasks', index=False)
                sheets_exported += 1
            else:
                sheets_failed += 1
            
            # 2. Bottlenecks
            df = export_bottlenecks_sheet()
            if df is not None and len(df) > 0:
                df.to_excel(writer, sheet_name='Bottlenecks', index=False)
                sheets_exported += 1
            else:
                logger.warning("   ⚠ Skipping Bottlenecks sheet (no data or error)")
            
            # 3. GPT Suggestions (with feedback fields)
            df = export_gpt_suggestions_sheet()
            if df is not None and len(df) > 0:
                df.to_excel(writer, sheet_name='GPT_Suggestions', index=False)
                sheets_exported += 1
            else:
                logger.warning("   ⚠ Skipping GPT Suggestions sheet (no data or error)")
            
            # 4. Improvements (NEW)
            df = export_improvements_sheet()
            if df is not None and len(df) > 0:
                df.to_excel(writer, sheet_name='Improvements', index=False)
                sheets_exported += 1
            else:
                logger.warning("   ⚠ Skipping Improvements sheet (no data or error)")
            
            # 5. Reassignments (NEW)
            df = export_reassignments_sheet()
            if df is not None and len(df) > 0:
                df.to_excel(writer, sheet_name='Reassignments', index=False)
                sheets_exported += 1
            else:
                logger.warning("   ⚠ Skipping Reassignments sheet (no data or error)")
            
            # 6. ML Predictions (NEW)
            df = export_ml_predictions_sheet()
            if df is not None and len(df) > 0:
                df.to_excel(writer, sheet_name='ML_Predictions', index=False)
                sheets_exported += 1
            else:
                logger.warning("   ⚠ Skipping ML Predictions sheet (no data or error)")
            
            # 7. Summary Metrics
            df = export_summary_metrics_sheet()
            if df is not None:
                df.to_excel(writer, sheet_name='Summary_Metrics', index=False)
                sheets_exported += 1
            else:
                sheets_failed += 1
        
        logger.info("\n" + "="*60)
        logger.info("✅ POWER BI DATA EXPORT - Complete")
        logger.info("="*60)
        logger.info(f"\n📁 Excel file created: {excel_path}")
        logger.info(f"📊 Sheets exported: {sheets_exported}")
        if sheets_failed > 0:
            logger.warning(f"⚠️  Sheets failed: {sheets_failed}")
        
        logger.info("\n📌 Power BI Instructions:")
        logger.info("   1. Open Power BI Desktop")
        logger.info("   2. Click 'Get Data' → 'Excel'")
        logger.info(f"   3. Select file: {excel_path}")
        logger.info("   4. Check all sheets → Click 'Load'")
        logger.info("   5. Create relationships between tables if needed")
        logger.info("   6. Start creating visualizations!\n")
        
        return True
    
    except Exception as e:
        logger.error(f"\n❌ Fatal error during export: {e}")
        return False


def export_quick(filename: str = 'powerbi_quick') -> bool:
    """
    Quick export without timestamp (overwrites previous file)
    Useful for scheduled/automated exports
    
    Args:
        filename: Base filename
        
    Returns:
        bool: Success status
    """
    return export_to_powerbi(filename, include_timestamp=False)


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'quick':
            # Quick export without timestamp
            success = export_quick()
        elif command == 'custom':
            # Custom filename
            filename = sys.argv[2] if len(sys.argv) > 2 else 'powerbi_custom'
            success = export_to_powerbi(filename)
        else:
            logger.error("Usage: python export_for_powerbi.py [quick|custom <name>]")
            success = False
    else:
        # Default: full export with timestamp
        success = export_to_powerbi()
    
    exit(0 if success else 1)
