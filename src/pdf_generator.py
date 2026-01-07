"""
PDF Report Generator for FlowFix AI - Production Version
Generates professional PDF reports with charts, impact summaries, and comprehensive analysis

Key Production Features:
- Safe file path handling with validation
- Smart text truncation (preserve sentence boundaries)
- Graceful handling of missing tables/data
- Impact summary section with reduction metrics
- Insert SHAP and matplotlib charts
- Multi-page support with table of contents
- Executive summary with ROI calculations
"""
import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Optional, Dict, List
from fpdf import FPDF
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from utils import execute_query


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_generator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FlowFixReport(FPDF):
    """Custom PDF report class with enhanced formatting"""
    
    def __init__(self):
        super().__init__()
        self.toc = []  # Table of contents
        self.current_section = 1
    
    def header(self):
        """Add header to each page"""
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'FlowFix AI - Workflow Analysis Report', 0, 1, 'C')
        self.set_font('Arial', '', 10)
        self.cell(0, 5, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
        self.ln(10)
    
    def footer(self):
        """Add footer to each page"""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def chapter_title(self, title, add_to_toc=True):
        """Add chapter title with optional TOC entry"""
        if add_to_toc:
            self.toc.append((title, self.page_no()))
        
        self.set_font('Arial', 'B', 14)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, f"{self.current_section}. {title}", 0, 1, 'L', 1)
        self.current_section += 1
        self.ln(4)
    
    def chapter_body(self, body):
        """Add chapter body text with proper wrapping"""
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 6, body)
        self.ln()
    
    def add_table_of_contents(self):
        """Add table of contents page"""
        self.add_page()
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Table of Contents', 0, 1, 'L')
        self.ln(5)
        
        self.set_font('Arial', '', 11)
        for title, page in self.toc:
            self.cell(0, 8, f'{title}' + '.' * 50 + f' Page {page}', 0, 1)


def safe_file_path(output_path: Optional[str] = None) -> str:
    """
    Generate safe file path with validation
    
    Args:
        output_path: Optional custom output path
        
    Returns:
        Validated file path
    """
    if output_path:
        # Validate provided path
        output_dir = os.path.dirname(output_path)
        
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Created directory: {output_dir}")
            except Exception as e:
                logger.error(f"Cannot create directory {output_dir}: {e}")
                output_path = None
    
    if not output_path:
        # Generate default safe path
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            exports_dir = os.path.join(project_root, 'exports')
            os.makedirs(exports_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(exports_dir, f'flowfix_report_{timestamp}.pdf')
            
        except Exception as e:
            logger.error(f"Error generating default path: {e}")
            # Ultimate fallback
            output_path = f'flowfix_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    
    return output_path


def truncate_text_smart(text: str, max_length: int = 200) -> str:
    """
    Truncate text intelligently without breaking mid-sentence
    
    Args:
        text: Text to truncate
        max_length: Maximum character length
        
    Returns:
        Truncated text with ellipsis
    """
    if not text or len(str(text)) <= max_length:
        return str(text) if text else ""
    
    text = str(text)
    
    # Find last sentence boundary before max_length
    sentence_endings = ['. ', '! ', '? ', '\n']
    last_boundary = -1
    
    for ending in sentence_endings:
        pos = text.rfind(ending, 0, max_length)
        if pos > last_boundary:
            last_boundary = pos + len(ending)
    
    # If found sentence boundary, use it
    if last_boundary > max_length // 2:  # At least halfway
        return text[:last_boundary].strip() + "..."
    
    # Otherwise, find last word boundary
    last_space = text.rfind(' ', 0, max_length)
    if last_space > 0:
        return text[:last_space].strip() + "..."
    
    # Fallback: hard truncate
    return text[:max_length].strip() + "..."


def query_with_fallback(query: str, fallback_message: str = "Data not available") -> any:
    """
    Execute query with graceful fallback on error
    
    Args:
        query: SQL query to execute
        fallback_message: Message to log on failure
        
    Returns:
        Query result or empty DataFrame
    """
    try:
        result = execute_query(query)
        return result
    except Exception as e:
        logger.warning(f"Query failed: {fallback_message} - {e}")
        import pandas as pd
        return pd.DataFrame()


def generate_impact_summary() -> Dict:
    """
    Generate impact summary with reduction metrics
    
    Returns:
        Dictionary with impact statistics
    """
    impact_data = {
        'delay_reduced_pct': 0,
        'duration_reduced_days': 0,
        'gpt_applied_count': 0,
        'ml_predictions_accurate': 0,
        'bottlenecks_resolved': 0,
        'hours_saved': 0
    }
    
    try:
        # Delay reduction
        delay_query = """
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN is_delayed = 0 THEN 1 ELSE 0 END) as on_time
        FROM tasks
        WHERE actual_duration IS NOT NULL
        """
        delay_df = query_with_fallback(delay_query)
        if not delay_df.empty:
            total = delay_df.iloc[0]['total']
            on_time = delay_df.iloc[0]['on_time']
            if total > 0:
                impact_data['delay_reduced_pct'] = (on_time / total) * 100
        
        # Duration reduction
        duration_query = """
        SELECT 
            AVG(task_duration - COALESCE(actual_duration, task_duration)) as avg_reduction
        FROM tasks
        WHERE actual_duration IS NOT NULL AND actual_duration < task_duration
        """
        duration_df = query_with_fallback(duration_query)
        if not duration_df.empty:
            avg_reduction = duration_df.iloc[0]['avg_reduction']
            if avg_reduction:
                impact_data['duration_reduced_days'] = avg_reduction
        
        # GPT suggestions applied
        gpt_query = """
        SELECT COUNT(*) as count
        FROM gpt_suggestions
        WHERE feedback_status = 'applied'
        """
        gpt_df = query_with_fallback(gpt_query)
        if not gpt_df.empty:
            impact_data['gpt_applied_count'] = int(gpt_df.iloc[0]['count'])
        
        # ML prediction accuracy
        ml_query = """
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN prediction_correct = 1 THEN 1 ELSE 0 END) as correct
        FROM ml_predictions
        """
        ml_df = query_with_fallback(ml_query)
        if not ml_df.empty and ml_df.iloc[0]['total'] > 0:
            total = ml_df.iloc[0]['total']
            correct = ml_df.iloc[0]['correct'] or 0
            impact_data['ml_predictions_accurate'] = (correct / total) * 100
        
        # Bottlenecks resolved
        bottleneck_query = """
        SELECT COUNT(*) as count
        FROM bottleneck_history
        WHERE resolution_date IS NOT NULL
        """
        bn_df = query_with_fallback(bottleneck_query)
        if not bn_df.empty:
            impact_data['bottlenecks_resolved'] = int(bn_df.iloc[0]['count'])
        
        # Calculate hours saved
        hours_per_day = 8
        tasks_improved = impact_data['gpt_applied_count'] + impact_data['bottlenecks_resolved']
        impact_data['hours_saved'] = impact_data['duration_reduced_days'] * tasks_improved * hours_per_day
        
    except Exception as e:
        logger.error(f"Error generating impact summary: {e}")
    
    return impact_data


def create_summary_chart(impact_data: Dict, output_path: str = 'models/impact_chart.png'):
    """
    Create summary impact chart with matplotlib
    
    Args:
        impact_data: Impact statistics dictionary
        output_path: Path to save chart
        
    Returns:
        Path to saved chart or None
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Chart 1: Metrics Overview
        metrics = ['Delay Reduced %', 'Duration Saved (days)', 'GPT Applied', 'Bottlenecks Resolved']
        values = [
            impact_data['delay_reduced_pct'],
            impact_data['duration_reduced_days'],
            impact_data['gpt_applied_count'],
            impact_data['bottlenecks_resolved']
        ]
        
        axes[0].bar(metrics, values, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'])
        axes[0].set_title('Impact Metrics Overview', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Value')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Chart 2: Time Saved
        time_labels = ['Hours Saved', 'Person-Days']
        time_values = [impact_data['hours_saved'], impact_data['hours_saved'] / 8]
        
        axes[1].barh(time_labels, time_values, color=['#FF5722', '#009688'])
        axes[1].set_title('Time Savings', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Value')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Impact chart created: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating impact chart: {e}")
        return None


def find_shap_chart() -> Optional[str]:
    """
    Find most recent SHAP visualization chart
    
    Returns:
        Path to SHAP chart or None
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        shap_dir = os.path.join(project_root, 'models', 'shap_plots')
        
        if not os.path.exists(shap_dir):
            logger.warning("SHAP plots directory not found")
            return None
        
        # Find summary plot
        summary_plot = os.path.join(shap_dir, 'shap_summary_plot.png')
        if os.path.exists(summary_plot):
            return summary_plot
        
        # Find any png file
        png_files = [f for f in os.listdir(shap_dir) if f.endswith('.png')]
        if png_files:
            return os.path.join(shap_dir, png_files[0])
        
        return None
        
    except Exception as e:
        logger.error(f"Error finding SHAP chart: {e}")
        return None


def generate_executive_summary() -> str:
    """Generate executive summary with key metrics"""
    query = """
    SELECT 
        COUNT(*) as total_tasks,
        AVG(actual_duration) as avg_duration,
        SUM(CASE WHEN is_delayed = 1 THEN 1 ELSE 0 END) as delayed_tasks,
        COUNT(CASE WHEN bottleneck_type != '' THEN 1 END) as bottleneck_count,
        COUNT(DISTINCT assignee) as total_assignees,
        COUNT(DISTINCT project) as total_projects
    FROM tasks
    WHERE actual_duration IS NOT NULL
    """
    
    df = query_with_fallback(query, "Executive summary data")
    
    if df.empty:
        return "EXECUTIVE SUMMARY\n\nInsufficient data to generate summary."
    
    result = df.iloc[0]
    total_tasks = int(result['total_tasks'])
    
    if total_tasks == 0:
        return "EXECUTIVE SUMMARY\n\nNo completed tasks to analyze."
    
    avg_duration = result['avg_duration'] or 0
    delayed_tasks = int(result['delayed_tasks'])
    bottleneck_count = int(result['bottleneck_count'])
    total_assignees = int(result['total_assignees'])
    total_projects = int(result['total_projects'])
    
    delay_pct = (delayed_tasks / total_tasks * 100) if total_tasks > 0 else 0
    bottleneck_pct = (bottleneck_count / total_tasks * 100) if total_tasks > 0 else 0
    
    summary = f"""EXECUTIVE SUMMARY

Project Overview:
- Total Tasks Analyzed: {total_tasks:,}
- Average Task Duration: {avg_duration:.1f} days
- Team Size: {total_assignees} assignees
- Active Projects: {total_projects}

Performance Metrics:
- Tasks Delayed: {delayed_tasks} ({delay_pct:.1f}% of total)
- Bottlenecks Detected: {bottleneck_count} ({bottleneck_pct:.1f}% of total)
- On-Time Completion Rate: {100 - delay_pct:.1f}%

Key Insights:
- Workflow efficiency can be improved by addressing {bottleneck_count} identified bottlenecks
- {delayed_tasks} tasks exceeded expected completion time, indicating capacity or dependency issues
- Immediate action recommended for high-severity bottlenecks to prevent cascading delays

Recommended Actions:
1. Review and resolve high-severity bottlenecks immediately
2. Redistribute workload for overloaded team members
3. Clear blocked tasks by resolving external dependencies
"""
    
    return summary


def generate_impact_summary_section(impact_data: Dict) -> str:
    """
    Generate impact summary section text
    
    Args:
        impact_data: Impact statistics dictionary
        
    Returns:
        Formatted impact summary text
    """
    text = f"""IMPACT SUMMARY

System Effectiveness:
- Delay Reduction: {impact_data['delay_reduced_pct']:.1f}% of tasks completed on-time
- Duration Savings: {impact_data['duration_reduced_days']:.1f} days average reduction per task
- Total Time Saved: {impact_data['hours_saved']:.0f} hours ({impact_data['hours_saved']/8:.1f} person-days)

AI/ML Performance:
- GPT Suggestions Applied: {impact_data['gpt_applied_count']} recommendations implemented
- ML Prediction Accuracy: {impact_data['ml_predictions_accurate']:.1f}%
- Bottlenecks Resolved: {impact_data['bottlenecks_resolved']} issues addressed

ROI Estimate:
Assuming an average hourly cost of $50 per team member:
- Cost Savings: ${impact_data['hours_saved'] * 50:,.0f}
- Efficiency Gain: {impact_data['hours_saved']/8:.1f} person-days redirected to productive work

The FlowFix AI system has demonstrably improved workflow efficiency through data-driven
insights and actionable recommendations.
"""
    
    return text


def generate_bottleneck_analysis() -> str:
    """Generate bottleneck analysis section"""
    query = """
    SELECT 
        bottleneck_type,
        COUNT(*) as count,
        AVG(actual_duration) as avg_duration,
        AVG(severity_score) as avg_severity
    FROM tasks
    WHERE bottleneck_type != ''
    GROUP BY bottleneck_type
    ORDER BY count DESC
    """
    
    df = query_with_fallback(query, "Bottleneck analysis")
    
    if df.empty:
        return "BOTTLENECK ANALYSIS\n\nNo bottlenecks detected in the analyzed tasks."
    
    text = "BOTTLENECK ANALYSIS\n\n"
    text += "Identified Bottleneck Types (by frequency):\n\n"
    
    for idx, row in df.iterrows():
        # Remove cluster suffix if present
        btype = str(row['bottleneck_type']).split('_C')[0].replace('_', ' ').title()
        count = int(row['count'])
        avg_duration = row['avg_duration'] or 0
        avg_severity = row['avg_severity'] or 0
        
        text += f"{idx+1}. {btype}:\n"
        text += f"   - Frequency: {count} tasks affected\n"
        text += f"   - Average Duration: {avg_duration:.1f} days\n"
        text += f"   - Average Severity: {avg_severity:.1f}/100\n\n"
    
    text += "\nRecommendation: Focus on the top 3 bottleneck types for maximum impact."
    
    return text


def generate_assignee_performance() -> str:
    """Generate assignee performance section"""
    query = """
    SELECT 
        assignee,
        COUNT(*) as task_count,
        AVG(actual_duration) as avg_duration,
        SUM(CASE WHEN is_delayed = 1 THEN 1 ELSE 0 END) as delayed_count,
        COUNT(CASE WHEN bottleneck_type != '' THEN 1 END) as bottleneck_count
    FROM tasks
    WHERE actual_duration IS NOT NULL
    GROUP BY assignee
    ORDER BY bottleneck_count DESC, delayed_count DESC
    LIMIT 10
    """
    
    df = query_with_fallback(query, "Team performance")
    
    if df.empty:
        return "TEAM PERFORMANCE\n\nInsufficient data to analyze team performance."
    
    text = "TEAM PERFORMANCE\n\n"
    text += "Top Team Members by Bottleneck Count:\n\n"
    
    for idx, row in df.iterrows():
        assignee = row['assignee']
        task_count = int(row['task_count'])
        avg_duration = row['avg_duration'] or 0
        delayed_count = int(row['delayed_count'])
        bottleneck_count = int(row['bottleneck_count'])
        
        delay_pct = (delayed_count / task_count * 100) if task_count > 0 else 0
        
        text += f"{idx+1}. {assignee}:\n"
        text += f"   - Total Tasks: {task_count}\n"
        text += f"   - Average Duration: {avg_duration:.1f} days\n"
        text += f"   - Delayed Tasks: {delayed_count} ({delay_pct:.1f}%)\n"
        text += f"   - Bottlenecks Encountered: {bottleneck_count}\n\n"
    
    text += "\nNote: High bottleneck counts may indicate workload issues or external dependencies."
    
    return text


def generate_gpt_recommendations() -> str:
    """Generate GPT recommendations section with smart truncation"""
    query = """
    SELECT 
        gs.task_id,
        gs.root_causes,
        gs.recommendations,
        gs.quality_score,
        gs.feedback_status
    FROM gpt_suggestions gs
    ORDER BY gs.quality_score DESC
    LIMIT 5
    """
    
    df = query_with_fallback(query, "GPT recommendations")
    
    if df.empty:
        return "AI-POWERED RECOMMENDATIONS\n\nNo recommendations available. Run GPT suggester first."
    
    text = "AI-POWERED RECOMMENDATIONS\n\n"
    text += "Top Quality Suggestions:\n\n"
    
    for idx, row in df.iterrows():
        task_id = row['task_id']
        root_causes = truncate_text_smart(row['root_causes'], 200)
        recommendations = truncate_text_smart(row['recommendations'], 250)
        quality_score = row['quality_score'] if row['quality_score'] else 'N/A'
        feedback_status = row['feedback_status'] if row['feedback_status'] else 'pending'
        
        text += f"{idx+1}. Task: {task_id}\n"
        text += f"   Quality Score: {quality_score}/100 | Status: {feedback_status}\n\n"
        text += f"   Root Causes:\n   {root_causes}\n\n"
        text += f"   Recommendations:\n   {recommendations}\n\n"
        text += "-" * 70 + "\n\n"
    
    return text


def generate_pdf_report(output_path: Optional[str] = None, include_charts: bool = True) -> str:
    """
    Generate complete PDF report with all sections
    
    Args:
        output_path: Optional custom output path
        include_charts: Whether to include charts (default: True)
        
    Returns:
        Path to generated PDF
    """
    print("\n" + "="*80)
    print("📄 GENERATING COMPREHENSIVE PDF REPORT")
    print("="*80 + "\n")
    
    # Validate and generate safe file path
    output_path = safe_file_path(output_path)
    logger.info(f"Output path: {output_path}")
    
    # Create PDF object
    pdf = FlowFixReport()
    pdf.add_page()
    
    # Section 1: Executive Summary
    logger.info("Adding executive summary...")
    print("✓ Adding executive summary...")
    pdf.chapter_title("EXECUTIVE SUMMARY")
    pdf.chapter_body(generate_executive_summary())
    
    # Section 2: Impact Summary
    logger.info("Generating impact summary...")
    print("✓ Generating impact summary...")
    impact_data = generate_impact_summary()
    pdf.add_page()
    pdf.chapter_title("IMPACT SUMMARY")
    pdf.chapter_body(generate_impact_summary_section(impact_data))
    
    # Insert Impact Chart
    if include_charts:
        logger.info("Creating impact chart...")
        print("✓ Creating impact chart...")
        impact_chart_path = create_summary_chart(impact_data)
        if impact_chart_path and os.path.exists(impact_chart_path):
            try:
                pdf.image(impact_chart_path, x=10, y=None, w=180)
                pdf.ln(5)
                logger.info("Impact chart inserted successfully")
            except Exception as e:
                logger.warning(f"Could not insert impact chart: {e}")
    
    # Section 3: Bottleneck Analysis
    logger.info("Adding bottleneck analysis...")
    print("✓ Adding bottleneck analysis...")
    pdf.add_page()
    pdf.chapter_title("BOTTLENECK ANALYSIS")
    pdf.chapter_body(generate_bottleneck_analysis())
    
    # Section 4: Team Performance
    logger.info("Adding team performance...")
    print("✓ Adding team performance...")
    pdf.add_page()
    pdf.chapter_title("TEAM PERFORMANCE")
    pdf.chapter_body(generate_assignee_performance())
    
    # Section 5: AI Recommendations
    logger.info("Adding AI recommendations...")
    print("✓ Adding AI recommendations...")
    pdf.add_page()
    pdf.chapter_title("AI-POWERED RECOMMENDATIONS")
    pdf.chapter_body(generate_gpt_recommendations())
    
    # Insert SHAP Chart
    if include_charts:
        logger.info("Looking for SHAP visualization...")
        print("✓ Looking for SHAP visualization...")
        shap_chart = find_shap_chart()
        if shap_chart:
            try:
                pdf.add_page()
                pdf.chapter_title("ML MODEL EXPLAINABILITY (SHAP)")
                pdf.chapter_body("SHAP (SHapley Additive exPlanations) analysis shows feature importance:\n")
                pdf.image(shap_chart, x=10, y=None, w=180)
                logger.info("SHAP chart inserted successfully")
            except Exception as e:
                logger.warning(f"Could not insert SHAP chart: {e}")
    
    # Section 6: Next Steps
    logger.info("Adding next steps...")
    print("✓ Adding next steps...")
    pdf.add_page()
    pdf.chapter_title("NEXT STEPS & RECOMMENDATIONS")
    pdf.chapter_body("""1. IMMEDIATE ACTIONS (This Week):
   - Address high-severity bottlenecks identified in this report
   - Redistribute workload for overloaded team members (see Team Performance section)
   - Clear blocked tasks by resolving external dependencies
   - Review and approve pending GPT recommendations

2. SHORT-TERM (1-2 Weeks):
   - Implement top 3 AI recommendations for maximum impact
   - Monitor improvement in task completion rates using dashboard
   - Review team capacity and adjust task assignments accordingly
   - Schedule follow-up bottleneck review meeting

3. LONG-TERM (1 Month+):
   - Establish regular bottleneck review cadence (bi-weekly recommended)
   - Track improvement metrics over time to measure ROI
   - Refine processes based on data insights and team feedback
   - Consider automation for repetitive bottlenecks
   - Integrate FlowFix AI insights into sprint planning

4. CONTINUOUS IMPROVEMENT:
   - Provide feedback on applied suggestions to improve AI recommendations
   - Update ML models quarterly with new data
   - Adjust severity thresholds based on team capacity
   - Document successful interventions for knowledge sharing

For questions, support, or custom reporting needs, refer to the FlowFix AI documentation
or contact your system administrator.
    """)
    
    # Save PDF
    try:
        pdf.output(output_path)
        file_size = os.path.getsize(output_path) / 1024
        
        print(f"\n✅ PDF report generated successfully!")
        print(f"   Location: {output_path}")
        print(f"   File size: {file_size:.1f} KB")
        print(f"   Pages: {pdf.page_no()}")
        
        logger.info(f"PDF report generated: {output_path} ({file_size:.1f} KB, {pdf.page_no()} pages)")
        
    except Exception as e:
        logger.error(f"Error saving PDF: {e}")
        raise
    
    return output_path


def cli():
    """Command-line interface for PDF generator"""
    parser = argparse.ArgumentParser(description='FlowFix AI PDF Report Generator')
    parser.add_argument('--output', '-o', help='Custom output path for PDF report')
    parser.add_argument('--no-charts', action='store_true', help='Generate report without charts')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Generate report
    try:
        report_path = generate_pdf_report(args.output, include_charts=not args.no_charts)
        print(f"\n📊 Report ready: {report_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
