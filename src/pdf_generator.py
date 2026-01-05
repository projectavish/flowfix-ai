"""
PDF Report Generator for FlowFix AI
Generates professional PDF reports with bottleneck analysis and recommendations
"""
import os
from datetime import datetime
from fpdf import FPDF
from utils import execute_query


class FlowFixReport(FPDF):
    """Custom PDF report class"""
    
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
    
    def chapter_title(self, title):
        """Add chapter title"""
        self.set_font('Arial', 'B', 14)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)
    
    def chapter_body(self, body):
        """Add chapter body text"""
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 6, body)
        self.ln()


def generate_executive_summary():
    """Generate executive summary data"""
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
    
    result = execute_query(query).iloc[0]
    
    summary = f"""
EXECUTIVE SUMMARY

Total Tasks Analyzed: {int(result['total_tasks']):,}
Average Task Duration: {result['avg_duration']:.1f} days
Tasks Delayed: {int(result['delayed_tasks'])} ({result['delayed_tasks']/result['total_tasks']*100:.1f}%)
Bottlenecks Detected: {int(result['bottleneck_count'])} ({result['bottleneck_count']/result['total_tasks']*100:.1f}%)
Team Size: {int(result['total_assignees'])} assignees
Active Projects: {int(result['total_projects'])}

KEY INSIGHTS:
- Workflow efficiency can be improved by addressing {int(result['bottleneck_count'])} identified bottlenecks
- {int(result['delayed_tasks'])} tasks exceeded expected completion time
- Immediate action recommended for high-severity bottlenecks
"""
    
    return summary


def generate_bottleneck_analysis():
    """Generate bottleneck analysis section"""
    query = """
    SELECT 
        bottleneck_type,
        COUNT(*) as count,
        AVG(actual_duration) as avg_duration
    FROM tasks
    WHERE bottleneck_type != ''
    GROUP BY bottleneck_type
    ORDER BY count DESC
    """
    
    df = execute_query(query)
    
    text = "BOTTLENECK ANALYSIS\n\n"
    
    for idx, row in df.iterrows():
        # Remove cluster suffix if present
        btype = row['bottleneck_type'].split('_C')[0]
        text += f"{btype}:\n"
        text += f"  - Count: {int(row['count'])} tasks\n"
        text += f"  - Avg Duration: {row['avg_duration']:.1f} days\n\n"
    
    return text


def generate_assignee_performance():
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
    ORDER BY bottleneck_count DESC
    LIMIT 10
    """
    
    df = execute_query(query)
    
    text = "TEAM PERFORMANCE\n\n"
    text += "Top Assignees by Bottleneck Count:\n\n"
    
    for idx, row in df.iterrows():
        text += f"{row['assignee']}:\n"
        text += f"  - Total Tasks: {int(row['task_count'])}\n"
        text += f"  - Avg Duration: {row['avg_duration']:.1f} days\n"
        text += f"  - Delayed: {int(row['delayed_count'])} ({row['delayed_count']/row['task_count']*100:.1f}%)\n"
        text += f"  - Bottlenecks: {int(row['bottleneck_count'])}\n\n"
    
    return text


def generate_gpt_recommendations():
    """Generate GPT recommendations section"""
    query = """
    SELECT 
        task_id,
        root_causes,
        recommendations
    FROM gpt_suggestions
    ORDER BY id
    LIMIT 5
    """
    
    df = execute_query(query)
    
    if len(df) == 0:
        return "AI RECOMMENDATIONS\n\nNo recommendations available. Run GPT suggester first.\n"
    
    text = "AI-POWERED RECOMMENDATIONS\n\n"
    
    for idx, row in df.iterrows():
        text += f"Task: {row['task_id']}\n\n"
        text += f"Root Causes: {row['root_causes'][:200]}...\n\n"
        text += f"Recommendations: {row['recommendations'][:200]}...\n\n"
        text += "-" * 60 + "\n\n"
    
    return text


def generate_pdf_report(output_path=None):
    """Generate complete PDF report"""
    print("\n" + "="*60)
    print("📄 GENERATING PDF REPORT")
    print("="*60 + "\n")
    
    # Create PDF object
    pdf = FlowFixReport()
    pdf.add_page()
    
    # Add sections
    print("Adding executive summary...")
    pdf.chapter_title("EXECUTIVE SUMMARY")
    pdf.chapter_body(generate_executive_summary())
    
    print("Adding bottleneck analysis...")
    pdf.chapter_title("BOTTLENECK ANALYSIS")
    pdf.chapter_body(generate_bottleneck_analysis())
    
    print("Adding team performance...")
    pdf.chapter_title("TEAM PERFORMANCE")
    pdf.chapter_body(generate_assignee_performance())
    
    print("Adding AI recommendations...")
    pdf.add_page()
    pdf.chapter_title("AI-POWERED RECOMMENDATIONS")
    pdf.chapter_body(generate_gpt_recommendations())
    
    # Add recommendations footer
    pdf.add_page()
    pdf.chapter_title("NEXT STEPS")
    pdf.chapter_body("""
1. IMMEDIATE ACTIONS:
   - Address high-severity bottlenecks identified in this report
   - Redistribute workload for overloaded team members
   - Clear blocked tasks by resolving dependencies

2. SHORT-TERM (1-2 weeks):
   - Implement AI recommendations for top bottlenecks
   - Monitor improvement in task completion rates
   - Review team capacity and adjust assignments

3. LONG-TERM (1 month+):
   - Establish regular bottleneck review meetings
   - Track improvement metrics over time
   - Refine processes based on data insights
   - Consider automation for repetitive bottlenecks

For questions or support, refer to the FlowFix AI documentation.
    """)
    
    # Save PDF
    if output_path is None:
        # Get project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        exports_dir = os.path.join(os.path.dirname(current_dir), 'exports')
        os.makedirs(exports_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(exports_dir, f'flowfix_report_{timestamp}.pdf')
    
    pdf.output(output_path)
    
    print(f"\n✅ PDF report generated: {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    
    return output_path


if __name__ == "__main__":
    # Generate report
    report_path = generate_pdf_report()
    print(f"\n📊 Report ready at: {report_path}")
