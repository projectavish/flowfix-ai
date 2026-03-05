"""
FlowFix AI - Professional Enterprise Dashboard
Industry-grade bottleneck analysis and AI-powered workflow optimization
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
from io import BytesIO
import base64
import os
import traceback

# Page config
st.set_page_config(
    page_title="FlowFix AI - Enterprise Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple Black & White CSS with Font Awesome Icons
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * { 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        -webkit-font-smoothing: antialiased;
    }

    .stApp { 
        background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%);
    }

    .main .block-container {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        max-width: 1400px;
        margin: 0 auto;
    }

    .main-header {
        font-size: 2.75rem;
        font-weight: 800;
        text-align: center;
        padding: 1.5rem 0 0.5rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #000000 0%, #374151 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .sub-header {
        text-align: center;
        color: #6b7280 !important;
        font-size: 1.125rem;
        margin-bottom: 2.5rem;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #111827 !important;
        font-weight: 700;
    }

    p, span, div, label {
        color: #374151 !important;
    }

    /* Metrics */
    div[data-testid="stMetricValue"] {
        color: #111827;
        font-size: 2.25rem;
        font-weight: 800;
    }

    div[data-testid="metric-container"] {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #e5e7eb;
        transition: all 0.3s ease;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }

    div[data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        border-color: #000000;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #111827 0%, #1f2937 100%);
    }

    section[data-testid="stSidebar"] * {
        color: #f9fafb !important;
    }

    /* Buttons */
    .stButton button {
        background: #000000;
        color: white;
        border: 2px solid #000000;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton button:hover {
        background: white;
        color: #000000;
        transform: translateY(-2px);
    }

    /* TABS - FIXED VERSION */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        padding: 0.5rem 0;
    }

    .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"] {
        background-color: white;
        color: #4b5563 !important;
        border: 2px solid #e5e7eb;
        border-radius: 10px;
        padding: 0.875rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"]:hover {
        border-color: #000000;
        color: #000000 !important;
        background-color: #f9fafb;
    }

    /* SELECTED TAB - FORCE WHITE TEXT */
    .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #000000 !important;
        border-color: #000000 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }

    .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"] * {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
        padding: 1rem;
    }

    .streamlit-expanderHeader:hover {
        background: #000000;
        color: white;
        border-color: #000000;
    }

    /* Severity Badges */
    .severity-critical {
        color: white;
        font-weight: 700;
        padding: 0.375rem 1rem;
        background: linear-gradient(135deg, #000000 0%, #374151 100%);
        border-radius: 6px;
        font-size: 0.875rem;
    }

    .severity-high {
        color: #000000;
        font-weight: 700;
        padding: 0.375rem 1rem;
        background: white;
        border: 2px solid #000000;
        border-radius: 6px;
        font-size: 0.875rem;
    }

    .severity-medium {
        color: #374151;
        font-weight: 600;
        padding: 0.375rem 1rem;
        background: #f3f4f6;
        border: 1px solid #6b7280;
        border-radius: 6px;
        font-size: 0.875rem;
    }

    .severity-low {
        color: #6b7280;
        font-weight: 500;
        padding: 0.375rem 1rem;
        background: white;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        font-size: 0.875rem;
    }

    /* Icon Cards */
    .icon-card {
        background: linear-gradient(135deg, #000000 0%, #1f2937 100%);
        padding: 1.25rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 0.75rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }

    .icon-card:hover {
        transform: scale(1.05);
    }

    .icon-card i {
        font-size: 2.25rem;
        color: white;
    }

    /* DataFrames */
    .dataframe {
        border: 2px solid #e5e7eb;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }

    .dataframe th {
        background: linear-gradient(135deg, #000000 0%, #1f2937 100%) !important;
        color: white !important;
        font-weight: 600;
        padding: 0.75rem;
    }

    .dataframe tr:hover {
        background-color: #f9fafb;
    }

    /* Scrollbars */
    ::-webkit-scrollbar { 
        width: 10px; 
        height: 10px; 
    }

    ::-webkit-scrollbar-track { 
        background: #f3f4f6;
        border-radius: 5px;
    }

    ::-webkit-scrollbar-thumb {
        background: #6b7280;
        border-radius: 5px;
        border: 2px solid #f3f4f6;
    }

    ::-webkit-scrollbar-thumb:hover { 
        background: #000000;
    }

    /* Plotly Charts */
    .js-plotly-plot {
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }

    /* Inputs */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid #e5e7eb;
    }

    .stSelectbox > div > div:hover {
        border-color: #000000;
    }

    /* Slider */
    .stSlider > div > div > div > div {
        background: #000000;
    }

    /* Sidebar Headers */
    section[data-testid="stSidebar"] h3 {
        font-size: 1.125rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        border-bottom: 1px solid #374151;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------
# Helpers
# -------------------------
def _safe_unique_sorted(df: pd.DataFrame, col: str):
    if df is None or df.empty or col not in df.columns:
        return []
    vals = df[col].dropna().astype(str).unique().tolist()
    return sorted(vals)


def _apply_date_filter(df: pd.DataFrame, date_filter: str):
    """
    Applies a date filter if the dataframe has any supported date column.
    Supported columns: detected_date, created_at, prediction_date
    """
    if df is None or df.empty:
        return df

    date_col = None
    for c in ["detected_date", "created_at", "prediction_date"]:
        if c in df.columns:
            date_col = c
            break

    if not date_col:
        return df

    try:
        tmp = df.copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp = tmp[tmp[date_col].notna()]
        if tmp.empty:
            return tmp

        now = pd.Timestamp.now()
        if date_filter == "Last 7 Days":
            cutoff = now - pd.Timedelta(days=7)
            return tmp[tmp[date_col] >= cutoff]
        if date_filter == "Last 30 Days":
            cutoff = now - pd.Timedelta(days=30)
            return tmp[tmp[date_col] >= cutoff]
        if date_filter == "Last 90 Days":
            cutoff = now - pd.Timedelta(days=90)
            return tmp[tmp[date_col] >= cutoff]

        return tmp  # All Time
    except Exception:
        return df


def apply_filters(df: pd.DataFrame, filters: dict):
    """Apply filters to dataframe, but only when columns exist"""
    if df is None or df.empty:
        return df

    filtered = df.copy()

    # Date filter (schema-aware)
    filtered = _apply_date_filter(filtered, filters.get("date_filter", "All Time"))

    # Column-aware categorical filters
    if filters.get("project") and filters["project"] != "All Projects" and "project" in filtered.columns:
        filtered = filtered[filtered["project"] == filters["project"]]

    if filters.get("assignee") and filters["assignee"] != "All Assignees" and "assignee" in filtered.columns:
        filtered = filtered[filtered["assignee"] == filters["assignee"]]

    if filters.get("priority") and filters["priority"] != "All Priorities" and "priority" in filtered.columns:
        filtered = filtered[filtered["priority"] == filters["priority"]]

    if filters.get("status") and filters["status"] != "All Statuses" and "status" in filtered.columns:
        filtered = filtered[filtered["status"] == filters["status"]]

    # Severity filter only if severity_score exists
    if "severity_score" in filtered.columns and "severity_range" in filters:
        lo, hi = filters["severity_range"]
        filtered = filtered[
            (pd.to_numeric(filtered["severity_score"], errors="coerce") >= lo) &
            (pd.to_numeric(filtered["severity_score"], errors="coerce") <= hi)
        ]

    return filtered


# -------------------------
# PDF Export Helper - FIXED with better error handling
# -------------------------
def generate_pdf_report(tasks_df, bottlenecks_df, filtered_tasks, filtered_bottlenecks, filters, metrics):
    """Generate a comprehensive PDF report with actual useful content"""
    try:
        from fpdf import FPDF
        
        class PDF(FPDF):
            def header(self):
                self.set_font('Helvetica', 'B', 16)
                self.cell(0, 10, 'FlowFix AI - Workflow Analysis Report', 0, 1, 'C')
                self.set_font('Helvetica', '', 10)
                self.cell(0, 5, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
                self.ln(5)
            
            def footer(self):
                self.set_y(-15)
                self.set_font('Helvetica', 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
        pdf = PDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Executive Summary
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Executive Summary', 0, 1)
        pdf.set_font('Helvetica', '', 10)
        
        summary_text = (
            f"This report analyzes {len(tasks_df):,} total tasks with {len(bottlenecks_df):,} "
            f"identified bottlenecks. Current view shows {len(filtered_tasks):,} filtered tasks "
            f"with {len(filtered_bottlenecks):,} bottlenecks."
        )
        pdf.multi_cell(0, 6, summary_text)
        pdf.ln(3)
        
        # Key Metrics
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 10, 'Key Performance Indicators', 0, 1)
        pdf.set_font('Helvetica', '', 10)
        
        col_width = 95
        total_tasks = metrics.get("total_tasks", len(tasks_df))
        completed_tasks = metrics.get("completed_tasks", 0)
        completion_rate = (completed_tasks / max(total_tasks, 1) * 100)
        
        pdf.cell(col_width, 8, f'Total Tasks: {total_tasks:,}', 0, 0)
        pdf.cell(col_width, 8, f'Completion Rate: {completion_rate:.1f}%', 0, 1)
        pdf.cell(col_width, 8, f'Bottlenecks: {len(filtered_bottlenecks):,}', 0, 0)
        pdf.cell(col_width, 8, f'Bottleneck Rate: {metrics.get("bottleneck_rate", 0):.1f}%', 0, 1)
        pdf.cell(col_width, 8, f'Average Duration: {metrics.get("avg_duration", 0):.1f} days', 0, 0)
        pdf.cell(col_width, 8, f'Delayed Tasks: {metrics.get("delayed_tasks", 0):,}', 0, 1)
        pdf.ln(5)
        
        # Active Filters
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 10, 'Active Filters', 0, 1)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 6, f"Date Range: {filters.get('date_filter', 'All Time')}", 0, 1)
        pdf.cell(0, 6, f"Project: {filters.get('project', 'All Projects')}", 0, 1)
        pdf.cell(0, 6, f"Assignee: {filters.get('assignee', 'All Assignees')}", 0, 1)
        pdf.cell(0, 6, f"Priority: {filters.get('priority', 'All Priorities')}", 0, 1)
        severity_range = filters.get('severity_range', (0, 100))
        pdf.cell(0, 6, f"Severity Range: {severity_range[0]} - {severity_range[1]}", 0, 1)
        pdf.ln(5)
        
        # Bottleneck Details
        if len(filtered_bottlenecks) > 0:
            pdf.add_page()
            pdf.set_font('Helvetica', 'B', 14)
            pdf.cell(0, 10, 'Bottleneck Analysis', 0, 1)
            pdf.ln(2)
            
            # Summary stats
            pdf.set_font('Helvetica', 'B', 12)
            pdf.cell(0, 8, 'Bottleneck Summary', 0, 1)
            pdf.set_font('Helvetica', '', 10)
            
            if 'severity_score' in filtered_bottlenecks.columns:
                severity = pd.to_numeric(filtered_bottlenecks['severity_score'], errors='coerce')
                pdf.cell(0, 6, f'Average Severity: {severity.mean():.1f}/100', 0, 1)
                pdf.cell(0, 6, f'Critical (>70): {(severity > 70).sum()} bottlenecks', 0, 1)
            
            if 'bottleneck_type' in filtered_bottlenecks.columns:
                type_counts = filtered_bottlenecks['bottleneck_type'].value_counts()
                if len(type_counts) > 0:
                    pdf.cell(0, 6, f'Most Common Type: {type_counts.index[0]} ({type_counts.iloc[0]} cases)', 0, 1)
            
            pdf.ln(5)
            
            # Top Bottlenecks Table
            pdf.set_font('Helvetica', 'B', 12)
            pdf.cell(0, 8, 'Top 10 Critical Bottlenecks', 0, 1)
            pdf.set_font('Helvetica', 'B', 9)
            
            # Table headers
            pdf.set_fill_color(0, 0, 0)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(40, 7, 'Task', 1, 0, 'C', True)
            pdf.cell(35, 7, 'Type', 1, 0, 'C', True)
            pdf.cell(25, 7, 'Severity', 1, 0, 'C', True)
            pdf.cell(30, 7, 'Assignee', 1, 0, 'C', True)
            pdf.cell(30, 7, 'Project', 1, 0, 'C', True)
            pdf.cell(30, 7, 'Status', 1, 1, 'C', True)
            
            pdf.set_text_color(0, 0, 0)
            pdf.set_font('Helvetica', '', 9)
            
            # Sort by severity if available
            display_df = filtered_bottlenecks.copy()
            if 'severity_score' in display_df.columns:
                display_df['severity_score_num'] = pd.to_numeric(display_df['severity_score'], errors='coerce')
                display_df = display_df.sort_values('severity_score_num', ascending=False)
            
            for idx, row in display_df.head(10).iterrows():
                task_name = str(row.get('task_name', row.get('task_id', 'N/A')))[:18]
                btype = str(row.get('bottleneck_type', 'N/A'))[:15]
                severity = str(row.get('severity_score', 'N/A'))[:8]
                assignee = str(row.get('assignee', 'N/A'))[:13]
                project = str(row.get('project', 'N/A'))[:13]
                status = str(row.get('status', 'N/A'))[:13]
                
                pdf.cell(40, 6, task_name, 1)
                pdf.cell(35, 6, btype, 1)
                pdf.cell(25, 6, severity, 1)
                pdf.cell(30, 6, assignee, 1)
                pdf.cell(30, 6, project, 1)
                pdf.cell(30, 6, status, 1)
                pdf.ln()
            
            pdf.ln(5)
            
            # Root Causes
            if 'root_cause_suggestion' in filtered_bottlenecks.columns:
                root_causes = filtered_bottlenecks['root_cause_suggestion'].dropna()
                if len(root_causes) > 0:
                    pdf.set_font('Helvetica', 'B', 12)
                    pdf.cell(0, 8, 'Common Root Causes', 0, 1)
                    pdf.set_font('Helvetica', '', 10)
                    
                    for i, (cause, count) in enumerate(root_causes.value_counts().head(5).items(), 1):
                        cause_text = f"{i}. {cause} ({count} occurrences)"
                        pdf.cell(0, 6, cause_text, 0, 1)
        
        # Team Performance
        if len(filtered_tasks) > 0 and 'assignee' in filtered_tasks.columns:
            pdf.add_page()
            pdf.set_font('Helvetica', 'B', 14)
            pdf.cell(0, 10, 'Team Performance', 0, 1)
            pdf.ln(2)
            
            pdf.set_font('Helvetica', 'B', 12)
            pdf.cell(0, 8, f'Team Members: {filtered_tasks["assignee"].nunique()}', 0, 1)
            
            if 'status' in filtered_tasks.columns:
                completed = (filtered_tasks['status'] == 'Completed').sum()
                total = len(filtered_tasks)
                pdf.cell(0, 6, f'Overall Completion Rate: {completed/total*100:.1f}% ({completed}/{total})', 0, 1)
            
            pdf.ln(3)
            
            # Top Performers
            pdf.set_font('Helvetica', 'B', 12)
            pdf.cell(0, 8, 'Workload Distribution', 0, 1)
            pdf.set_font('Helvetica', 'B', 9)
            
            pdf.set_fill_color(0, 0, 0)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(60, 7, 'Assignee', 1, 0, 'C', True)
            pdf.cell(40, 7, 'Total Tasks', 1, 0, 'C', True)
            pdf.cell(45, 7, 'Completed', 1, 0, 'C', True)
            pdf.cell(45, 7, 'Completion %', 1, 1, 'C', True)
            
            pdf.set_text_color(0, 0, 0)
            pdf.set_font('Helvetica', '', 9)
            
            if 'status' in filtered_tasks.columns:
                perf = filtered_tasks.groupby('assignee').agg({
                    'task_id': 'count',
                    'status': lambda x: (x == 'Completed').sum()
                }).reset_index()
                perf.columns = ['Assignee', 'Total', 'Completed']
                perf['Rate'] = (perf['Completed'] / perf['Total'] * 100).round(1)
                perf = perf.sort_values('Total', ascending=False)
                
                for _, row in perf.head(15).iterrows():
                    pdf.cell(60, 6, str(row['Assignee'])[:25], 1)
                    pdf.cell(40, 6, str(row['Total']), 1, 0, 'C')
                    pdf.cell(45, 6, str(row['Completed']), 1, 0, 'C')
                    rate_text = f"{row['Rate']:.1f}%"
                    pdf.cell(45, 6, rate_text, 1, 0, 'C')
                    pdf.ln()
        
        # FIXED: Moved outside the if block - always return pdf_bytes at end of function
        pdf_bytes = pdf.output(dest='BYTES')
        return pdf_bytes
        
    except Exception as e:
        error_detail = traceback.format_exc()
        st.error(f"PDF Generation Error: {str(e)}")
        with st.expander("Show detailed error"):
            st.code(error_detail)
        return None


# -------------------------
# UI Components
# -------------------------
def render_sidebar(tasks_df: pd.DataFrame):
    """Render professional sidebar with filters and controls"""
    
    # Initialize default values in session state if not present
    if 'reset_trigger' not in st.session_state:
        st.session_state.reset_trigger = False
    
    # If reset was triggered, set defaults
    if st.session_state.reset_trigger:
        st.session_state.date_filter_key = "All Time"
        st.session_state.project_key = "All Projects"
        st.session_state.assignee_key = "All Assignees"
        st.session_state.priority_key = "All Priorities"
        st.session_state.status_key = "All Statuses"
       
        st.session_state.reset_trigger = False
        st.rerun()
    
    st.sidebar.markdown(
        '<h1 style="text-align: center;"><i class="fas fa-rocket"></i> FlowFix AI</h1>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        '<p style="text-align: center; opacity: 0.9;">Enterprise Workflow Optimizer</p>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown("---")

    # RESET BUTTON
    if st.sidebar.button("🔄 Reset All Filters", type="primary"):
        st.session_state.reset_trigger = True
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown('<h3><i class="fas fa-filter"></i> Filters</h3>', unsafe_allow_html=True)

    # Date range filter
    st.sidebar.markdown('<p style="margin-top: 1rem;"><i class="fas fa-calendar"></i> <strong>Date Range</strong></p>', unsafe_allow_html=True)
    date_filter = st.sidebar.radio(
        "Select Period",
        ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"],
        key='date_filter_key'
    )

    # Project filter
    projects = ["All Projects"] + _safe_unique_sorted(tasks_df, "project")
    selected_project = st.sidebar.selectbox("🏢 Project", projects, key='project_key')

    # Assignee filter
    assignees = ["All Assignees"] + _safe_unique_sorted(tasks_df, "assignee")
    selected_assignee = st.sidebar.selectbox("👤 Assignee", assignees, key='assignee_key')

    # Priority filter
    priorities = ["All Priorities", "Critical", "High", "Medium", "Low"]
    selected_priority = st.sidebar.selectbox("🎯 Priority", priorities, key='priority_key')

    # Status filter
    statuses = ["All Statuses"] + _safe_unique_sorted(tasks_df, "status")
    selected_status = st.sidebar.selectbox("📌 Status", statuses, key='status_key')

    # Severity filter
    st.sidebar.markdown('<p style="margin-top: 1rem;"><i class="fas fa-fire"></i> <strong>Severity Threshold</strong></p>', unsafe_allow_html=True)
    severity_range = st.sidebar.slider("Score Range", 0, 100,(0, 100), key='severity_key')

    return {
        "project": selected_project,
        "assignee": selected_assignee,
        "priority": selected_priority,
        "status": selected_status,
        "severity_range": severity_range,
        "date_filter": date_filter
    }


def render_executive_dashboard(tasks_df, bottlenecks_df, suggestions_df, metrics):
    """Render executive overview with key metrics - FIXED: Only called once"""
    st.markdown('''
        <h1 class="main-header">
            <i class="fas fa-chart-line"></i> FlowFix AI Enterprise Dashboard
        </h1>
    ''', unsafe_allow_html=True)
    st.markdown('''
        <p class="sub-header">
            <i class="fas fa-brain"></i> Real-time Workflow Intelligence & AI-Powered Bottleneck Resolution
        </p>
    ''', unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    total_tasks = int(metrics.get("total_tasks", len(tasks_df) if tasks_df is not None else 0))
    completed_tasks = int(metrics.get("completed_tasks", 0))
    completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0

    with col1:
        st.markdown('<div class="icon-card"><i class="fas fa-tasks"></i></div>', unsafe_allow_html=True)
        st.metric("Total Tasks", f"{total_tasks:,}", f"{completion_rate:.1f}% complete")

    with col2:
        active_tasks = 0
        if tasks_df is not None and not tasks_df.empty and "status" in tasks_df.columns:
            active_tasks = int((tasks_df["status"] == "In Progress").sum())
        st.markdown('<div class="icon-card"><i class="fas fa-bolt"></i></div>', unsafe_allow_html=True)
        st.metric("Active Tasks", f"{active_tasks:,}", f"{(active_tasks/total_tasks*100):.1f}%" if total_tasks > 0 else "0%")

    with col3:
        bottleneck_count = len(bottlenecks_df) if bottlenecks_df is not None else 0
        bottleneck_rate = float(metrics.get("bottleneck_rate", 0))
        st.markdown('<div class="icon-card"><i class="fas fa-exclamation-triangle"></i></div>', unsafe_allow_html=True)
        st.metric("Bottlenecks", f"{bottleneck_count:,}", f"{bottleneck_rate:.1f}%", delta_color="inverse")

    with col4:
        delay_rate = float(metrics.get("delay_rate", 0))
        delayed_tasks = int(metrics.get("delayed_tasks", 0))
        st.markdown('<div class="icon-card"><i class="fas fa-clock"></i></div>', unsafe_allow_html=True)
        st.metric("Delays", f"{delayed_tasks:,}", f"{delay_rate:.1f}%", delta_color="inverse")

    with col5:
        avg_duration = float(metrics.get("avg_duration", 0))
        st.markdown('<div class="icon-card"><i class="fas fa-chart-bar"></i></div>', unsafe_allow_html=True)
        st.metric("Avg Duration", f"{avg_duration:.1f}d", "Target: 5d")

    with col6:
        ai_suggestions = len(suggestions_df) if suggestions_df is not None else 0
        st.markdown('<div class="icon-card"><i class="fas fa-robot"></i></div>', unsafe_allow_html=True)
        st.metric("AI Insights", f"{ai_suggestions:,}", f"+{ai_suggestions} new")

    st.markdown("---")

    colA, colB = st.columns([2, 1])

    with colA:
        if bottlenecks_df is not None and not bottlenecks_df.empty and "detected_date" in bottlenecks_df.columns:
            df = bottlenecks_df.copy()
            df["detected_date"] = pd.to_datetime(df["detected_date"], errors="coerce")
            df = df[df["detected_date"].notna()]

            if not df.empty:
                id_col = "bottleneck_id" if "bottleneck_id" in df.columns else ("id" if "id" in df.columns else None)
                if id_col is None:
                    df["_count_helper"] = 1
                    id_col = "_count_helper"

                daily = df.groupby(df["detected_date"].dt.date).agg({
                    id_col: "count",
                    "severity_score": "mean" if "severity_score" in df.columns else "count"
                }).reset_index()

                daily.columns = ["Date", "Count", "Avg_Severity"]

                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(x=daily["Date"], y=daily["Count"], name="Bottlenecks"), secondary_y=False)
                fig.add_trace(go.Scatter(
                    x=daily["Date"], y=daily["Avg_Severity"], name="Avg Severity",
                    line=dict(width=3), mode="lines+markers"
                ), secondary_y=True)

                fig.update_layout(
                    title="📈 Bottleneck Detection Trends",
                    hovermode="x unified",
                    height=400,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                fig.update_xaxes(title_text="Date")
                fig.update_yaxes(title_text="Bottleneck Count", secondary_y=False)
                fig.update_yaxes(title_text="Severity Score", secondary_y=True)

                st.plotly_chart(fig, use_container_width=True)

    with colB:
        if bottlenecks_df is not None and not bottlenecks_df.empty and "priority" in bottlenecks_df.columns:
            priority_counts = bottlenecks_df["priority"].value_counts()

            fig = go.Figure(data=[go.Pie(
                labels=priority_counts.index,
                values=priority_counts.values,
                hole=0.4,
                textinfo="label+percent",
                textposition="outside"
            )])

            fig.update_layout(title="🎯 Priority Distribution", height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


def render_bottleneck_analysis(bottlenecks_df, filters):
    """Comprehensive bottleneck analysis"""
    st.header("🔍 Bottleneck Deep Dive")

    if bottlenecks_df is None or bottlenecks_df.empty:
        st.success("✅ No bottlenecks detected! Your workflow is running smoothly.")
        return

    filtered = apply_filters(bottlenecks_df, filters)

    if filtered is None or filtered.empty:
        st.info("No bottlenecks match your filters")
        return

    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔬 Root Cause", "🎯 Impact"])

    with tab1:
        col1, col2, col3 = st.columns(3)

        severity_series = pd.to_numeric(filtered["severity_score"], errors="coerce") if "severity_score" in filtered.columns else pd.Series([], dtype=float)
        with col1:
            critical = int((severity_series > 70).sum()) if not severity_series.empty else 0
            st.metric("Critical Bottlenecks", critical)

        with col2:
            st.metric("Avg Severity", f"{severity_series.mean():.1f}/100" if not severity_series.empty else "N/A")

        with col3:
            resolved = int(filtered["resolution_date"].notna().sum()) if "resolution_date" in filtered.columns else 0
            st.metric("Resolved", f"{resolved} ({(resolved/len(filtered)*100):.1f}%)" if len(filtered) > 0 else "0%")

        colA, colB = st.columns(2)

        with colA:
            if "bottleneck_type" in filtered.columns:
                type_counts = filtered["bottleneck_type"].value_counts().head(10)
                fig = px.bar(
                    x=type_counts.values,
                    y=type_counts.index,
                    orientation="h",
                    title="Top Bottleneck Types",
                    labels={"x": "Count", "y": "Type"}
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)

        with colB:
            if "assignee" in filtered.columns:
                assignee_counts = filtered["assignee"].value_counts().head(10)
                fig = px.bar(
                    x=assignee_counts.values,
                    y=assignee_counts.index,
                    orientation="h",
                    title="Bottlenecks by Assignee",
                    labels={"x": "Count", "y": "Assignee"}
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)

        if "project" in filtered.columns and "bottleneck_type" in filtered.columns and "severity_score" in filtered.columns:
            heatmap_data = filtered.pivot_table(
                values="severity_score",
                index="project",
                columns="bottleneck_type",
                aggfunc="mean"
            ).fillna(0)

            fig = px.imshow(
                heatmap_data,
                title="Severity Heatmap: Project vs Bottleneck Type",
                labels=dict(x="Bottleneck Type", y="Project", color="Avg Severity"),
                aspect="auto"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("🔬 Root Cause Analysis")

        if "root_cause_suggestion" in filtered.columns:
            root_causes = filtered["root_cause_suggestion"].dropna()
            if len(root_causes) > 0:
                colA, colB = st.columns([2, 1])

                with colA:
                    st.markdown("**Most Common Root Causes:**")
                    for i, (cause, cnt) in enumerate(root_causes.value_counts().head(10).items(), 1):
                        st.markdown(f"{i}. **{cause}** ({cnt} occurrences)")

                with colB:
                    st.markdown("**Quick Insights:**")
                    st.markdown(f"- {root_causes.nunique()} unique root causes identified")
                    st.markdown(f"- {len(root_causes)} total diagnoses")
                    st.markdown(f"- Top cause: **{root_causes.value_counts().index[0]}**")
            else:
                st.info("No root cause data available")
        else:
            st.info("No root cause data available")

    with tab3:
        st.subheader("🎯 Business Impact Analysis")
        colA, colB = st.columns(2)

        with colA:
            if "delay_days" in filtered.columns:
                delay_series = pd.to_numeric(filtered["delay_days"], errors="coerce").fillna(0)
                total_delay = float(delay_series.sum())
                avg_delay = float(delay_series.mean()) if len(delay_series) > 0 else 0.0

                st.metric("Total Delay Days", f"{total_delay:.0f}")
                st.metric("Avg Delay per Bottleneck", f"{avg_delay:.1f} days")

                fig = px.histogram(
                    filtered,
                    x="delay_days",
                    nbins=20,
                    title="Delay Distribution",
                    labels={"delay_days": "Delay (days)", "count": "Frequency"}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Delay impact not available (no delay_days column).")

        with colB:
            if "project" in filtered.columns and "severity_score" in filtered.columns:
                id_col = "bottleneck_id" if "bottleneck_id" in filtered.columns else ("id" if "id" in filtered.columns else None)
                if id_col is None:
                    filtered["_count_helper"] = 1
                    id_col = "_count_helper"

                agg_dict = {id_col: "count", "severity_score": "mean"}
                if "delay_days" in filtered.columns:
                    agg_dict["delay_days"] = "sum"

                project_impact = filtered.groupby("project").agg(agg_dict).round(2)
                cols = ["Bottlenecks", "Avg Severity"] + (["Total Delay"] if "delay_days" in filtered.columns else [])
                project_impact.columns = cols
                project_impact = project_impact.sort_values("Bottlenecks", ascending=False)

                st.dataframe(project_impact.head(10), use_container_width=True, height=350)
            else:
                st.info("Project impact not available.")

    st.markdown("---")
    st.subheader("📋 Detailed Bottleneck Records")

    display_cols = ["task_id", "task_name", "assignee", "project", "bottleneck_type", "severity_score", "priority", "status", "detected_date"]
    existing_cols = [c for c in display_cols if c in filtered.columns]

    if existing_cols:
        st.dataframe(filtered[existing_cols].head(100), use_container_width=True, height=400)
    else:
        st.info("No displayable bottleneck columns found.")


def render_ai_recommendations(suggestions_df, filters):
    """Display AI-powered recommendations"""
    st.header("🤖 AI-Powered Recommendations")

    if suggestions_df is None or suggestions_df.empty:
        st.info("No AI suggestions generated yet. Run the GPT suggester to generate recommendations.")
        return

    filtered = apply_filters(suggestions_df, filters)

    if filtered is None or filtered.empty:
        st.info("No suggestions match your filters")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Suggestions", len(filtered))

    with col2:
        if "quality_score" in filtered.columns:
            q = pd.to_numeric(filtered["quality_score"], errors="coerce")
            high_quality = int((q >= 80).sum())
        else:
            high_quality = 0
        st.metric("High Quality", high_quality)

    with col3:
        applied = int(pd.to_numeric(filtered["applied"], errors="coerce").fillna(0).sum()) if "applied" in filtered.columns else 0
        st.metric("Applied", applied)

    with col4:
        avg_quality = float(pd.to_numeric(filtered["quality_score"], errors="coerce").mean()) if "quality_score" in filtered.columns else 0
        st.metric("Avg Quality", f"{avg_quality:.0f}/100" if avg_quality else "N/A")

    st.markdown("---")
    colA, colB = st.columns([2, 1])

    with colA:
        st.subheader("📝 Recent Suggestions")

        for _, row in filtered.head(10).iterrows():
            title = str(row.get("task_name", "Task"))[:60]
            quality = row.get("quality_score", 0)
            with st.expander(f"🎯 {title}... - Quality: {quality}/100"):
                st.markdown(f"**Task ID:** {row.get('task_id', 'N/A')}")
                st.markdown(f"**Assignee:** {row.get('assignee', 'N/A')} | **Project:** {row.get('project', 'N/A')}")
                st.markdown(f"**Priority:** {row.get('priority', 'N/A')} | **Urgency:** {row.get('urgency', 'N/A')}")

                st.markdown("**💡 AI Suggestion:**")
                st.markdown(str(row.get("suggestion_text", ""))[:500] + ("..." if len(str(row.get("suggestion_text", ""))) > 500 else ""))

                root_causes = row.get("root_causes", None)
                if pd.notna(root_causes) and str(root_causes).strip():
                    st.markdown("**🔍 Root Causes:**")
                    st.markdown(str(root_causes)[:300])

                recs = row.get("recommendations", None)
                if pd.notna(recs) and str(recs).strip():
                    st.markdown("**✅ Recommendations:**")
                    st.markdown(str(recs)[:300])

                colx, coly, colz = st.columns(3)
                with colx:
                    st.button("👍 Apply", key=f"apply_{row.get('task_id','na')}_{row.name}")
                with coly:
                    st.button("👎 Dismiss", key=f"dismiss_{row.get('task_id','na')}_{row.name}")
                with colz:
                    st.button("📋 Copy", key=f"copy_{row.get('task_id','na')}_{row.name}")

    with colB:
        st.subheader("📊 Suggestion Analytics")

        if "quality_score" in filtered.columns:
            fig = px.histogram(
                filtered,
                x="quality_score",
                nbins=10,
                title="Quality Score Distribution",
                labels={"quality_score": "Quality Score", "count": "Count"}
            )
            st.plotly_chart(fig, use_container_width=True)

        if "urgency" in filtered.columns:
            urgency_counts = filtered["urgency"].astype(str).value_counts()
            fig = px.pie(values=urgency_counts.values, names=urgency_counts.index, title="Urgency Breakdown")
            st.plotly_chart(fig, use_container_width=True)

        if "model_used" in filtered.columns and len(filtered) > 0:
            st.markdown("**🎯 Model Stats:**")
            st.markdown(f"- Model: {filtered['model_used'].iloc[0]}")
            if "quality_score" in filtered.columns:
                st.markdown(f"- Avg Quality: {pd.to_numeric(filtered['quality_score'], errors='coerce').mean():.1f}/100")
            st.markdown(f"- Total Generated: {len(filtered)}")


def render_team_performance(tasks_df, filters):
    """Team and assignee performance analysis"""
    st.header("👥 Team Performance")

    if tasks_df is None or tasks_df.empty:
        st.info("No tasks data available.")
        return

    filtered = apply_filters(tasks_df, filters)

    col1, col2, col3, col4 = st.columns(4)

    total_assignees = filtered["assignee"].nunique() if "assignee" in filtered.columns else 0

    with col1:
        st.metric("Team Members", total_assignees)

    with col2:
        avg_tasks = (len(filtered) / total_assignees) if total_assignees > 0 else 0
        st.metric("Avg Tasks/Person", f"{avg_tasks:.1f}")

    with col3:
        completed = int((filtered["status"] == "Completed").sum()) if "status" in filtered.columns else 0
        st.metric("Completed", completed)

    with col4:
        in_progress = int((filtered["status"] == "In Progress").sum()) if "status" in filtered.columns else 0
        st.metric("In Progress", in_progress)

    st.markdown("---")
    colA, colB = st.columns(2)

    with colA:
        if "assignee" in filtered.columns and not filtered.empty:
            assignee_tasks = filtered["assignee"].value_counts().head(15)
            if not assignee_tasks.empty:
                fig = px.bar(
                    x=assignee_tasks.values,
                    y=assignee_tasks.index,
                    orientation="h",
                    title="Task Distribution by Assignee",
                    labels={"x": "Number of Tasks", "y": "Assignee"}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No assignee data for selected filters")
        else:
            st.info("No assignee data available")

    with colB:
        if "assignee" in filtered.columns and "status" in filtered.columns and "task_id" in filtered.columns and not filtered.empty:
            assignee_stats = filtered.groupby("assignee").agg(
                task_count=("task_id", "count"),
                completed=("status", lambda x: (x == "Completed").sum())
            )
            if not assignee_stats.empty:
                assignee_stats["completion_rate"] = (assignee_stats["completed"] / assignee_stats["task_count"] * 100).round(1)
                assignee_stats = assignee_stats.sort_values("completion_rate", ascending=False).head(15)

                fig = px.bar(
                    x=assignee_stats["completion_rate"],
                    y=assignee_stats.index,
                    orientation="h",
                    title="Completion Rate by Assignee (%)",
                    labels={"x": "Completion Rate (%)", "y": "Assignee"}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No completion data for selected filters")
        else:
            st.info("No status data available")

    st.subheader("🏆 Performance Leaderboard")

    needed = {"assignee", "task_id", "status"}
    if needed.issubset(set(filtered.columns)) and not filtered.empty:
        leaderboard = filtered.groupby("assignee").agg(
            total_tasks=("task_id", "count"),
            completed=("status", lambda x: (x == "Completed").sum())
        )

        if "actual_duration" in filtered.columns:
            leaderboard["avg_duration"] = filtered.groupby("assignee")["actual_duration"].mean()

        if "priority" in filtered.columns:
            leaderboard["high_priority"] = filtered.groupby("assignee")["priority"].apply(lambda x: (x == "High").sum())

        leaderboard["completion_pct"] = (leaderboard["completed"] / leaderboard["total_tasks"] * 100).round(1)
        leaderboard = leaderboard.sort_values("completed", ascending=False)

        st.dataframe(leaderboard.head(20), use_container_width=True, height=400)
    else:
        st.info("Leaderboard not available due to missing columns.")


def render_project_insights(tasks_df, filters):
    """Project-level insights and analytics"""
    st.header("🏢 Project Insights")

    if tasks_df is None or tasks_df.empty:
        st.info("No tasks data available.")
        return

    filtered = apply_filters(tasks_df, filters)
    
    # GUARD: Check if filtered data exists
    if filtered is None or filtered.empty:
        st.info("No projects match your current filters. Try adjusting the filters in the sidebar.")
        return

    col1, col2, col3, col4 = st.columns(4)

    total_projects = filtered["project"].nunique() if "project" in filtered.columns else 0

    with col1:
        st.metric("Active Projects", total_projects)

    with col2:
        avg_tasks = (len(filtered) / total_projects) if total_projects > 0 else 0
        st.metric("Avg Tasks/Project", f"{avg_tasks:.1f}")

    with col3:
        if "actual_duration" in filtered.columns:
            st.metric("Avg Duration", f"{pd.to_numeric(filtered['actual_duration'], errors='coerce').mean():.1f}d")
        else:
            st.metric("Avg Duration", "N/A")

    with col4:
        if "priority" in filtered.columns:
            high_priority = int(filtered["priority"].isin(["High", "Critical"]).sum())
        else:
            high_priority = 0
        st.metric("High Priority", high_priority)

    st.markdown("---")
    colA, colB = st.columns(2)

    with colA:
        if "project" in filtered.columns:
            project_tasks = filtered["project"].value_counts().head(15)
            # CRITICAL FIX: Check if data exists before plotting
            if not project_tasks.empty:
                fig = px.bar(
                    x=project_tasks.values,
                    y=project_tasks.index,
                    orientation="h",
                    title="Tasks by Project",
                    labels={"x": "Number of Tasks", "y": "Project"}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No project data to display")
        else:
            st.info("No project column found")

    with colB:
        if {"project", "status"}.issubset(set(filtered.columns)):
            # Check if we have data to group
            if not filtered.empty:
                project_status = filtered.groupby(["project", "status"]).size().reset_index(name="count")
                if not project_status.empty:
                    fig = px.bar(
                        project_status,
                        x="project",
                        y="count",
                        color="status",
                        title="Status Breakdown by Project",
                        labels={"count": "Number of Tasks", "project": "Project"},
                        barmode="stack"
                    )
                    fig.update_layout(height=500, xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No status data available for chart")
            else:
                st.info("No data for status breakdown")
        else:
            st.info("Project or Status data missing")

    st.subheader("📊 Project Performance Summary")

    needed = {"project", "task_id", "status"}
    if needed.issubset(set(filtered.columns)) and not filtered.empty:
        try:
            project_stats = filtered.groupby("project").agg(
                total_tasks=("task_id", "count"),
                completed=("status", lambda x: (x == "Completed").sum())
            )

            if "actual_duration" in filtered.columns:
                project_stats["avg_duration"] = filtered.groupby("project")["actual_duration"].mean()

            if "priority" in filtered.columns:
                project_stats["high_priority"] = filtered.groupby("project")["priority"].apply(lambda x: x.isin(["High", "Critical"]).sum())

            project_stats["completion_pct"] = (project_stats["completed"] / project_stats["total_tasks"] * 100).round(1)
            project_stats = project_stats.sort_values("total_tasks", ascending=False)

            if not project_stats.empty:
                st.dataframe(project_stats, use_container_width=True, height=400)
            else:
                st.info("No project statistics available")
        except Exception as e:
            st.error("Could not generate project statistics")
    else:
        st.info("Project summary not available due to missing columns.")


# -------------------------
# Main
# -------------------------
def main():
    """Main application - CSV only, no database"""
    
    st.sidebar.markdown("---")
    st.sidebar.markdown('<h3><i class="fas fa-upload"></i> Data Source</h3>', unsafe_allow_html=True)
    
    # Auto-generate test data if it doesn't exist
    default_csv_path = "test_tasks.csv"
    if not os.path.exists(default_csv_path):
        with st.spinner("Generating demo data..."):
            # Generate fresh data
            np.random.seed(42)
            n = 500
            end_date = datetime(2026, 2, 26)
            dates = [end_date - timedelta(days=int(x)) for x in np.random.randint(0, 100, n)]
            
            df = pd.DataFrame({
                'task_id': [f'T{i:04d}' for i in range(n)],
                'task_name': [f'Task {i}' for i in range(n)],
                'assignee': np.random.choice(['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank'], n),
                'project': np.random.choice(['Alpha', 'Beta', 'Gamma', 'Delta'], n),
                'status': np.random.choice(['In Progress', 'Completed', 'Blocked', 'Not Started'], n, p=[0.4, 0.3, 0.2, 0.1]),
                'priority': np.random.choice(['Critical', 'High', 'Medium', 'Low'], n, p=[0.1, 0.3, 0.4, 0.2]),
                'created_at': dates,
                'actual_duration': np.random.randint(1, 20, n),
                'bottleneck_type': np.random.choice(['Process', 'Resource', 'Technical', 'Communication', None], n, p=[0.2, 0.2, 0.2, 0.2, 0.2]),
                'severity_score': np.random.randint(10, 95, n),
                'detected_date': [d if np.random.random() > 0.3 else None for d in dates],
                'root_cause_suggestion': np.random.choice([
                    'Resource constraint', 'Unclear requirements', 'Technical debt', 
                    'Communication gap', 'External dependency', None
                ], n, p=[0.2, 0.2, 0.2, 0.2, 0.15, 0.05])
            })
            df.to_csv(default_csv_path, index=False)
    
    # File upload option
    uploaded_file = st.sidebar.file_uploader("Upload your CSV (optional)", type=['csv'])
    
    if uploaded_file is not None:
        tasks_df = pd.read_csv(uploaded_file)
        source_name = uploaded_file.name
        st.sidebar.success(f"📤 Using: {source_name}")
    else:
        tasks_df = pd.read_csv(default_csv_path)
        source_name = "Demo Data"
        st.sidebar.info("📊 Using demo data (upload your own CSV to override)")
    
    # Convert dates
    for col in ['created_at', 'detected_date']:
        if col in tasks_df.columns:
            tasks_df[col] = pd.to_datetime(tasks_df[col], errors='coerce')
    
    # Create bottleneck subset
    if 'bottleneck_type' in tasks_df.columns:
        bottlenecks_df = tasks_df[tasks_df['bottleneck_type'].notna()].copy()
    else:
        bottlenecks_df = pd.DataFrame()
    
    suggestions_df = pd.DataFrame()
    
       # Sidebar filters
    filters = render_sidebar(tasks_df)
    
    # Apply filters FIRST
    filtered_tasks = apply_filters(tasks_df, filters)
    filtered_bottlenecks = apply_filters(bottlenecks_df, filters) if not bottlenecks_df.empty else bottlenecks_df
    
    # THEN calculate metrics on FILTERED data
    if "actual_duration" in filtered_tasks.columns:
        delayed_mask = filtered_tasks["actual_duration"] > 10
        delayed_count = int(delayed_mask.sum())
        delay_rate = (delayed_count / len(filtered_tasks) * 100) if len(filtered_tasks) > 0 else 0
        avg_duration = filtered_tasks["actual_duration"].mean()
    else:
        delayed_count = 0
        delay_rate = 0
        avg_duration = 0

    metrics = {
        "total_tasks": len(filtered_tasks),
        "completed_tasks": int((filtered_tasks["status"] == "Completed").sum()) if "status" in filtered_tasks.columns else 0,
        "bottleneck_rate": round(len(filtered_bottlenecks)/len(filtered_tasks)*100, 1) if len(filtered_tasks) > 0 else 0,
        "delay_rate": round(delay_rate, 1),
        "delayed_tasks": delayed_count,
        "avg_duration": avg_duration
    }
    
    # Sidebar stats
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Total Tasks:** {len(tasks_df):,}")
    st.sidebar.markdown(f"**Showing:** {len(filtered_tasks):,} filtered tasks")
    if 'created_at' in tasks_df.columns:
        st.sidebar.markdown(f"**Date Range:** {tasks_df['created_at'].min().date()} to {tasks_df['created_at'].max().date()}")

    # Export to PDF button - FIXED: Uses new comprehensive PDF generator
    st.sidebar.markdown("---")
    if st.sidebar.button("📄 Export PDF Report"):
        pdf_bytes = generate_pdf_report(tasks_df, bottlenecks_df, filtered_tasks, filtered_bottlenecks, filters, metrics)
        if pdf_bytes:
            b64 = base64.b64encode(pdf_bytes).decode()
            filename = f"flowfix_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">📥 Click to Download PDF</a>'
            st.sidebar.markdown(href, unsafe_allow_html=True)
            st.sidebar.success("Report generated successfully!")
        else:
            st.sidebar.error("PDF generation failed. Check error details above.")

    # Render dashboard - FIXED: Only called once here
    render_executive_dashboard(filtered_tasks, filtered_bottlenecks, suggestions_df, metrics)

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚨 Bottleneck Analysis",
        "🤖 AI Recommendations", 
        "👥 Team Performance",
        "🏢 Project Insights",
        "📈 Data Summary"
    ])

    with tab1:
        try:
            render_bottleneck_analysis(filtered_bottlenecks, filters)
        except Exception as e:
            st.error(f"Bottleneck Analysis error: {str(e)}")

    with tab2:
        try:
            render_ai_recommendations(suggestions_df, filters)
        except Exception as e:
            st.error(f"AI Recommendations error: {str(e)}")

    with tab3:
        try:
            render_team_performance(filtered_tasks, filters)
        except Exception as e:
            st.error(f"Team Performance error: {str(e)}")

    with tab4:
        try:
            render_project_insights(filtered_tasks, filters)
        except Exception as e:
            st.error(f"Project Insights error: {str(e)}")

    with tab5:
        st.header("📊 Data Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Source Info")
            st.write(f"**Source:** {source_name}")
            st.write(f"**Total Records:** {len(tasks_df):,}")
            st.write(f"**Bottlenecks:** {len(bottlenecks_df):,}")
            
            if 'created_at' in tasks_df.columns:
                st.write(f"**Date Range:** {tasks_df['created_at'].min().date()} to {tasks_df['created_at'].max().date()}")
                st.write(f"**Timespan:** {(tasks_df['created_at'].max() - tasks_df['created_at'].min()).days} days")
        
        with col2:
            st.subheader("Current View")
            st.write(f"**Date Filter:** {filters.get('date_filter', 'All Time')}")
            st.write(f"**Filtered Tasks:** {len(filtered_tasks):,}")
            
            if len(filtered_tasks) > 0:
                st.write(f"**Projects:** {filtered_tasks['project'].nunique() if 'project' in filtered_tasks.columns else 'N/A'}")
                st.write(f"**Team Members:** {filtered_tasks['assignee'].nunique() if 'assignee' in filtered_tasks.columns else 'N/A'}")
                
                if 'status' in filtered_tasks.columns:
                    completed = (filtered_tasks['status'] == 'Completed').sum()
                    rate = (completed / len(filtered_tasks) * 100)
                    st.write(f"**Completion Rate:** {rate:.1f}%")
        
        st.markdown("---")
        st.subheader("About This Dashboard")
        st.markdown("""
        **FlowFix AI** analyzes workflow bottlenecks and team performance.
        
        **Features:**
        - Upload your own CSV data (task_id, assignee, project, status, priority, created_at)
        - Interactive filtering by date, project, assignee, priority, status
        - Real-time bottleneck detection and severity analysis
        - Team performance metrics and workload distribution
        - Export comprehensive reports to PDF with bottleneck details and team stats
        
        **Demo Data:** 500 tasks spanning the last 100 days with realistic distributions.
        """)

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #94a3b8; padding: 2rem 0;'>
            <h3>FlowFix AI Enterprise Dashboard</h3>
            <p>Powered by ML + LLMs | Real-time Workflow Optimization</p>
            <p style='font-size: 0.9rem;'>© 2026 FlowFix AI. Built with Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()