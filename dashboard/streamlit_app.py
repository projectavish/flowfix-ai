"""
FlowFix AI - Interactive Streamlit Dashboard
Real-time bottleneck exploration, filters, and visualizations
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.utils import execute_query, get_summary_metrics
from src.bottleneck_detector import run_bottleneck_detection

# Page config
st.set_page_config(
    page_title="FlowFix AI Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .severity-high {
        color: #d62728;
        font-weight: bold;
    }
    .severity-medium {
        color: #ff7f0e;
        font-weight: bold;
    }
    .severity-low {
        color: #2ca02c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def load_tasks_data():
    """Load all tasks from database"""
    query = "SELECT * FROM tasks"
    return execute_query(query)


@st.cache_data(ttl=300)
def load_bottlenecks_data():
    """Load bottleneck history"""
    query = """
    SELECT 
        bh.*,
        t.task_name,
        t.assignee,
        t.project,
        t.status
    FROM bottleneck_history bh
    JOIN tasks t ON bh.task_id = t.task_id
    ORDER BY bh.detected_at DESC
    """
    return execute_query(query)


@st.cache_data(ttl=300)
def load_gpt_suggestions():
    """Load GPT suggestions"""
    query = """
    SELECT 
        g.*,
        t.task_name,
        t.assignee,
        t.project
    FROM gpt_suggestions g
    JOIN tasks t ON g.task_id = t.task_id
    ORDER BY g.created_at DESC
    """
    return execute_query(query)


def render_sidebar():
    """Render sidebar with filters"""
    st.sidebar.title("🔧 FlowFix AI")
    st.sidebar.markdown("---")
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Run bottleneck detection
    st.sidebar.subheader("⚙️ Actions")
    if st.sidebar.button("🔍 Run Bottleneck Detection", use_container_width=True):
        with st.spinner("Running detection..."):
            run_bottleneck_detection()
            st.cache_data.clear()
            st.success("Detection complete!")
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Filters
    st.sidebar.subheader("🎯 Filters")
    
    tasks_df = load_tasks_data()
    
    # Project filter
    projects = ['All'] + sorted(tasks_df['project'].unique().tolist())
    selected_project = st.sidebar.selectbox("Project", projects)
    
    # Assignee filter
    assignees = ['All'] + sorted(tasks_df['assignee'].unique().tolist())
    selected_assignee = st.sidebar.selectbox("Assignee", assignees)
    
    # Priority filter
    priorities = ['All'] + ['Critical', 'High', 'Medium', 'Low']
    selected_priority = st.sidebar.selectbox("Priority", priorities)
    
    # Status filter
    statuses = ['All'] + sorted(tasks_df['status'].unique().tolist())
    selected_status = st.sidebar.selectbox("Status", statuses)
    
    # Severity filter
    severity_range = st.sidebar.slider(
        "Severity Score Range",
        0, 100, (0, 100)
    )
    
    return {
        'project': selected_project,
        'assignee': selected_assignee,
        'priority': selected_priority,
        'status': selected_status,
        'severity_range': severity_range
    }


def apply_filters(df, filters):
    """Apply filters to dataframe"""
    filtered = df.copy()
    
    if filters['project'] != 'All':
        filtered = filtered[filtered['project'] == filters['project']]
    
    if filters['assignee'] != 'All':
        filtered = filtered[filtered['assignee'] == filters['assignee']]
    
    if filters['priority'] != 'All':
        filtered = filtered[filtered['priority'] == filters['priority']]
    
    if filters['status'] != 'All':
        filtered = filtered[filtered['status'] == filters['status']]
    
    if 'severity_score' in filtered.columns:
        filtered = filtered[
            (filtered['severity_score'] >= filters['severity_range'][0]) &
            (filtered['severity_score'] <= filters['severity_range'][1])
        ]
    
    return filtered


def render_kpi_cards(metrics):
    """Render KPI metric cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📋 Total Tasks",
            value=metrics.get('total_tasks', 0),
            delta=f"{metrics.get('completed_tasks', 0)} completed"
        )
    
    with col2:
        delay_rate = metrics.get('delay_rate', 0)
        st.metric(
            label="⏰ Delay Rate",
            value=f"{delay_rate:.1f}%",
            delta=f"{metrics.get('delayed_tasks', 0)} delayed",
            delta_color="inverse"
        )
    
    with col3:
        bottleneck_rate = metrics.get('bottleneck_rate', 0)
        st.metric(
            label="🚨 Bottleneck Rate",
            value=f"{bottleneck_rate:.1f}%",
            delta=f"{metrics.get('bottleneck_count', 0)} bottlenecks",
            delta_color="inverse"
        )
    
    with col4:
        avg_duration = metrics.get('avg_duration', 0)
        st.metric(
            label="📊 Avg Duration",
            value=f"{avg_duration:.1f} days",
            delta=f"{metrics.get('total_projects', 0)} projects"
        )


def render_bottleneck_overview(bottlenecks_df, filters):
    """Render bottleneck overview section"""
    st.header("🔍 Bottleneck Overview")
    
    if len(bottlenecks_df) == 0:
        st.success("✅ No bottlenecks detected!")
        return
    
    filtered_bottlenecks = apply_filters(bottlenecks_df, filters)
    
    if len(filtered_bottlenecks) == 0:
        st.info("No bottlenecks match the current filters")
        return
    
    # Summary stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Bottlenecks", len(filtered_bottlenecks))
    
    with col2:
        avg_severity = filtered_bottlenecks['severity_score'].mean()
        st.metric("Avg Severity", f"{avg_severity:.1f}/100")
    
    with col3:
        high_severity = len(filtered_bottlenecks[filtered_bottlenecks['severity_score'] > 70])
        st.metric("High Severity (>70)", high_severity)
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Bottleneck types
        st.subheader("By Type")
        type_counts = filtered_bottlenecks['bottleneck_type'].value_counts()
        fig = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="Bottleneck Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Severity distribution
        st.subheader("Severity Distribution")
        fig = px.histogram(
            filtered_bottlenecks,
            x='severity_score',
            nbins=20,
            title="Severity Score Distribution",
            labels={'severity_score': 'Severity Score'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Bottleneck table
    st.subheader("📋 Bottleneck Details")
    
    display_cols = ['task_id', 'task_name', 'assignee', 'project', 'bottleneck_type', 
                   'severity_score', 'delay_days', 'priority', 'status']
    
    # Color code severity
    def color_severity(val):
        if val > 70:
            return 'background-color: #ffcccc'
        elif val > 40:
            return 'background-color: #ffffcc'
        else:
            return 'background-color: #ccffcc'
    
    styled_df = filtered_bottlenecks[display_cols].style.applymap(
        color_severity,
        subset=['severity_score']
    )
    
    st.dataframe(styled_df, use_container_width=True, height=400)


def render_assignee_analysis(tasks_df, filters):
    """Render assignee workload analysis"""
    st.header("👥 Assignee Analysis")
    
    filtered_tasks = apply_filters(tasks_df, filters)
    
    # Workload by assignee
    workload = filtered_tasks.groupby('assignee').agg({
        'task_id': 'count',
        'is_delayed': 'sum',
        'actual_duration': 'mean',
        'bottleneck_type': lambda x: (x != '').sum()
    }).reset_index()
    
    workload.columns = ['Assignee', 'Total Tasks', 'Delayed Tasks', 'Avg Duration', 'Bottlenecks']
    workload['Delay Rate %'] = (workload['Delayed Tasks'] / workload['Total Tasks'] * 100).round(1)
    
    # Sort by bottlenecks
    workload = workload.sort_values('Bottlenecks', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart - workload
        fig = px.bar(
            workload.head(10),
            x='Assignee',
            y='Total Tasks',
            color='Bottlenecks',
            title="Top 10 Assignees by Workload",
            labels={'Total Tasks': 'Number of Tasks'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Scatter - delay rate vs workload
        fig = px.scatter(
            workload,
            x='Total Tasks',
            y='Delay Rate %',
            size='Bottlenecks',
            hover_data=['Assignee'],
            title="Delay Rate vs Workload",
            labels={'Total Tasks': 'Task Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Workload table
    st.dataframe(workload, use_container_width=True)


def render_project_insights(tasks_df, filters):
    """Render project-level insights"""
    st.header("📊 Project Insights")
    
    filtered_tasks = apply_filters(tasks_df, filters)
    
    # Project metrics
    project_metrics = filtered_tasks.groupby('project').agg({
        'task_id': 'count',
        'is_delayed': 'sum',
        'actual_duration': 'mean',
        'bottleneck_type': lambda x: (x != '').sum()
    }).reset_index()
    
    project_metrics.columns = ['Project', 'Total Tasks', 'Delayed', 'Avg Duration', 'Bottlenecks']
    project_metrics['Health Score'] = (
        100 - (project_metrics['Delayed'] / project_metrics['Total Tasks'] * 50) -
        (project_metrics['Bottlenecks'] / project_metrics['Total Tasks'] * 50)
    ).round(1)
    
    # Timeline view
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Project health
        fig = px.bar(
            project_metrics,
            x='Project',
            y='Health Score',
            color='Health Score',
            title="Project Health Scores",
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Project summary
        st.metric("Total Projects", len(project_metrics))
        st.metric("Avg Tasks/Project", f"{project_metrics['Total Tasks'].mean():.0f}")
        healthy_projects = len(project_metrics[project_metrics['Health Score'] > 70])
        st.metric("Healthy Projects (>70)", healthy_projects)
    
    # Project table
    st.dataframe(project_metrics.sort_values('Health Score'), use_container_width=True)


def render_gpt_suggestions(suggestions_df, filters):
    """Render GPT suggestions"""
    st.header("🤖 AI Suggestions")
    
    if len(suggestions_df) == 0:
        st.info("No GPT suggestions generated yet")
        return
    
    filtered_suggestions = apply_filters(suggestions_df, filters)
    
    if len(filtered_suggestions) == 0:
        st.info("No suggestions match the current filters")
        return
    
    # Stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Suggestions", len(filtered_suggestions))
    
    with col2:
        applied = len(filtered_suggestions[filtered_suggestions.get('feedback_status', '') == 'applied'])
        st.metric("Applied", applied)
    
    with col3:
        pending = len(filtered_suggestions[filtered_suggestions.get('feedback_status', '') == 'pending'])
        st.metric("Pending", pending)
    
    # Show recent suggestions
    st.subheader("Recent Suggestions")
    
    for idx, row in filtered_suggestions.head(10).iterrows():
        with st.expander(f"📝 {row['task_name']} - {row['assignee']}"):
            st.markdown(f"**Task ID:** {row['task_id']}")
            st.markdown(f"**Project:** {row['project']}")
            st.markdown(f"**Generated:** {row['created_at']}")
            st.markdown("---")
            st.markdown(row.get('suggestion_text', 'No suggestion text available'))


def main():
    """Main dashboard function"""
    # Title
    st.markdown('<div class="main-header">🔧 FlowFix AI Dashboard</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar filters
    filters = render_sidebar()
    
    # Load data
    try:
        metrics = get_summary_metrics()
        tasks_df = load_tasks_data()
        bottlenecks_df = load_bottlenecks_data()
        suggestions_df = load_gpt_suggestions()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Make sure the database has been initialized and populated")
        return
    
    # KPI Cards
    render_kpi_cards(metrics)
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Bottlenecks",
        "👥 Assignees",
        "📊 Projects",
        "🤖 AI Suggestions",
        "📈 Trends"
    ])
    
    with tab1:
        render_bottleneck_overview(bottlenecks_df, filters)
    
    with tab2:
        render_assignee_analysis(tasks_df, filters)
    
    with tab3:
        render_project_insights(tasks_df, filters)
    
    with tab4:
        render_gpt_suggestions(suggestions_df, filters)
    
    with tab5:
        st.header("📈 Trends Over Time")
        st.info("Coming soon: Historical trend analysis and forecasting")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "FlowFix AI Dashboard | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
