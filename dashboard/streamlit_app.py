"""
FlowFix AI - Professional Enterprise Dashboard
Industry-grade bottleneck analysis and AI-powered workflow optimization
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from utils import execute_query, get_summary_metrics

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
        font-family: 'Inter', sans-serif;
    }
    
    /* Clean white background */
    .stApp {
        background: #ffffff;
    }
    
    /* Main content area */
    .main .block-container {
        background: white;
        border-radius: 8px;
        padding: 2rem;
        border: 1px solid #e5e5e5;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #000000 !important;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        text-align: center;
        color: #666666 !important;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    /* All text elements */
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: #000000 !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #000000 !important;
    }
    
    /* Metric cards */
    div[data-testid="stMetricValue"] {
        color: #000000;
        font-size: 2rem;
        font-weight: 700;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #333333;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    div[data-testid="stMetricDelta"] {
        color: #666666;
        font-size: 0.875rem;
    }
    
    div[data-testid="metric-container"] {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border: 2px solid #000000;
        transition: all 0.2s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 4px 4px 0 #000000;
    }
    
    /* Sidebar - black */
    section[data-testid="stSidebar"] {
        background: #000000;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Buttons */
    .stButton button {
        background: #000000;
        color: white;
        border: 2px solid #000000;
        border-radius: 4px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stButton button:hover {
        background: white;
        color: #000000;
        border: 2px solid #000000;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border: 2px solid #e5e5e5;
        border-radius: 4px;
        padding: 0.75rem 1.25rem;
        color: #333333;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        border-color: #000000;
        color: #000000;
    }
    
    .stTabs [aria-selected="true"] {
        background: #000000 !important;
        color: white !important;
        border-color: #000000 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: white;
        border: 2px solid #e5e5e5;
        border-radius: 4px;
        font-weight: 600;
        color: #000000;
        transition: all 0.2s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: #000000;
        color: white;
        border-color: #000000;
    }
    
    /* Severity badges */
    .severity-critical {
        color: white;
        font-weight: 600;
        padding: 0.25rem 0.75rem;
        background: #000000;
        border-radius: 4px;
        display: inline-block;
        font-size: 0.875rem;
    }
    
    .severity-high {
        color: #000000;
        font-weight: 600;
        padding: 0.25rem 0.75rem;
        background: white;
        border: 2px solid #000000;
        border-radius: 4px;
        display: inline-block;
        font-size: 0.875rem;
    }
    
    .severity-medium {
        color: #000000;
        font-weight: 600;
        padding: 0.25rem 0.75rem;
        background: #f5f5f5;
        border: 1px solid #000000;
        border-radius: 4px;
        display: inline-block;
        font-size: 0.875rem;
    }
    
    .severity-low {
        color: #666666;
        font-weight: 600;
        padding: 0.25rem 0.75rem;
        background: white;
        border: 1px solid #666666;
        border-radius: 4px;
        display: inline-block;
        font-size: 0.875rem;
    }
    
    /* Icon cards */
    .icon-card {
        background: #000000;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .icon-card i {
        font-size: 2rem;
        color: white;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: #000000;
    }
    
    /* Dataframe styling */
    .dataframe {
        border: 2px solid #000000;
        border-radius: 4px;
    }
    
    .dataframe th {
        background: #000000 !important;
        color: white !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f5f5f5;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #000000;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #333333;
    }
    
    /* Chart containers */
    .js-plotly-plot {
        border: 2px solid #000000;
        border-radius: 4px;
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
        bh.id,
        bh.task_id,
        bh.bottleneck_type,
        bh.severity_score,
        bh.detected_date,
        bh.resolution_date,
        bh.delay_days,
        bh.priority,
        bh.root_cause_suggestion,
        t.task_name,
        t.assignee,
        t.project,
        t.status
    FROM bottleneck_history bh
    JOIN tasks t ON bh.task_id = t.task_id
    ORDER BY bh.detected_date DESC
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
        t.project,
        t.bottleneck_type,
        t.priority
    FROM gpt_suggestions g
    JOIN tasks t ON g.task_id = t.task_id
    ORDER BY g.created_at DESC
    """
    return execute_query(query)


@st.cache_data(ttl=300)
def load_ml_predictions():
    """Load ML predictions"""
    query = """
    SELECT * FROM ml_predictions
    ORDER BY prediction_date DESC
    LIMIT 100
    """
    try:
        return execute_query(query)
    except:
        return pd.DataFrame()


def render_sidebar(tasks_df):
    """Render professional sidebar with filters and controls"""
    st.sidebar.markdown('<h1 style="text-align: center;"><i class="fas fa-rocket"></i> FlowFix AI</h1>', unsafe_allow_html=True)
    st.sidebar.markdown('<p style="text-align: center; opacity: 0.9;">Enterprise Workflow Optimizer</p>', unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    # Quick Actions
    st.sidebar.markdown('<h3><i class="fas fa-bolt"></i> Quick Actions</h3>', unsafe_allow_html=True)
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🔄 Refresh", width='stretch'):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        if st.button("📊 Export", width='stretch'):
            st.info("Export feature coming soon!")
    
    st.sidebar.markdown("---")
    
    # Advanced Filters
    st.sidebar.markdown('<h3><i class="fas fa-filter"></i> Filters</h3>', unsafe_allow_html=True)
    
    # Date range filter
    st.sidebar.markdown('<p style="margin-top: 1rem;"><i class="fas fa-calendar"></i> <strong>Date Range</strong></p>', unsafe_allow_html=True)
    date_filter = st.sidebar.radio(
        "Select Period",
        ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"],
        label_visibility="collapsed"
    )
    
    # Project filter
    projects = ['All Projects'] + sorted(tasks_df['project'].unique().tolist())
    selected_project = st.sidebar.selectbox("🏢 Project", projects)
    
    # Assignee filter
    assignees = ['All Assignees'] + sorted(tasks_df['assignee'].unique().tolist())
    selected_assignee = st.sidebar.selectbox("👤 Assignee", assignees)
    
    # Priority filter
    priorities = ['All Priorities', 'Critical', 'High', 'Medium', 'Low']
    selected_priority = st.sidebar.selectbox("🎯 Priority", priorities)
    
    # Status filter
    statuses = ['All Statuses'] + sorted(tasks_df['status'].unique().tolist())
    selected_status = st.sidebar.selectbox("📌 Status", statuses)
    
    # Severity filter
    st.sidebar.markdown('<p style="margin-top: 1rem;"><i class="fas fa-fire"></i> <strong>Severity Threshold</strong></p>', unsafe_allow_html=True)
    severity_range = st.sidebar.slider(
        "Score Range",
        0, 100, (0, 100),
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Statistics
    st.sidebar.markdown('<h3><i class="fas fa-chart-line"></i> Quick Stats</h3>', unsafe_allow_html=True)
    st.sidebar.metric("Total Tasks", len(tasks_df))
    st.sidebar.metric("Active", len(tasks_df[tasks_df['status'] == 'In Progress']))
    st.sidebar.metric("Completed", len(tasks_df[tasks_df['status'] == 'Completed']))
    
    return {
        'project': selected_project,
        'assignee': selected_assignee,
        'priority': selected_priority,
        'status': selected_status,
        'severity_range': severity_range,
        'date_filter': date_filter
    }


def apply_filters(df, filters):
    """Apply filters to dataframe"""
    filtered = df.copy()
    
    if filters['project'] != 'All Projects':
        filtered = filtered[filtered['project'] == filters['project']]
    
    if filters['assignee'] != 'All Assignees':
        filtered = filtered[filtered['assignee'] == filters['assignee']]
    
    if filters['priority'] != 'All Priorities':
        filtered = filtered[filtered['priority'] == filters['priority']]
    
    if filters['status'] != 'All Statuses':
        filtered = filtered[filtered['status'] == filters['status']]
    
    if 'severity_score' in filtered.columns:
        filtered = filtered[
            (filtered['severity_score'] >= filters['severity_range'][0]) &
            (filtered['severity_score'] <= filters['severity_range'][1])
        ]
    
    return filtered


def render_executive_dashboard(tasks_df, bottlenecks_df, suggestions_df, metrics):
    """Render executive overview with key metrics"""
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
    
    # Executive KPI Row with icons
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        total_tasks = metrics.get('total_tasks', 0)
        completed = metrics.get('completed_tasks', 0)
        completion_rate = (completed / total_tasks * 100) if total_tasks > 0 else 0
        st.markdown('<div class="icon-card"><i class="fas fa-tasks"></i></div>', unsafe_allow_html=True)
        st.metric(
            "Total Tasks",
            f"{total_tasks:,}",
            f"{completion_rate:.1f}% complete"
        )
    
    with col2:
        active_tasks = len(tasks_df[tasks_df['status'] == 'In Progress'])
        st.markdown('<div class="icon-card"><i class="fas fa-bolt"></i></div>', unsafe_allow_html=True)
        st.metric(
            "Active Tasks",
            f"{active_tasks:,}",
            f"{active_tasks/total_tasks*100:.1f}%" if total_tasks > 0 else "0%"
        )
    
    with col3:
        bottleneck_count = len(bottlenecks_df)
        bottleneck_rate = metrics.get('bottleneck_rate', 0)
        st.markdown('<div class="icon-card"><i class="fas fa-exclamation-triangle"></i></div>', unsafe_allow_html=True)
        st.metric(
            "Bottlenecks",
            f"{bottleneck_count:,}",
            f"{bottleneck_rate:.1f}%",
            delta_color="inverse"
        )
    
    with col4:
        delay_rate = metrics.get('delay_rate', 0)
        delayed_tasks = metrics.get('delayed_tasks', 0)
        st.markdown('<div class="icon-card"><i class="fas fa-clock"></i></div>', unsafe_allow_html=True)
        st.metric(
            "Delays",
            f"{delayed_tasks:,}",
            f"{delay_rate:.1f}%",
            delta_color="inverse"
        )
    
    with col5:
        avg_duration = metrics.get('avg_duration', 0)
        st.markdown('<div class="icon-card"><i class="fas fa-chart-bar"></i></div>', unsafe_allow_html=True)
        st.metric(
            "Avg Duration",
            f"{avg_duration:.1f}d",
            "Target: 5d"
        )
    
    with col6:
        ai_suggestions = len(suggestions_df)
        st.markdown('<div class="icon-card"><i class="fas fa-robot"></i></div>', unsafe_allow_html=True)
        st.metric(
            "AI Insights",
            f"{ai_suggestions:,}",
            f"+{ai_suggestions} new"
        )
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Bottleneck Trends
        if len(bottlenecks_df) > 0 and 'detected_date' in bottlenecks_df.columns:
            bottlenecks_df['detected_date'] = pd.to_datetime(bottlenecks_df['detected_date'])
            daily_data = bottlenecks_df.groupby(bottlenecks_df['detected_date'].dt.date).agg({
                'id': 'count',
                'severity_score': 'mean'
            }).reset_index()
            daily_data.columns = ['Date', 'Count', 'Avg_Severity']
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Bar(x=daily_data['Date'], y=daily_data['Count'], name='Bottlenecks', marker_color='#667eea'),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(x=daily_data['Date'], y=daily_data['Avg_Severity'], name='Avg Severity', 
                          line=dict(color='#f59e0b', width=3), mode='lines+markers'),
                secondary_y=True
            )
            
            fig.update_layout(
                title='📈 Bottleneck Detection Trends',
                hovermode='x unified',
                height=400,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="Bottleneck Count", secondary_y=False)
            fig.update_yaxes(title_text="Severity Score", secondary_y=True)
            
            st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Priority Distribution
        if len(bottlenecks_df) > 0:
            priority_counts = bottlenecks_df['priority'].value_counts()
            
            colors = {
                'Critical': '#dc2626',
                'High': '#ea580c',
                'Medium': '#f59e0b',
                'Low': '#10b981'
            }
            
            fig = go.Figure(data=[go.Pie(
                labels=priority_counts.index,
                values=priority_counts.values,
                hole=0.4,
                marker=dict(colors=[colors.get(p, '#667eea') for p in priority_counts.index]),
                textinfo='label+percent',
                textposition='outside'
            )])
            
            fig.update_layout(
                title='🎯 Priority Distribution',
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, width='stretch')


def render_bottleneck_analysis(bottlenecks_df, filters):
    """Comprehensive bottleneck analysis"""
    st.header("🔍 Bottleneck Deep Dive")
    
    if len(bottlenecks_df) == 0:
        st.success("✅ No bottlenecks detected! Your workflow is running smoothly.")
        return
    
    filtered = apply_filters(bottlenecks_df, filters)
    
    if len(filtered) == 0:
        st.info("No bottlenecks match your filters")
        return
    
    # Analysis Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔬 Root Cause", "🎯 Impact"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Critical Bottlenecks", len(filtered[filtered['severity_score'] > 70]))
        
        with col2:
            st.metric("Avg Severity", f"{filtered['severity_score'].mean():.1f}/100")
        
        with col3:
            resolved = filtered['resolution_date'].notna().sum()
            st.metric("Resolved", f"{resolved} ({resolved/len(filtered)*100:.1f}%)")
        
        # Bottleneck Type Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            type_counts = filtered['bottleneck_type'].value_counts().head(10)
            fig = px.bar(
                x=type_counts.values,
                y=type_counts.index,
                orientation='h',
                title='Top Bottleneck Types',
                labels={'x': 'Count', 'y': 'Type'},
                color=type_counts.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            assignee_counts = filtered['assignee'].value_counts().head(10)
            fig = px.bar(
                x=assignee_counts.values,
                y=assignee_counts.index,
                orientation='h',
                title='Bottlenecks by Assignee',
                labels={'x': 'Count', 'y': 'Assignee'},
                color=assignee_counts.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, width='stretch')
        
        # Severity Heatmap
        if 'project' in filtered.columns and 'bottleneck_type' in filtered.columns:
            heatmap_data = filtered.pivot_table(
                values='severity_score',
                index='project',
                columns='bottleneck_type',
                aggfunc='mean'
            ).fillna(0)
            
            fig = px.imshow(
                heatmap_data,
                title='Severity Heatmap: Project vs Bottleneck Type',
                labels=dict(x="Bottleneck Type", y="Project", color="Avg Severity"),
                color_continuous_scale='RdYlGn_r',
                aspect='auto'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
    
    with tab2:
        st.subheader("🔬 Root Cause Analysis")
        
        # Extract root causes
        root_causes = filtered['root_cause_suggestion'].dropna()
        
        if len(root_causes) > 0:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Most Common Root Causes:**")
                for i, cause in enumerate(root_causes.value_counts().head(10).items(), 1):
                    st.markdown(f"{i}. **{cause[0]}** ({cause[1]} occurrences)")
            
            with col2:
                st.markdown("**Quick Insights:**")
                st.markdown(f"- {len(root_causes.unique())} unique root causes identified")
                st.markdown(f"- {len(root_causes)} total diagnoses")
                st.markdown(f"- Top cause: **{root_causes.value_counts().index[0]}**")
        else:
            st.info("No root cause data available")
    
    with tab3:
        st.subheader("🎯 Business Impact Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Delay impact
            if 'delay_days' in filtered.columns:
                total_delay = filtered['delay_days'].sum()
                avg_delay = filtered['delay_days'].mean()
                
                st.metric("Total Delay Days", f"{total_delay:.0f}")
                st.metric("Avg Delay per Bottleneck", f"{avg_delay:.1f} days")
                
                fig = px.histogram(
                    filtered,
                    x='delay_days',
                    nbins=20,
                    title='Delay Distribution',
                    labels={'delay_days': 'Delay (days)', 'count': 'Frequency'}
                )
                st.plotly_chart(fig, width='stretch')
        
        with col2:
            # Project impact
            if 'project' in filtered.columns:
                project_impact = filtered.groupby('project').agg({
                    'id': 'count',
                    'severity_score': 'mean',
                    'delay_days': 'sum'
                }).round(2)
                project_impact.columns = ['Bottlenecks', 'Avg Severity', 'Total Delay']
                project_impact = project_impact.sort_values('Bottlenecks', ascending=False)
                
                st.dataframe(
                    project_impact.head(10),
                    width='stretch',
                    height=350
                )
    
    # Detailed Table
    st.markdown("---")
    st.subheader("📋 Detailed Bottleneck Records")
    
    display_cols = ['task_id', 'task_name', 'assignee', 'project', 'bottleneck_type', 
                    'severity_score', 'priority', 'status', 'detected_date']
    
    st.dataframe(
        filtered[display_cols].head(100),
        width='stretch',
        height=400
    )


def render_ai_recommendations(suggestions_df, filters):
    """Display AI-powered recommendations"""
    st.header("🤖 AI-Powered Recommendations")
    
    if len(suggestions_df) == 0:
        st.info("No AI suggestions generated yet. Run the GPT suggester to generate recommendations.")
        return
    
    filtered = apply_filters(suggestions_df, filters)
    
    if len(filtered) == 0:
        st.info("No suggestions match your filters")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Suggestions", len(filtered))
    
    with col2:
        high_quality = len(filtered[filtered['quality_score'] >= 80])
        st.metric("High Quality", high_quality)
    
    with col3:
        applied = filtered['applied'].sum() if 'applied' in filtered.columns else 0
        st.metric("Applied", applied)
    
    with col4:
        avg_quality = filtered['quality_score'].mean() if 'quality_score' in filtered.columns else 0
        st.metric("Avg Quality", f"{avg_quality:.0f}/100")
    
    st.markdown("---")
    
    # Display suggestions
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Recent Suggestions")
        
        for idx, row in filtered.head(10).iterrows():
            with st.expander(f"🎯 {row['task_name'][:60]}... - Quality: {row.get('quality_score', 0)}/100"):
                st.markdown(f"**Task ID:** {row['task_id']}")
                st.markdown(f"**Assignee:** {row['assignee']} | **Project:** {row['project']}")
                st.markdown(f"**Priority:** {row.get('priority', 'N/A')} | **Urgency:** {row.get('urgency', 'N/A')}")
                
                st.markdown("**💡 AI Suggestion:**")
                st.markdown(f"{row['suggestion_text'][:500]}...")
                
                if 'root_causes' in row and pd.notna(row['root_causes']):
                    st.markdown("**🔍 Root Causes:**")
                    st.markdown(row['root_causes'][:300])
                
                if 'recommendations' in row and pd.notna(row['recommendations']):
                    st.markdown("**✅ Recommendations:**")
                    st.markdown(row['recommendations'][:300])
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if st.button("👍 Apply", key=f"apply_{row['task_id']}"):
                        st.success("Marked as applied!")
                with col_b:
                    if st.button("👎 Dismiss", key=f"dismiss_{row['task_id']}"):
                        st.info("Suggestion dismissed")
                with col_c:
                    st.button("📋 Copy", key=f"copy_{row['task_id']}")
    
    with col2:
        st.subheader("📊 Suggestion Analytics")
        
        # Quality distribution
        if 'quality_score' in filtered.columns:
            fig = px.histogram(
                filtered,
                x='quality_score',
                nbins=10,
                title='Quality Score Distribution',
                labels={'quality_score': 'Quality Score', 'count': 'Count'}
            )
            st.plotly_chart(fig, width='stretch')
        
        # Urgency breakdown
        if 'urgency' in filtered.columns:
            urgency_counts = filtered['urgency'].value_counts()
            fig = px.pie(
                values=urgency_counts.values,
                names=urgency_counts.index,
                title='Urgency Breakdown'
            )
            st.plotly_chart(fig, width='stretch')
        
        # Model performance
        if 'model_used' in filtered.columns:
            st.markdown("**🎯 Model Stats:**")
            st.markdown(f"- Model: {filtered['model_used'].iloc[0]}")
            st.markdown(f"- Avg Quality: {filtered['quality_score'].mean():.1f}/100")
            st.markdown(f"- Total Generated: {len(filtered)}")


def render_team_performance(tasks_df, filters):
    """Team and assignee performance analysis"""
    st.header("👥 Team Performance")
    
    filtered = apply_filters(tasks_df, filters)
    
    # Team metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_assignees = filtered['assignee'].nunique()
        st.metric("Team Members", total_assignees)
    
    with col2:
        avg_tasks = len(filtered) / total_assignees if total_assignees > 0 else 0
        st.metric("Avg Tasks/Person", f"{avg_tasks:.1f}")
    
    with col3:
        completed = len(filtered[filtered['status'] == 'Completed'])
        st.metric("Completed", completed)
    
    with col4:
        in_progress = len(filtered[filtered['status'] == 'In Progress'])
        st.metric("In Progress", in_progress)
    
    st.markdown("---")
    
    # Performance analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Task distribution
        assignee_tasks = filtered['assignee'].value_counts().head(15)
        fig = px.bar(
            x=assignee_tasks.values,
            y=assignee_tasks.index,
            orientation='h',
            title='Task Distribution by Assignee',
            labels={'x': 'Number of Tasks', 'y': 'Assignee'},
            color=assignee_tasks.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Completion rate by assignee
        assignee_stats = filtered.groupby('assignee').agg({
            'task_id': 'count',
            'status': lambda x: (x == 'Completed').sum()
        })
        assignee_stats['completion_rate'] = (assignee_stats['status'] / assignee_stats['task_id'] * 100).round(1)
        assignee_stats = assignee_stats.sort_values('completion_rate', ascending=False).head(15)
        
        fig = px.bar(
            x=assignee_stats['completion_rate'],
            y=assignee_stats.index,
            orientation='h',
            title='Completion Rate by Assignee (%)',
            labels={'x': 'Completion Rate (%)', 'y': 'Assignee'},
            color=assignee_stats['completion_rate'],
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, width='stretch')
    
    # Performance leaderboard
    st.subheader("🏆 Performance Leaderboard")
    
    leaderboard = filtered.groupby('assignee').agg({
        'task_id': 'count',
        'status': lambda x: (x == 'Completed').sum(),
        'actual_duration': 'mean',
        'priority': lambda x: (x == 'High').sum()
    }).round(2)
    
    leaderboard.columns = ['Total Tasks', 'Completed', 'Avg Duration', 'High Priority']
    leaderboard['Completion %'] = (leaderboard['Completed'] / leaderboard['Total Tasks'] * 100).round(1)
    leaderboard = leaderboard.sort_values('Completed', ascending=False)
    
    st.dataframe(
        leaderboard.head(20),
        width='stretch',
        height=400
    )


def render_project_insights(tasks_df, filters):
    """Project-level insights and analytics"""
    st.header("🏢 Project Insights")
    
    filtered = apply_filters(tasks_df, filters)
    
    # Project overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_projects = filtered['project'].nunique()
        st.metric("Active Projects", total_projects)
    
    with col2:
        avg_tasks = len(filtered) / total_projects if total_projects > 0 else 0
        st.metric("Avg Tasks/Project", f"{avg_tasks:.1f}")
    
    with col3:
        if 'actual_duration' in filtered.columns:
            avg_duration = filtered['actual_duration'].mean()
            st.metric("Avg Duration", f"{avg_duration:.1f}d")
    
    with col4:
        high_priority = len(filtered[filtered['priority'].isin(['High', 'Critical'])])
        st.metric("High Priority", high_priority)
    
    st.markdown("---")
    
    # Project visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Project size
        project_tasks = filtered['project'].value_counts().head(15)
        fig = px.bar(
            x=project_tasks.values,
            y=project_tasks.index,
            orientation='h',
            title='Tasks by Project',
            labels={'x': 'Number of Tasks', 'y': 'Project'},
            color=project_tasks.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Project status breakdown
        project_status = filtered.groupby(['project', 'status']).size().reset_index(name='count')
        fig = px.bar(
            project_status,
            x='project',
            y='count',
            color='status',
            title='Status Breakdown by Project',
            labels={'count': 'Number of Tasks', 'project': 'Project'},
            barmode='stack'
        )
        fig.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig, width='stretch')
    
    # Project performance table
    st.subheader("📊 Project Performance Summary")
    
    project_stats = filtered.groupby('project').agg({
        'task_id': 'count',
        'status': lambda x: (x == 'Completed').sum(),
        'actual_duration': 'mean',
        'priority': lambda x: (x.isin(['High', 'Critical'])).sum()
    }).round(2)
    
    project_stats.columns = ['Total Tasks', 'Completed', 'Avg Duration', 'High Priority']
    project_stats['Completion %'] = (project_stats['Completed'] / project_stats['Total Tasks'] * 100).round(1)
    project_stats = project_stats.sort_values('Total Tasks', ascending=False)
    
    st.dataframe(
        project_stats,
        width='stretch',
        height=400
    )


def main():
    """Main application"""
    try:
        # Load data
        tasks_df = load_tasks_data()
        bottlenecks_df = load_bottlenecks_data()
        suggestions_df = load_gpt_suggestions()
        metrics = get_summary_metrics()
        
        # Render sidebar and get filters
        filters = render_sidebar(tasks_df)
        
        # Main dashboard
        render_executive_dashboard(tasks_df, bottlenecks_df, suggestions_df, metrics)
        
        st.markdown("---")
        
        # Detailed Analysis Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🚨 Bottleneck Analysis",
            "🤖 AI Recommendations",
            "👥 Team Performance",
            "🏢 Project Insights",
            "📈 Advanced Analytics"
        ])
        
        with tab1:
            render_bottleneck_analysis(bottlenecks_df, filters)
        
        with tab2:
            render_ai_recommendations(suggestions_df, filters)
        
        with tab3:
            render_team_performance(tasks_df, filters)
        
        with tab4:
            render_project_insights(tasks_df, filters)
        
        with tab5:
            st.header("📈 Advanced Analytics")
            st.info("🚧 Advanced predictive analytics and forecasting coming soon!")
            
            # Placeholder for ML insights
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔮 Predictive Insights")
                st.markdown("""
                - Risk prediction models
                - Workload forecasting
                - Capacity planning
                - Anomaly detection
                """)
            
            with col2:
                st.subheader("📊 Custom Reports")
                st.markdown("""
                - Executive summaries
                - Team reports
                - Project health scores
                - Trend analysis
                """)
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #94a3b8; padding: 2rem 0;'>
                <h3>FlowFix AI Enterprise Dashboard</h3>
                <p>Powered by Advanced ML & GPT-4 | Real-time Workflow Optimization</p>
                <p style='font-size: 0.9rem;'>© 2026 FlowFix AI. Built with ❤️ using Streamlit</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"Error loading dashboard: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
