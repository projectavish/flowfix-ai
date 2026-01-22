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

# Add project root and src to path (robust for Streamlit execution)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from src.utils import execute_query, get_summary_metrics

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

    * { font-family: 'Inter', sans-serif; }

    .stApp { background: #ffffff; }

    .main .block-container {
        background: white;
        border-radius: 8px;
        padding: 2rem;
        border: 1px solid #e5e5e5;
    }

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

    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: #000000 !important;
    }

    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #000000 !important;
    }

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

    section[data-testid="stSidebar"] { background: #000000; }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] * {
        color: white !important;
    }

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

    .stProgress > div > div > div > div { background: #000000; }

    .dataframe {
        border: 2px solid #000000;
        border-radius: 4px;
    }

    .dataframe th {
        background: #000000 !important;
        color: white !important;
    }

    ::-webkit-scrollbar { width: 8px; height: 8px; }

    ::-webkit-scrollbar-track { background: #f5f5f5; }

    ::-webkit-scrollbar-thumb {
        background: #000000;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover { background: #333333; }

    .js-plotly-plot {
        border: 2px solid #000000;
        border-radius: 4px;
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
# Data loaders
# -------------------------
@st.cache_data(ttl=300)
def load_tasks_data():
    """Load all tasks from database"""
    query = "SELECT * FROM tasks"
    try:
        return execute_query(query)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_bottlenecks_data():
    """Load bottleneck history"""
    query = """
    SELECT
        bh.bottleneck_id,
        bh.task_id,
        bh.bottleneck_type,
        bh.severity_score,
        bh.root_cause_suggestion,
        bh.detected_date,
        bh.resolution_date,
        t.task_name,
        t.assignee,
        t.project,
        t.status,
        t.priority
    FROM bottleneck_history bh
    JOIN tasks t ON bh.task_id = t.task_id
    ORDER BY bh.detected_date DESC
    """
    try:
        return execute_query(query)
    except Exception:
        return pd.DataFrame()


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
    try:
        return execute_query(query)
    except Exception:
        return pd.DataFrame()


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
    except Exception:
        return pd.DataFrame()


# -------------------------
# UI
# -------------------------
def render_sidebar(tasks_df: pd.DataFrame):
    """Render professional sidebar with filters and controls"""
    st.sidebar.markdown(
        '<h1 style="text-align: center;"><i class="fas fa-rocket"></i> FlowFix AI</h1>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        '<p style="text-align: center; opacity: 0.9;">Enterprise Workflow Optimizer</p>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown("---")

    # Quick Actions
    st.sidebar.markdown('<h3><i class="fas fa-bolt"></i> Quick Actions</h3>', unsafe_allow_html=True)
    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("🔄 Refresh", width="stretch"):
            st.cache_data.clear()
            st.rerun()

    with col2:
        if st.button("📊 Export", width="stretch"):
            st.info("Export feature coming soon!")

    st.sidebar.markdown("---")
    st.sidebar.markdown('<h3><i class="fas fa-filter"></i> Filters</h3>', unsafe_allow_html=True)

    # Date range filter
    st.sidebar.markdown(
        '<p style="margin-top: 1rem;"><i class="fas fa-calendar"></i> <strong>Date Range</strong></p>',
        unsafe_allow_html=True
    )
    date_filter = st.sidebar.radio(
        "Select Period",
        ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"],
        label_visibility="collapsed"
    )

    # Project filter
    projects = ["All Projects"] + _safe_unique_sorted(tasks_df, "project")
    selected_project = st.sidebar.selectbox("🏢 Project", projects)

    # Assignee filter
    assignees = ["All Assignees"] + _safe_unique_sorted(tasks_df, "assignee")
    selected_assignee = st.sidebar.selectbox("👤 Assignee", assignees)

    # Priority filter
    priorities = ["All Priorities", "Critical", "High", "Medium", "Low"]
    selected_priority = st.sidebar.selectbox("🎯 Priority", priorities)

    # Status filter
    statuses = ["All Statuses"] + _safe_unique_sorted(tasks_df, "status")
    selected_status = st.sidebar.selectbox("📌 Status", statuses)

    # Severity filter (only truly meaningful for bottleneck_history, but safe elsewhere)
    st.sidebar.markdown(
        '<p style="margin-top: 1rem;"><i class="fas fa-fire"></i> <strong>Severity Threshold</strong></p>',
        unsafe_allow_html=True
    )
    severity_range = st.sidebar.slider(
        "Score Range",
        0, 100, (0, 100),
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown('<h3><i class="fas fa-chart-line"></i> Quick Stats</h3>', unsafe_allow_html=True)

    total_tasks = len(tasks_df) if tasks_df is not None else 0
    st.sidebar.metric("Total Tasks", total_tasks)

    if tasks_df is not None and not tasks_df.empty and "status" in tasks_df.columns:
        st.sidebar.metric("Active", int((tasks_df["status"] == "In Progress").sum()))
        st.sidebar.metric("Completed", int((tasks_df["status"] == "Completed").sum()))
    else:
        st.sidebar.metric("Active", 0)
        st.sidebar.metric("Completed", 0)

    return {
        "project": selected_project,
        "assignee": selected_assignee,
        "priority": selected_priority,
        "status": selected_status,
        "severity_range": severity_range,
        "date_filter": date_filter
    }


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

                st.plotly_chart(fig, width="stretch")

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
            st.plotly_chart(fig, width="stretch")


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
                st.plotly_chart(fig, width="stretch")

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
                st.plotly_chart(fig, width="stretch")

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
            st.plotly_chart(fig, width="stretch")

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
                st.plotly_chart(fig, width="stretch")
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

                st.dataframe(project_impact.head(10), width="stretch", height=350)
            else:
                st.info("Project impact not available.")

    st.markdown("---")
    st.subheader("📋 Detailed Bottleneck Records")

    display_cols = ["task_id", "task_name", "assignee", "project", "bottleneck_type", "severity_score", "priority", "status", "detected_date"]
    existing_cols = [c for c in display_cols if c in filtered.columns]

    if existing_cols:
        st.dataframe(filtered[existing_cols].head(100), width="stretch", height=400)
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
            st.plotly_chart(fig, width="stretch")

        if "urgency" in filtered.columns:
            urgency_counts = filtered["urgency"].astype(str).value_counts()
            fig = px.pie(values=urgency_counts.values, names=urgency_counts.index, title="Urgency Breakdown")
            st.plotly_chart(fig, width="stretch")

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
        if "assignee" in filtered.columns:
            assignee_tasks = filtered["assignee"].value_counts().head(15)
            fig = px.bar(
                x=assignee_tasks.values,
                y=assignee_tasks.index,
                orientation="h",
                title="Task Distribution by Assignee",
                labels={"x": "Number of Tasks", "y": "Assignee"}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, width="stretch")

    with colB:
        if "assignee" in filtered.columns and "status" in filtered.columns and "task_id" in filtered.columns:
            assignee_stats = filtered.groupby("assignee").agg(
                task_count=("task_id", "count"),
                completed=("status", lambda x: (x == "Completed").sum())
            )
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
            st.plotly_chart(fig, width="stretch")

    st.subheader("🏆 Performance Leaderboard")

    needed = {"assignee", "task_id", "status"}
    if needed.issubset(set(filtered.columns)):
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

        st.dataframe(leaderboard.head(20), width="stretch", height=400)
    else:
        st.info("Leaderboard not available due to missing columns.")


def render_project_insights(tasks_df, filters):
    """Project-level insights and analytics"""
    st.header("🏢 Project Insights")

    if tasks_df is None or tasks_df.empty:
        st.info("No tasks data available.")
        return

    filtered = apply_filters(tasks_df, filters)

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
            fig = px.bar(
                x=project_tasks.values,
                y=project_tasks.index,
                orientation="h",
                title="Tasks by Project",
                labels={"x": "Number of Tasks", "y": "Project"}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, width="stretch")

    with colB:
        if {"project", "status"}.issubset(set(filtered.columns)):
            project_status = filtered.groupby(["project", "status"]).size().reset_index(name="count")
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
            st.plotly_chart(fig, width="stretch")

    st.subheader("📊 Project Performance Summary")

    needed = {"project", "task_id", "status"}
    if needed.issubset(set(filtered.columns)):
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

        st.dataframe(project_stats, width="stretch", height=400)
    else:
        st.info("Project summary not available due to missing columns.")


# -------------------------
# Main
# -------------------------
def main():
    """Main application"""
    try:
        tasks_df = load_tasks_data()
        bottlenecks_df = load_bottlenecks_data()
        suggestions_df = load_gpt_suggestions()
        metrics = get_summary_metrics()

        filters = render_sidebar(tasks_df)

        render_executive_dashboard(tasks_df, bottlenecks_df, suggestions_df, metrics)

        st.markdown("---")

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

            colA, colB = st.columns(2)
            with colA:
                st.subheader("🔮 Predictive Insights")
                st.markdown("""
                - Risk prediction models
                - Workload forecasting
                - Capacity planning
                - Anomaly detection
                """)

            with colB:
                st.subheader("📊 Custom Reports")
                st.markdown("""
                - Executive summaries
                - Team reports
                - Project health scores
                - Trend analysis
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

    except Exception as e:
        st.error(f"Error loading dashboard: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
