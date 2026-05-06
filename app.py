"""
app.py - Main Streamlit UI for the AI Data Analyst System.

Tab layout:
  📊 Overview      → dataset summary + quick stats
  📈 Visualizations → auto-generated charts
  🤖 AI Insights   → LLM-generated findings + trend/anomaly explanations
  💬 Chat          → natural language Q&A about the dataset
  📄 Report        → downloadable markdown report
"""

import json
from io import BytesIO

import pandas as pd
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER

from agent import DataAnalystAgent
from utils import load_dataframe, validate_file


# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ---------- Global ---------- */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ---------- Header ---------- */
.app-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border-bottom: 1px solid #30363d;
    padding: 1.5rem 2rem;
    margin: -1rem -1rem 1.5rem -1rem;
}
.app-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: #58a6ff;
    letter-spacing: -0.5px;
    margin: 0;
}
.app-subtitle {
    font-size: 0.85rem;
    color: #c9d1d9;
    margin: 0.25rem 0 0 0;
}

/* ---------- Metric cards ---------- */
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-card h3 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #58a6ff;
    margin: 0 0 0.2rem 0;
}
.metric-card p {
    font-size: 0.78rem;
    color: #c9d1d9;
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ---------- Chat ---------- */
.chat-user {
    background: #1c2128;
    border-left: 3px solid #58a6ff;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin-bottom: 0.75rem;
    font-size: 0.92rem;
    color: #e6edf3;
}
.chat-assistant {
    background: #161b22;
    border-left: 3px solid #3fb950;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin-bottom: 0.75rem;
    font-size: 0.92rem;
    color: #e6edf3;
}
.code-block {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: #e6edf3;
    margin-top: 0.5rem;
    overflow-x: auto;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}

/* FIX 2: Text visibility - Improve sidebar text color */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: #c9d1d9 !important;
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    border-bottom: 1px solid #30363d;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #c9d1d9;
}
.stTabs [aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
}

/* FIX 2: Text visibility - Improve tab text color for non-active tabs */
.stTabs [data-baseweb="tab"] > div {
    color: #c9d1d9 !important;
}

/* ---------- Insight sections ---------- */
.insight-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    color: #e6edf3;
}

/* FIX 2: Text visibility - Improve general text color */
.insight-box p,
.insight-box span,
.insight-box h1,
.insight-box h2,
.insight-box h3,
.insight-box h4,
.insight-box h5,
.insight-box h6 {
    color: #e6edf3 !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Session State Init ───────────────────────────────────────────────────────

def init_state():
    defaults = {
        "agent":     DataAnalystAgent(),
        "df":        None,
        "analysis":  None,
        "insights":  None,
        "charts":    None,
        "report":    None,
        "chat_msgs": [],
        "file_name": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ─── FIX 3: PDF Export Helper ─────────────────────────────────────────────────

def markdown_to_pdf(markdown_text: str, filename: str = "report.pdf") -> BytesIO:
    """
    Convert plain markdown/text report to PDF using ReportLab.
    
    Args:
        markdown_text: The markdown report content
        filename: Output filename (for reference)
    
    Returns:
        BytesIO object containing PDF data
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor='#58a6ff',
        spaceAfter=12,
        alignment=TA_CENTER,
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor='#58a6ff',
        spaceAfter=8,
        spaceBefore=8,
    )
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        textColor='#1a1a1a',
        spaceAfter=6,
    )
    
    # Parse markdown into lines and add to story
    lines = markdown_text.split('\n')
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines but add spacing
        if not stripped:
            story.append(Spacer(1, 0.1*inch))
            continue
        
        # Detect headers by # markdown syntax
        if stripped.startswith('# '):
            title = stripped[2:].strip()
            story.append(Paragraph(title, title_style))
            story.append(Spacer(1, 0.1*inch))
        elif stripped.startswith('## '):
            heading = stripped[3:].strip()
            story.append(Paragraph(heading, heading_style))
        elif stripped.startswith('### '):
            heading = stripped[4:].strip()
            story.append(Paragraph(heading, heading_style))
        else:
            # Regular body text
            # Escape special XML characters for ReportLab
            safe_text = stripped.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(safe_text, body_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer


# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
  <p class="app-title">📊 AI Data Analyst</p>
  <p class="app-subtitle">
    Powered by Meta Llama 3.3-70b · Upload CSV or Excel → instant insights
  </p>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar – File Upload ────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📂 Upload Dataset")
    uploaded = st.file_uploader(
        "CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Max 50 MB",
    )

    if uploaded:
        # Only reload if it's a new file
        if uploaded.name != st.session_state.file_name:
            valid, err = validate_file(uploaded)
            if not valid:
                st.error(err)
            else:
                with st.spinner("Loading dataset…"):
                    df, err = load_dataframe(uploaded)
                if err:
                    st.error(err)
                else:
                    st.session_state.df        = df
                    st.session_state.file_name = uploaded.name
                    st.session_state.analysis  = None
                    st.session_state.insights  = None
                    st.session_state.charts    = None
                    st.session_state.report    = None
                    st.session_state.chat_msgs = []
                    agent = st.session_state.agent
                    agent.load_dataset(df)
                    st.success(f"✅ Loaded **{uploaded.name}**")

    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("---")
        st.markdown(f"**Rows:** {len(df):,}")
        st.markdown(f"**Cols:** {len(df.columns)}")
        st.markdown(f"**File:** `{st.session_state.file_name}`")
        st.markdown("---")

        if st.button("🔄 Re-run Analysis", use_container_width=True):
            st.session_state.analysis = None
            st.session_state.insights = None
            st.session_state.charts   = None
            st.session_state.report   = None
            st.rerun()

        if st.button("🗑️ Clear Cache", use_container_width=True):
            import llm as llm_mod
            llm_mod.clear_cache()
            st.success("Cache cleared.")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem;color:#c9d1d9;'>
    <b>Stack:</b> Python · pandas · Plotly<br>
    Streamlit · Meta Llama 3.3-70b<br><br>
    <b>LLM API:</b> Groq (free tier)<br>
    <a href='https://console.groq.com' target='_blank'>Get API key →</a>
    </div>
    """, unsafe_allow_html=True)


# ─── Main Content ─────────────────────────────────────────────────────────────

if st.session_state.df is None:
    # Landing screen
    st.markdown("""
    <div style='text-align:center;padding:4rem 2rem;'>
      <div style='font-size:4rem;margin-bottom:1rem;'>📂</div>
      <h2 style='color:#58a6ff;font-family:"IBM Plex Mono",monospace;'>
        Upload a CSV or Excel file to begin
      </h2>
      <p style='color:#c9d1d9;max-width:500px;margin:0 auto;'>
        The system will automatically analyse your data, generate visualisations,
        detect trends and anomalies, and provide AI-powered business insights.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    cols = st.columns(3)
    features = [
        ("📊", "Automatic Analysis", "Summary stats, trends, anomalies, and correlations"),
        ("🤖", "AI Insights", "Business recommendations powered by Meta Llama 3.3-70b"),
        ("💬", "Chat Interface", "Ask natural language questions about your data"),
    ]
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="padding:1.5rem;">
              <div style="font-size:2rem;margin-bottom:0.75rem;">{icon}</div>
              <h3 style="font-size:1rem;color:#e6edf3;font-family:'IBM Plex Sans',sans-serif;">
                {title}
              </h3>
              <p style="font-size:0.85rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    st.stop()


# ─── Run analysis (cached in session) ────────────────────────────────────────

agent: DataAnalystAgent = st.session_state.agent

if st.session_state.analysis is None:
    with st.spinner("🔍 Analysing dataset…"):
        st.session_state.analysis = agent.run_analysis()
    with st.spinner("📈 Generating charts…"):
        st.session_state.charts = agent.generate_charts()

analysis = st.session_state.analysis
df       = st.session_state.df


# ─── Quick Metric Bar ─────────────────────────────────────────────────────────

m1, m2, m3, m4, m5 = st.columns(5)
metrics = [
    (f"{analysis['shape']['rows']:,}", "Rows"),
    (str(analysis["shape"]["cols"]), "Columns"),
    (str(len(analysis["column_types"].get("numeric", []))), "Numeric"),
    (str(len(analysis["column_types"].get("categorical", []))), "Categorical"),
    (f"{analysis['missing']['missing_pct_overall']}%", "Missing"),
]
for col, (val, label) in zip([m1, m2, m3, m4, m5], metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <h3>{val}</h3>
          <p>{label}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab_overview, tab_viz, tab_ai, tab_chat, tab_report = st.tabs([
    "📊 Overview",
    "📈 Visualisations",
    "🤖 AI Insights",
    "💬 Chat",
    "📄 Report",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab_overview:
    st.markdown("### Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True, height=320)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Column Types")
        col_types = analysis["column_types"]
        type_data = []
        for dtype, cols in col_types.items():
            for col in cols:
                type_data.append({"Column": col, "Type": dtype.title()})
        if type_data:
            st.dataframe(pd.DataFrame(type_data), use_container_width=True, height=300)

    with c2:
        st.markdown("### Missing Values")
        miss = analysis["missing"]
        if miss["by_column"]:
            miss_df = pd.DataFrame([
                {"Column": c, "Missing": v["count"], "Pct (%)": v["pct"]}
                for c, v in miss["by_column"].items()
            ]).sort_values("Missing", ascending=False)
            st.dataframe(miss_df, use_container_width=True, height=300)
        else:
            st.success("✅ No missing values found!")

    st.markdown("### Numeric Summary Statistics")
    stats = analysis.get("stats", {}).get("numeric", {})
    if stats:
        stats_df = pd.DataFrame(stats).T.reset_index().rename(columns={"index": "Column"})
        st.dataframe(stats_df, use_container_width=True)

    st.markdown("### Categorical Summary")
    cat_stats = analysis.get("stats", {}).get("categorical", {})
    if cat_stats:
        cat_df = pd.DataFrame([
            {"Column": c, "Unique": s["unique"], "Top Value": s["top"],
             "Top Freq": s["top_freq"], "Missing": s["missing"]}
            for c, s in cat_stats.items()
        ])
        st.dataframe(cat_df, use_container_width=True)

    # Correlations table
    top_pairs = analysis.get("correlations", {}).get("top_pairs", [])
    if top_pairs:
        st.markdown("### Top Correlations")
        corr_df = pd.DataFrame(top_pairs)[["col_a", "col_b", "r", "strength", "direction"]]
        corr_df.columns = ["Column A", "Column B", "Pearson r", "Strength", "Direction"]
        st.dataframe(corr_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

with tab_viz:
    charts = st.session_state.charts or []
    if not charts:
        st.info("No charts generated. The dataset may not have plottable columns.")
    else:
        # FIX 1: Chart performance - Limit and lazy render charts
        MAX_CHARTS_DISPLAY = 10
        total_charts = len(charts)
        
        if total_charts > MAX_CHARTS_DISPLAY:
            st.info(f"📊 Showing first {MAX_CHARTS_DISPLAY} of {total_charts} charts (lazy loaded for performance)")
        else:
            st.markdown(f"### {total_charts} Charts Generated Automatically")
        
        # Display first batch of charts normally
        charts_to_show = charts[:MAX_CHARTS_DISPLAY]
        for i in range(0, len(charts_to_show), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(charts_to_show):
                    with col:
                        st.markdown(f"**{charts_to_show[idx]['title']}**")
                        st.plotly_chart(
                            charts_to_show[idx]["fig"],
                            use_container_width=True,
                            key=f"chart_{idx}",
                        )
        
        # FIX 1: Chart performance - Lazy load remaining charts in expanders
        remaining_charts = charts[MAX_CHARTS_DISPLAY:]
        if remaining_charts:
            st.markdown("---")
            with st.expander(f"📂 View Remaining {len(remaining_charts)} Charts (Lazy Loaded)", expanded=False):
                for i in range(0, len(remaining_charts), 2):
                    cols = st.columns(2)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(remaining_charts):
                            with col:
                                st.markdown(f"**{remaining_charts[idx]['title']}**")
                                st.plotly_chart(
                                    remaining_charts[idx]["fig"],
                                    use_container_width=True,
                                    key=f"chart_lazy_{idx}",
                                )

    # Custom scatter plot builder
    st.markdown("---")
    st.markdown("### 🔧 Custom Chart Builder")
    num_cols = analysis["column_types"].get("numeric", [])
    cat_cols = analysis["column_types"].get("categorical", [])

    if len(num_cols) >= 2:
        cc1, cc2, cc3 = st.columns(3)
        x_col = cc1.selectbox("X axis (numeric)", num_cols, key="cx")
        y_col = cc2.selectbox("Y axis (numeric)", num_cols,
                              index=min(1, len(num_cols)-1), key="cy")
        color_col = cc3.selectbox("Colour by (optional)", ["None"] + cat_cols, key="cc")
        if st.button("Generate Scatter Plot"):
            import visualizer as viz
            fig = viz.plot_scatter(
                df, x_col, y_col,
                color_col=None if color_col == "None" else color_col,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least 2 numeric columns for the custom scatter builder.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: AI INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════

with tab_ai:
    st.markdown("### 🤖 AI-Generated Business Insights")
    st.caption("Powered by Meta Llama 3.3-70b via Groq API · Results are cached per session")

    if st.session_state.insights is None:
        with st.spinner("Generating insights… (this may take 10-20 seconds)"):
            st.session_state.insights = agent.get_insights()

    with st.container():
        st.markdown(f"""
        <div class="insight-box">
        {st.session_state.insights}
        </div>
        """, unsafe_allow_html=True)

    # Trends section
    st.markdown("---")
    st.markdown("### 📈 Trend Explanations")
    trends = analysis.get("trends", [])
    if trends:
        if st.button("🔍 Explain Trends with AI", key="btn_trends"):
            with st.spinner("Analysing trends…"):
                trend_exp = agent.get_trend_explanation()
            st.markdown(trend_exp)
    else:
        st.info("No statistically significant trends were detected.")
        for t in trends:
            st.markdown(
                f"- **{t['column']}**: {t['strength'].title()} {t['direction']} trend "
                f"(R²={t['r_squared']})"
            )

    # Anomalies section
    st.markdown("---")
    st.markdown("### 🚨 Anomaly Report")
    anomalies = analysis.get("anomalies", {})
    if anomalies:
        cols_a = st.columns(3)
        for i, (col, info) in enumerate(list(anomalies.items())[:6]):
            with cols_a[i % 3]:
                st.markdown(f"""
                <div class="metric-card" style="margin-bottom:0.75rem;">
                  <h3 style="color:#f85149;">{info['count']}</h3>
                  <p>{col}</p>
                  <p style="font-size:0.72rem;margin-top:4px;">{info['pct']}% of rows</p>
                </div>
                """, unsafe_allow_html=True)
        if st.button("🔍 Explain Anomalies with AI", key="btn_anom"):
            with st.spinner("Analysing anomalies…"):
                anom_exp = agent.get_anomaly_explanation()
            st.markdown(anom_exp)
    else:
        st.success("✅ No significant anomalies detected.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: CHAT
# ══════════════════════════════════════════════════════════════════════════════

with tab_chat:
    st.markdown("### 💬 Ask Questions About Your Dataset")
    st.caption(
        "Ask anything: 'How many rows have missing values?', "
        "'What is the average sales?', 'Which category is most common?'"
    )

    # Render chat history
    for msg in st.session_state.chat_msgs:
        css_class = "chat-user" if msg["role"] == "user" else "chat-assistant"
        icon = "👤" if msg["role"] == "user" else "🤖"
        st.markdown(f"""
        <div class="{css_class}">
          <strong>{icon}</strong>&nbsp;&nbsp;{msg["content"]}
        </div>
        """, unsafe_allow_html=True)
        if "code" in msg:
            st.markdown(f"""
            <div class="code-block">
            <strong>Generated Code:</strong><br>
            {msg["code"].replace(chr(10), "<br>")}
            </div>
            """, unsafe_allow_html=True)

    # Suggested questions
    if not st.session_state.chat_msgs:
        st.markdown("**💡 Suggested questions:**")
        suggestions = [
            "How many rows have missing values?",
            "What are the top 5 most common values?",
            "What is the average of each numeric column?",
            "Are there any outliers in the data?",
        ]
        s_cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with s_cols[i % 2]:
                if st.button(suggestion, key=f"sugg_{i}", use_container_width=True):
                    st.session_state._pending_question = suggestion
                    st.rerun()

    # Handle pending question from suggestion buttons
    if hasattr(st.session_state, "_pending_question"):
        question = st.session_state._pending_question
        del st.session_state._pending_question
        with st.spinner("Thinking…"):
            response = agent.chat(question)
        st.session_state.chat_msgs.append({"role": "user", "content": question})
        entry = {"role": "assistant", "content": response["content"]}
        if "code" in response:
            entry["code"] = response["code"]
        st.session_state.chat_msgs.append(entry)
        st.rerun()

    # Chat input
    question = st.chat_input("Ask a question about your dataset…")
    if question:
        with st.spinner("Thinking…"):
            response = agent.chat(question)
        st.session_state.chat_msgs.append({"role": "user", "content": question})
        entry = {"role": "assistant", "content": response["content"]}
        if "code" in response:
            entry["code"] = response["code"]
        st.session_state.chat_msgs.append(entry)
        st.rerun()

    if st.session_state.chat_msgs:
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state.chat_msgs = []
            agent.chat_history = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: REPORT
# ══════════════════════════════════════════════════════════════════════════════

with tab_report:
    st.markdown("### 📄 Download Analysis Report")

    if st.button("📝 Generate Full Report", use_container_width=False):
        with st.spinner("Building report…"):
            if st.session_state.insights is None:
                st.session_state.insights = agent.get_insights()
            st.session_state.report = agent.build_report()

    if st.session_state.report:
        st.markdown(st.session_state.report)
        
        # FIX 3: PDF export - Add markdown download button
        st.download_button(
            label="⬇️ Download Report (.md)",
            data=st.session_state.report,
            file_name=f"analysis_report_{st.session_state.file_name or 'dataset'}.md",
            mime="text/markdown",
            use_container_width=True,
        )
        
        # FIX 3: PDF export - Add PDF download button with reportlab conversion
        pdf_buffer = markdown_to_pdf(st.session_state.report)
        st.download_button(
            label="⬇️ Download Report (.pdf)",
            data=pdf_buffer,
            file_name=f"analysis_report_{st.session_state.file_name or 'dataset'}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        # Also offer the raw analysis JSON
        analysis_json = json.dumps(
            {k: v for k, v in analysis.items() if k != "correlations"},
            indent=2, default=str
        )
        st.download_button(
            label="⬇️ Download Analysis JSON",
            data=analysis_json,
            file_name="analysis_data.json",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.info("Click 'Generate Full Report' to create a comprehensive markdown report.")

