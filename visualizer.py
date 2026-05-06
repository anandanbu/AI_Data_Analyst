"""
visualizer.py - Visualization Engine for the AI Data Analyst system.

Automatically selects chart types based on column dtype:
  - Numeric  → histogram + box plot
  - Categorical → horizontal bar chart
  - Datetime (time series) → line chart
  - Two numeric cols → scatter plot
  - Correlation matrix → heatmap
"""

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Design Tokens ───────────────────────────────────────────────────────────

PALETTE = px.colors.qualitative.Bold
TEMPLATE = "plotly_dark"
FONT_FAMILY = "IBM Plex Mono, monospace"
BG_COLOR = "#0e1117"
PAPER_COLOR = "#161b22"


def _base_layout(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=16, family=FONT_FAMILY, color="#e6edf3")),
        template=TEMPLATE,
        paper_bgcolor=PAPER_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(family=FONT_FAMILY, color="#8b949e"),
        margin=dict(l=40, r=40, t=60, b=40),
    )


# ─── Individual Chart Generators ─────────────────────────────────────────────

def plot_histogram(df: pd.DataFrame, col: str) -> go.Figure:
    """Histogram + KDE overlay for a numeric column."""
    fig = px.histogram(
        df, x=col, nbins=40,
        color_discrete_sequence=[PALETTE[0]],
        marginal="box",
        opacity=0.85,
        title=f"Distribution of {col}",
    )
    fig.update_layout(**_base_layout(f"Distribution — {col}"))
    return fig


def plot_bar(df: pd.DataFrame, col: str, top_n: int = 20) -> go.Figure:
    """Horizontal bar chart for a categorical column."""
    vc = df[col].value_counts().head(top_n).sort_values()
    fig = go.Figure(go.Bar(
        x=vc.values,
        y=vc.index.astype(str),
        orientation="h",
        marker_color=PALETTE[1],
        opacity=0.9,
    ))
    fig.update_layout(
        **_base_layout(f"Top {top_n} Values — {col}"),
        xaxis_title="Count",
        yaxis_title=col,
    )
    return fig


def plot_line(df: pd.DataFrame, date_col: str, value_col: str) -> go.Figure:
    """Line chart for a time series."""
    tmp = df[[date_col, value_col]].dropna().sort_values(date_col)
    fig = go.Figure(go.Scatter(
        x=tmp[date_col],
        y=tmp[value_col],
        mode="lines",
        line=dict(color=PALETTE[2], width=2),
        name=value_col,
    ))
    fig.update_layout(
        **_base_layout(f"{value_col} over {date_col}"),
        xaxis_title=date_col,
        yaxis_title=value_col,
    )
    return fig


def plot_scatter(df: pd.DataFrame, col_x: str, col_y: str,
                 color_col: Optional[str] = None) -> go.Figure:
    """Scatter plot for two numeric columns, optionally coloured by a categorical col."""
    kwargs = dict(x=col_x, y=col_y, opacity=0.65,
                  color_discrete_sequence=PALETTE, title=f"{col_x} vs {col_y}")
    if color_col and color_col in df.columns:
        kwargs["color"] = color_col
    fig = px.scatter(df.sample(min(3000, len(df))), **kwargs)
    fig.update_layout(**_base_layout(f"{col_x} vs {col_y}"))
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> Optional[go.Figure]:
    """Pearson correlation heatmap for numeric columns."""
    num_df = df.select_dtypes(include="number")
    if len(num_df.columns) < 2:
        return None

    corr = num_df.corr(method="pearson").round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=corr.values,
        texttemplate="%{text}",
        textfont=dict(size=10),
    ))
    fig.update_layout(**_base_layout("Correlation Matrix"))
    return fig


def plot_boxplot(df: pd.DataFrame, cols: list[str]) -> go.Figure:
    """Side-by-side box plots for multiple numeric columns (normalised)."""
    fig = go.Figure()
    for i, col in enumerate(cols[:10]):        # Limit to 10 columns
        s = df[col].dropna()
        fig.add_trace(go.Box(
            y=s.values,
            name=col,
            marker_color=PALETTE[i % len(PALETTE)],
            boxmean="sd",
        ))
    fig.update_layout(**_base_layout("Numeric Column Distributions"))
    return fig


def plot_missing_heatmap(df: pd.DataFrame) -> Optional[go.Figure]:
    """Visualise missing values as a heatmap."""
    miss_df = df.isnull().astype(int)
    if miss_df.sum().sum() == 0:
        return None
    # Only show columns with at least one missing value
    miss_df = miss_df[miss_df.columns[miss_df.sum() > 0]]
    if miss_df.empty:
        return None
    # Sample rows for display
    miss_df = miss_df.head(200)
    fig = go.Figure(go.Heatmap(
        z=miss_df.values.T,
        x=list(range(len(miss_df))),
        y=miss_df.columns.tolist(),
        colorscale=[[0, BG_COLOR], [1, "#f85149"]],
        showscale=False,
    ))
    fig.update_layout(**_base_layout("Missing Value Map (red = missing)"))
    return fig


# ─── Auto Chart Generator ─────────────────────────────────────────────────────

def auto_generate_charts(df: pd.DataFrame) -> list[dict]:
    """
    Automatically choose and generate appropriate charts for the dataset.
    Returns a list of {"title": str, "fig": go.Figure} dicts.

    Decision logic:
      1. Correlation heatmap (if ≥2 numeric cols)
      2. Box plots for all numeric cols
      3. Histograms for up to 6 numeric cols
      4. Bar charts for up to 6 categorical cols
      5. Time-series lines if a datetime col exists
      6. Missing value map if missing data present
    """
    charts = []
    num_cols  = df.select_dtypes(include="number").columns.tolist()
    cat_cols  = df.select_dtypes(include=["object", "category"]).columns.tolist()
    dt_cols   = df.select_dtypes(include="datetime64").columns.tolist()

    # Try to auto-parse object columns as datetime
    for col in list(cat_cols):
        try:
            parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
            if parsed.notna().mean() > 0.8:
                df = df.copy()
                df[col] = parsed
                dt_cols.append(col)
                cat_cols.remove(col)
        except Exception:
            pass

    # 1. Correlation heatmap
    if len(num_cols) >= 2:
        fig = plot_correlation_heatmap(df)
        if fig:
            charts.append({"title": "Correlation Matrix", "fig": fig})

    # 2. Box plots
    if num_cols:
        charts.append({
            "title": "Numeric Distributions (Box Plots)",
            "fig": plot_boxplot(df, num_cols),
        })

    # 3. Histograms (up to 6)
    for col in num_cols[:6]:
        charts.append({"title": f"Histogram — {col}", "fig": plot_histogram(df, col)})

    # 4. Bar charts (up to 6)
    for col in cat_cols[:6]:
        if df[col].nunique() <= 50:    # Skip high-cardinality text columns
            charts.append({"title": f"Bar Chart — {col}", "fig": plot_bar(df, col)})

    # 5. Time-series
    if dt_cols and num_cols:
        dt_col = dt_cols[0]
        for val_col in num_cols[:3]:    # Up to 3 time-series charts
            charts.append({
                "title": f"{val_col} over time",
                "fig": plot_line(df, dt_col, val_col),
            })

    # 6. Missing value map
    miss_fig = plot_missing_heatmap(df)
    if miss_fig:
        charts.append({"title": "Missing Value Map", "fig": miss_fig})

    return charts
