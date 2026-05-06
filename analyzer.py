"""
analyzer.py - Data Analysis Engine for the AI Data Analyst system.

Responsibilities:
  - Detect column types (numeric, categorical, datetime)
  - Handle and report missing values
  - Generate summary statistics
  - Detect trends (via linear regression on numeric columns)
  - Detect anomalies (IQR-based outlier detection)
  - Compute correlations
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# ─── Column Classification ────────────────────────────────────────────────────

def classify_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Classify each column into: numeric, categorical, datetime, boolean, other.
    Also attempts to auto-parse string columns that look like dates.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    bool_cols    = df.select_dtypes(include="bool").columns.tolist()
    dt_cols      = df.select_dtypes(include="datetime64").columns.tolist()
    cat_cols     = []
    other_cols   = []

    for col in df.select_dtypes(include=["object", "category"]).columns:
        # Try to parse as datetime
        try:
            parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
            if parsed.notna().mean() > 0.8:      # 80%+ parse success → datetime
                df[col] = parsed
                dt_cols.append(col)
                continue
        except Exception:
            pass
        # If fewer than 50 unique values (or < 5% of rows) treat as categorical
        if df[col].nunique() < 50 or df[col].nunique() / len(df) < 0.05:
            cat_cols.append(col)
        else:
            other_cols.append(col)

    return {
        "numeric":     numeric_cols,
        "categorical": cat_cols,
        "datetime":    dt_cols,
        "boolean":     bool_cols,
        "other":       other_cols,
    }


# ─── Missing Value Analysis ───────────────────────────────────────────────────

def analyze_missing(df: pd.DataFrame) -> dict[str, Any]:
    """
    Return a detailed missing-value report.
    """
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    cols_with_missing = missing[missing > 0]

    return {
        "total_missing": int(missing.sum()),
        "total_cells": int(df.size),
        "missing_pct_overall": round(missing.sum() / df.size * 100, 2),
        "by_column": {
            col: {
                "count": int(cnt),
                "pct": float(missing_pct[col]),
            }
            for col, cnt in cols_with_missing.items()
        },
        "complete_columns": int((missing == 0).sum()),
    }


# ─── Summary Statistics ───────────────────────────────────────────────────────

def compute_summary_stats(df: pd.DataFrame) -> dict[str, Any]:
    """
    Extended descriptive statistics for numeric and categorical columns.
    """
    col_types = classify_columns(df.copy())
    result = {"numeric": {}, "categorical": {}}

    for col in col_types["numeric"]:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        skewness = float(s.skew())
        kurtosis = float(s.kurtosis())
        result["numeric"][col] = {
            "count":    int(s.count()),
            "mean":     round(float(s.mean()), 4),
            "median":   round(float(s.median()), 4),
            "std":      round(float(s.std()), 4),
            "min":      round(float(s.min()), 4),
            "max":      round(float(s.max()), 4),
            "q1":       round(float(s.quantile(0.25)), 4),
            "q3":       round(float(s.quantile(0.75)), 4),
            "iqr":      round(float(s.quantile(0.75) - s.quantile(0.25)), 4),
            "skewness": round(skewness, 4),
            "kurtosis": round(kurtosis, 4),
            "skew_desc": _describe_skew(skewness),
        }

    for col in col_types["categorical"]:
        vc = df[col].value_counts()
        result["categorical"][col] = {
            "unique":   int(df[col].nunique()),
            "top":      str(vc.index[0]) if len(vc) else None,
            "top_freq": int(vc.iloc[0]) if len(vc) else None,
            "top5":     {str(k): int(v) for k, v in vc.head(5).items()},
            "missing":  int(df[col].isna().sum()),
        }

    return result


def _describe_skew(skew: float) -> str:
    if skew > 1:      return "Highly right-skewed"
    if skew > 0.5:    return "Moderately right-skewed"
    if skew > -0.5:   return "Approximately symmetric"
    if skew > -1:     return "Moderately left-skewed"
    return "Highly left-skewed"


# ─── Trend Detection ──────────────────────────────────────────────────────────

def detect_trends(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    For each numeric column, run a linear regression against the row index
    (or a datetime index if one exists) to detect monotonic trends.
    Returns a list of trend dicts for columns with statistically significant trends.
    """
    col_types = classify_columns(df.copy())
    numeric_cols = col_types["numeric"]
    trends = []

    # Use datetime column as x-axis if available
    x = np.arange(len(df), dtype=float)
    x_label = "row index"
    if col_types["datetime"]:
        dt_col = col_types["datetime"][0]
        try:
            x_dt = pd.to_datetime(df[dt_col])
            x = (x_dt - x_dt.min()).dt.total_seconds().values.astype(float)
            x_label = dt_col
        except Exception:
            pass

    for col in numeric_cols[:20]:     # Cap at 20 columns for speed
        y = df[col].values.astype(float)
        valid = ~np.isnan(x) & ~np.isnan(y)
        if valid.sum() < 10:          # Need at least 10 points
            continue

        slope, intercept, r_value, p_value, _ = stats.linregress(x[valid], y[valid])
        r2 = r_value ** 2

        if p_value < 0.05 and r2 > 0.1:   # Statistically significant + decent fit
            trends.append({
                "column": col,
                "slope": round(slope, 6),
                "r_squared": round(r2, 4),
                "p_value": round(p_value, 6),
                "direction": "increasing" if slope > 0 else "decreasing",
                "strength": _trend_strength(r2),
                "x_axis": x_label,
            })

    return sorted(trends, key=lambda t: t["r_squared"], reverse=True)


def _trend_strength(r2: float) -> str:
    if r2 > 0.7: return "strong"
    if r2 > 0.4: return "moderate"
    return "weak"


# ─── Anomaly Detection ────────────────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame) -> dict[str, Any]:
    """
    IQR-based outlier detection for numeric columns.
    Returns per-column outlier counts and the actual outlier rows (up to 5).
    """
    col_types = classify_columns(df.copy())
    anomalies = {}

    for col in col_types["numeric"]:
        s = df[col].dropna()
        if len(s) < 10:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_mask = (df[col] < lower) | (df[col] > upper)
        outlier_count = int(outlier_mask.sum())
        if outlier_count == 0:
            continue

        outlier_rows = df[outlier_mask][col].head(5).tolist()
        anomalies[col] = {
            "count": outlier_count,
            "pct": round(outlier_count / len(df) * 100, 2),
            "lower_bound": round(float(lower), 4),
            "upper_bound": round(float(upper), 4),
            "sample_values": [round(float(v), 4) for v in outlier_rows],
        }

    return anomalies


# ─── Correlation Analysis ─────────────────────────────────────────────────────

def compute_correlations(df: pd.DataFrame) -> dict[str, Any]:
    """
    Pearson correlation matrix for numeric columns.
    Also identifies the top strongly-correlated pairs.
    """
    col_types = classify_columns(df.copy())
    numeric_cols = col_types["numeric"]

    if len(numeric_cols) < 2:
        return {"matrix": {}, "top_pairs": [], "note": "Need ≥2 numeric columns."}

    corr_df = df[numeric_cols].corr(method="pearson")

    # Extract top correlated pairs (excluding self-correlations)
    pairs = []
    cols = numeric_cols
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_df.iloc[i, j]
            if abs(r) > 0.3:    # Only meaningful correlations
                pairs.append({
                    "col_a": cols[i],
                    "col_b": cols[j],
                    "r": round(float(r), 4),
                    "strength": _corr_strength(r),
                    "direction": "positive" if r > 0 else "negative",
                })

    pairs = sorted(pairs, key=lambda p: abs(p["r"]), reverse=True)

    return {
        "matrix": corr_df.round(4).to_dict(),
        "top_pairs": pairs[:10],
    }


def _corr_strength(r: float) -> str:
    a = abs(r)
    if a > 0.8: return "very strong"
    if a > 0.6: return "strong"
    if a > 0.4: return "moderate"
    return "weak"


# ─── Full Analysis Report ─────────────────────────────────────────────────────

def run_full_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """
    Orchestrate all analysis steps and return a consolidated report.
    This is the main entry point called by the agent.
    """
    col_types = classify_columns(df.copy())
    return {
        "shape":         {"rows": len(df), "cols": len(df.columns)},
        "column_types":  col_types,
        "missing":       analyze_missing(df),
        "stats":         compute_summary_stats(df),
        "trends":        detect_trends(df),
        "anomalies":     detect_anomalies(df),
        "correlations":  compute_correlations(df),
        "duplicates":    int(df.duplicated().sum()),
        "memory_mb":     round(df.memory_usage(deep=True).sum() / 1024**2, 2),
    }
