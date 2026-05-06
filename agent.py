"""
agent.py - Agent Logic & Orchestration for the AI Data Analyst system.

The agent:
  1. Decides what analyses to run based on the dataset characteristics
  2. Decides when to call the LLM (minimise unnecessary API calls)
  3. Decides which charts to generate
  4. Routes chat questions → LLM answer vs pandas code execution
  5. Assembles the final analysis package
"""

import json
import re
from typing import Any, Optional

import pandas as pd

import analyzer
import llm
import visualizer
from utils import execute_code_safely, sample_dataframe, summarise_dataframe


# ─── Analysis Decision Logic ──────────────────────────────────────────────────

class DataAnalystAgent:
    """
    Stateful agent that orchestrates the full analysis pipeline.
    State is held per-session in Streamlit's session_state.
    """

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.summary: dict = {}
        self.analysis: dict = {}
        self.insights: str = ""
        self.charts: list[dict] = []
        self.chat_history: list[dict] = []
        self._sample_rows: str = ""

    # ─── Setup ────────────────────────────────────────────────────────────────

    def load_dataset(self, df: pd.DataFrame) -> None:
        """Accept a loaded DataFrame and reset analysis state."""
        self.df = df
        self.summary = {}
        self.analysis = {}
        self.insights = ""
        self.charts = []
        self.chat_history = []
        self._sample_rows = sample_dataframe(df, n_rows=5)

    # ─── Analysis Orchestration ───────────────────────────────────────────────

    def run_analysis(self) -> dict[str, Any]:
        """
        Decide which analyses to run based on dataset characteristics,
        then execute them all. Returns the consolidated analysis dict.
        """
        if self.df is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")

        df = self.df
        decisions = self._make_analysis_decisions(df)

        result: dict[str, Any] = {
            "shape": {"rows": len(df), "cols": len(df.columns)},
            "decisions": decisions,
        }

        # Always run: column classification + missing values + summary stats
        result["column_types"] = analyzer.classify_columns(df.copy())
        result["missing"]      = analyzer.analyze_missing(df)
        result["stats"]        = analyzer.compute_summary_stats(df)
        result["duplicates"]   = int(df.duplicated().sum())

        # Conditionally run more expensive analyses
        if decisions["run_trend"]:
            result["trends"] = analyzer.detect_trends(df)
        else:
            result["trends"] = []

        if decisions["run_anomaly"]:
            result["anomalies"] = analyzer.detect_anomalies(df)
        else:
            result["anomalies"] = {}

        if decisions["run_correlation"]:
            result["correlations"] = analyzer.compute_correlations(df)
        else:
            result["correlations"] = {"matrix": {}, "top_pairs": []}

        self.analysis = result
        self.summary  = summarise_dataframe(df)
        return result

    def _make_analysis_decisions(self, df: pd.DataFrame) -> dict[str, bool]:
        """
        Rule-based decisions on what to run.
        Keeps analysis focused and avoids wasted computation.
        """
        num_cols = df.select_dtypes(include="number").columns.tolist()
        dt_cols  = df.select_dtypes(include="datetime64").columns.tolist()
        # Also check string cols that might be dates
        has_datetime = bool(dt_cols) or any(
            any(kw in c.lower() for kw in ["date", "time", "month", "year"])
            for c in df.columns
        )

        return {
            "run_trend":       has_datetime and len(num_cols) >= 1,
            "run_anomaly":     len(num_cols) >= 1 and len(df) >= 20,
            "run_correlation": len(num_cols) >= 2,
            "generate_charts": True,   # Always generate charts
            "call_llm":        True,   # Always call LLM for insights (cached)
        }

    # ─── Chart Generation ─────────────────────────────────────────────────────

    def generate_charts(self) -> list[dict]:
        """Generate all auto-charts for the current dataset."""
        if self.df is None:
            return []
        self.charts = visualizer.auto_generate_charts(self.df)
        return self.charts

    # ─── LLM Calls (with gate logic) ──────────────────────────────────────────

    def get_insights(self) -> str:
        """
        Generate LLM insights — only call the API once per dataset load.
        Subsequent calls return the cached string.
        """
        if self.insights:
            return self.insights
        if not self.analysis:
            self.run_analysis()
        self.insights = llm.generate_insights(self.summary, self._sample_rows)
        return self.insights

    def get_anomaly_explanation(self) -> str:
        """Ask LLM to explain anomalies — only if anomalies exist."""
        anomalies = self.analysis.get("anomalies", {})
        if not anomalies:
            return "No significant anomalies were detected in this dataset."
        col_stats = self.analysis.get("stats", {}).get("numeric", {})
        return llm.explain_anomalies(anomalies, col_stats)

    def get_trend_explanation(self) -> str:
        """Ask LLM to explain trends — only if trends exist."""
        trends = self.analysis.get("trends", [])
        return llm.explain_trends(trends)

    # ─── Chat Interface ───────────────────────────────────────────────────────

    def chat(self, question: str) -> dict[str, Any]:
        """
        Process a user question about the dataset.

        Decision tree:
          1. If question looks computational (count, sum, filter, etc.)
             → generate pandas code → execute safely → return result
          2. Otherwise → ask LLM with summary context → return answer
        """
        if self.df is None:
            return {"type": "error", "content": "No dataset loaded."}

        self.chat_history.append({"role": "user", "content": question})

        if self._is_computational_question(question):
            response = self._handle_computational_question(question)
        else:
            response = self._handle_conversational_question(question)

        self.chat_history.append({"role": "assistant", "content": response["content"]})
        return response

    def _is_computational_question(self, q: str) -> bool:
        """
        Heuristic: does the question require running pandas code?
        """
        computational_keywords = [
            "count", "how many", "sum", "total", "average", "mean", "max", "min",
            "top", "bottom", "filter", "where", "show me rows", "list", "find",
            "calculate", "compute", "percentage", "ratio", "group by", "which rows",
        ]
        q_lower = q.lower()
        return any(kw in q_lower for kw in computational_keywords)

    def _handle_computational_question(self, question: str) -> dict[str, Any]:
        """
        Generate + execute pandas code to answer a computational question.
        Falls back to LLM answer if code execution fails.
        """
        df = self.df
        dtypes = {col: str(df[col].dtype) for col in df.columns}

        # Ask LLM to write the pandas code
        code = llm.generate_pandas_code(question, df.columns.tolist(), dtypes)

        # Strip markdown fences if present
        code = re.sub(r"```python\s*", "", code)
        code = re.sub(r"```\s*", "", code)
        code = code.strip()

        result, error = execute_code_safely(code, df)

        if error:
            # Fallback: answer via LLM without code execution
            fallback = llm.answer_question(question, self.summary, self._sample_rows)
            return {
                "type": "llm_answer",
                "content": fallback,
                "code": code,
                "code_error": error,
            }

        # Format the result nicely
        if isinstance(result, pd.DataFrame):
            content = f"Query result ({len(result)} rows):\n\n{result.head(20).to_markdown(index=False)}"
        elif isinstance(result, pd.Series):
            content = f"Result:\n\n{result.head(20).to_markdown()}"
        else:
            content = f"Result: **{result}**"

        return {
            "type": "code_result",
            "content": content,
            "code": code,
        }

    def _handle_conversational_question(self, question: str) -> dict[str, Any]:
        """Answer a conceptual / descriptive question using LLM + summary context."""
        answer = llm.answer_question(question, self.summary, self._sample_rows)
        return {"type": "llm_answer", "content": answer}

    # ─── Report Generation ────────────────────────────────────────────────────

    def build_report(self) -> str:
        """
        Build a full markdown report combining analysis + LLM insights.
        """
        if not self.analysis:
            self.run_analysis()

        df = self.df
        analysis = self.analysis
        insights = self.get_insights()

        num_cols = analysis.get("column_types", {}).get("numeric", [])
        cat_cols = analysis.get("column_types", {}).get("categorical", [])
        trends   = analysis.get("trends", [])
        anomalies = analysis.get("anomalies", {})
        corr_pairs = analysis.get("correlations", {}).get("top_pairs", [])

        lines = [
            "# AI Data Analysis Report",
            "",
            "## 1. Dataset Overview",
            f"- **Rows:** {analysis['shape']['rows']:,}",
            f"- **Columns:** {analysis['shape']['cols']}",
            f"- **Numeric Columns:** {len(num_cols)}",
            f"- **Categorical Columns:** {len(cat_cols)}",
            f"- **Missing Values:** {analysis['missing']['total_missing']:,} "
              f"({analysis['missing']['missing_pct_overall']}%)",
            f"- **Duplicate Rows:** {analysis['duplicates']:,}",
            "",
            "## 2. AI-Generated Insights",
            insights,
            "",
            "## 3. Key Statistics",
        ]

        stats = analysis.get("stats", {}).get("numeric", {})
        for col, s in list(stats.items())[:8]:
            lines += [
                f"### {col}",
                f"- Mean: {s['mean']} | Median: {s['median']} | Std: {s['std']}",
                f"- Range: [{s['min']}, {s['max']}] | Skewness: {s['skew_desc']}",
            ]

        lines += ["", "## 4. Trends Detected"]
        if trends:
            for t in trends:
                lines.append(f"- **{t['column']}**: {t['strength'].title()} {t['direction']} trend "
                              f"(R²={t['r_squared']}, p={t['p_value']})")
        else:
            lines.append("- No statistically significant trends detected.")

        lines += ["", "## 5. Anomalies"]
        if anomalies:
            for col, a in list(anomalies.items())[:8]:
                lines.append(f"- **{col}**: {a['count']} outliers ({a['pct']}%) "
                              f"outside [{a['lower_bound']}, {a['upper_bound']}]")
        else:
            lines.append("- No anomalies detected.")

        lines += ["", "## 6. Top Correlations"]
        if corr_pairs:
            for p in corr_pairs[:5]:
                lines.append(f"- **{p['col_a']}** ↔ **{p['col_b']}**: "
                              f"{p['strength'].title()} {p['direction']} correlation (r={p['r']})")
        else:
            lines.append("- No significant correlations found.")

        lines += ["", "---", "*Generated by AI Data Analyst · Powered by Meta Llama 3.3-70b*"]
        return "\n".join(lines)
