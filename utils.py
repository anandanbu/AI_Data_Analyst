"""
utils.py - Shared utility functions for the AI Data Analyst system.
Handles file validation, safe code execution, and data sampling.
"""

import io
import re
import traceback
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Any


# ─── File Validation ────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_FILE_SIZE_MB = 50


def validate_file(file_obj) -> Tuple[bool, str]:
    """
    Validate uploaded file by extension and size.
    Returns (is_valid, error_message).
    """
    name = file_obj.name.lower()
    ext = "." + name.rsplit(".", 1)[-1] if "." in name else ""

    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type '{ext}'. Please upload CSV or Excel (.xlsx/.xls)."

    # Check size (Streamlit UploadedFile supports len())
    file_obj.seek(0, 2)          # Seek to end
    size_mb = file_obj.tell() / (1024 * 1024)
    file_obj.seek(0)             # Reset

    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"File is {size_mb:.1f} MB. Maximum allowed size is {MAX_FILE_SIZE_MB} MB."

    return True, ""


def load_dataframe(file_obj) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load a CSV or Excel file into a pandas DataFrame.
    Returns (df, error_message). On success error_message is empty.
    """
    name = file_obj.name.lower()
    try:
        if name.endswith(".csv"):
            # Try common encodings
            for enc in ("utf-8", "latin-1", "cp1252"):
                try:
                    file_obj.seek(0)
                    df = pd.read_csv(file_obj, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                return None, "Could not decode CSV file. Try saving it as UTF-8."
        elif name.endswith((".xlsx", ".xls")):
            file_obj.seek(0)
            df = pd.read_excel(file_obj)
        else:
            return None, "Unsupported file type."

        if df.empty:
            return None, "The uploaded file appears to be empty."

        return df, ""

    except Exception as e:
        return None, f"Failed to load file: {e}"


# ─── Data Sampling ───────────────────────────────────────────────────────────

def sample_dataframe(df: pd.DataFrame, n_rows: int = 5, max_cols: int = 20) -> str:
    """
    Return a compact string representation of a DataFrame sample.
    Limits columns to avoid bloating the LLM prompt.
    """
    cols = df.columns.tolist()
    if len(cols) > max_cols:
        cols = cols[:max_cols]
    sample = df[cols].head(n_rows)
    return sample.to_string(index=False)


def summarise_dataframe(df: pd.DataFrame) -> dict:
    """
    Build a lightweight summary dict suitable for LLM prompts.
    Never sends the full dataset.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    # Numeric describe (only key stats to save tokens)
    num_summary = {}
    for col in numeric_cols[:15]:          # Cap at 15 numeric columns
        s = df[col].dropna()
        num_summary[col] = {
            "mean": round(float(s.mean()), 4) if len(s) else None,
            "std":  round(float(s.std()),  4) if len(s) else None,
            "min":  round(float(s.min()),  4) if len(s) else None,
            "max":  round(float(s.max()),  4) if len(s) else None,
            "nulls": int(df[col].isna().sum()),
        }

    # Categorical top values
    cat_summary = {}
    for col in cat_cols[:10]:
        vc = df[col].value_counts()
        cat_summary[col] = {
            "unique": int(df[col].nunique()),
            "top5": vc.head(5).to_dict(),
            "nulls": int(df[col].isna().sum()),
        }

    return {
        "shape": {"rows": len(df), "cols": len(df.columns)},
        "columns": df.columns.tolist(),
        "numeric_cols": numeric_cols,
        "categorical_cols": cat_cols,
        "datetime_cols": date_cols,
        "missing_total": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "numeric_summary": num_summary,
        "categorical_summary": cat_summary,
    }


# ─── Safe Code Execution ─────────────────────────────────────────────────────

# Patterns that are never allowed in generated code
_BLOCKED_PATTERNS = [
    r"\bos\b", r"\bsubprocess\b", r"\beval\b", r"\bexec\b",
    r"\bopen\b", r"\b__import__\b", r"\bshutil\b", r"\bsocket\b",
    r"\bpickle\b", r"\bctypes\b", r"\bimportlib\b",
    r"import\s+os", r"import\s+sys", r"import\s+subprocess",
]

_ALLOWED_IMPORTS = {"pandas", "numpy", "math", "statistics", "re", "json", "datetime"}


def is_safe_code(code: str) -> Tuple[bool, str]:
    """
    Check generated Python code for dangerous patterns.
    Returns (is_safe, reason).
    """
    for pattern in _BLOCKED_PATTERNS:
        if re.search(pattern, code):
            return False, f"Blocked pattern detected: '{pattern}'"
    return True, ""


def execute_code_safely(code: str, df: pd.DataFrame) -> Tuple[Any, str]:
    """
    Execute LLM-generated pandas code in a restricted namespace.
    Returns (result, error_message).
    """
    safe, reason = is_safe_code(code)
    if not safe:
        return None, f"Code safety check failed: {reason}"

    # Restricted globals - only pandas + numpy available
    restricted_globals = {
        "__builtins__": {
            "len": len, "range": range, "enumerate": enumerate,
            "zip": zip, "map": map, "filter": filter, "sorted": sorted,
            "sum": sum, "min": min, "max": max, "abs": abs,
            "round": round, "int": int, "float": float, "str": str,
            "bool": bool, "list": list, "dict": dict, "tuple": tuple,
            "set": set, "print": print, "isinstance": isinstance,
            "type": type,
        },
        "pd": pd,
        "np": np,
        "df": df.copy(),     # Pass a copy so the original is never mutated
    }

    local_ns = {}
    try:
        exec(compile(code, "<generated>", "exec"), restricted_globals, local_ns)
        # Return the last assigned variable named 'result', or all locals
        if "result" in local_ns:
            return local_ns["result"], ""
        # Fallback: return the last value if only one variable was assigned
        if local_ns:
            return list(local_ns.values())[-1], ""
        return "Code executed successfully (no output variable named 'result').", ""
    except Exception:
        return None, traceback.format_exc(limit=5)


# ─── Formatting Helpers ───────────────────────────────────────────────────────

def format_number(n) -> str:
    """Pretty-format large numbers."""
    try:
        n = float(n)
        if abs(n) >= 1_000_000:
            return f"{n/1_000_000:.2f}M"
        if abs(n) >= 1_000:
            return f"{n/1_000:.1f}K"
        return f"{n:.2f}"
    except Exception:
        return str(n)


def infer_date_column(df: pd.DataFrame) -> Optional[str]:
    """Heuristically find the most likely datetime column."""
    # Already-parsed datetime columns
    dt_cols = df.select_dtypes(include="datetime64").columns.tolist()
    if dt_cols:
        return dt_cols[0]

    # String columns with date-like names
    date_keywords = ["date", "time", "timestamp", "created", "updated", "month", "year", "day"]
    for col in df.columns:
        if any(kw in col.lower() for kw in date_keywords):
            try:
                pd.to_datetime(df[col], infer_datetime_format=True)
                return col
            except Exception:
                pass
    return None
