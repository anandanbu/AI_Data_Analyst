"""
llm.py - LLM Interaction Layer using Meta Llama 3.3-70b via Groq API.

Features:
  - Response caching (in-memory, keyed by prompt hash)
  - Smart token budgeting — never sends raw dataset
  - Structured prompt templates for different tasks
  - Graceful error handling
"""

import hashlib
import json
import os
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ─── Client Setup ─────────────────────────────────────────────────────────────

_client: Optional[Groq] = None
_cache: dict[str, str] = {}          # In-memory LLM response cache

MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1500


def get_client() -> Groq:
    """Lazy-initialise the Groq client."""
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not found. "
                "Please add it to your .env file. "
                "Get a free key at https://console.groq.com"
            )
        _client = Groq(api_key=api_key)
    return _client


# ─── Cache Helpers ────────────────────────────────────────────────────────────

def _cache_key(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()


def _from_cache(prompt: str) -> Optional[str]:
    return _cache.get(_cache_key(prompt))


def _to_cache(prompt: str, response: str) -> None:
    _cache[_cache_key(prompt)] = response


def clear_cache() -> None:
    _cache.clear()


# ─── Core Completion ──────────────────────────────────────────────────────────

def _complete(system_prompt: str, user_prompt: str,
              use_cache: bool = True, max_tokens: int = MAX_TOKENS) -> str:
    """
    Single-turn completion. Returns the assistant's text.
    Uses cache by default to avoid redundant API calls.
    """
    full_prompt = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"

    if use_cache:
        cached = _from_cache(full_prompt)
        if cached:
            return cached + "  *(cached)*"

    try:
        client = get_client()
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        text = response.choices[0].message.content.strip()
        if use_cache:
            _to_cache(full_prompt, text)
        return text
    except Exception as e:
        return f"❌ LLM Error: {e}"


# ─── Prompt Templates ─────────────────────────────────────────────────────────

_ANALYST_SYSTEM = """You are an expert data analyst and business intelligence consultant.
Your task is to analyse the dataset summary provided and generate clear, actionable insights.
Be specific, quantitative where possible, and always recommend next steps.
Format your response with clear sections using markdown headers.
Do NOT hallucinate data — only reference what is explicitly provided in the summary."""

_CHAT_SYSTEM = """You are a friendly and conversational AI data analyst assistant.
A non-technical user is asking you questions about their dataset. Answer like a knowledgeable
friend explaining data insights - NOT like a developer or programmer.

STRICT RULES you must always follow:
- NEVER write or show any code, Python, pandas, or any programming syntax whatsoever
- NEVER mention technical terms like pandas, DataFrame, .groupby(), .value_counts(), etc.
- ALWAYS give the actual answer directly using numbers from the dataset summary provided
- If the exact figure is available, state it plainly e.g. "There are 1,234 missing values in CustomerID"
- If the data does not contain enough detail, give your best plain-English interpretation
- Speak in warm, clear, conversational sentences a business user would understand
- Use short bullet points when listing multiple findings, keep language simple
- Keep your answer concise: 3 to 6 sentences is ideal unless a detailed breakdown is needed
- Think of yourself as a business analyst presenting findings to a manager, never as a coder"""

_CODE_SYSTEM = """You are an expert Python/pandas developer.
Generate ONLY executable Python code that answers the user's question about the DataFrame `df`.
Rules:
1. The DataFrame is already loaded as `df`
2. Store your final answer in a variable called `result`
3. Do NOT import os, sys, subprocess, or any file I/O
4. Do NOT use eval() or exec()
5. Keep code concise — prefer one-liners where possible
6. Return ONLY the Python code, no explanation, no markdown fences"""


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_insights(summary: dict, sample_rows: str) -> str:
    """
    Generate high-level business insights from the dataset summary.
    Called once after analysis; result is cached.
    """
    prompt = f"""Dataset Summary:
{json.dumps(summary, indent=2, default=str)[:3000]}

Sample Rows:
{sample_rows[:500]}

Please provide:
1. **Executive Summary** — 3-4 sentence overview of the dataset
2. **Key Findings** — Top 5 data-driven findings with specific numbers
3. **Data Quality Issues** — Missing values, outliers, or concerns
4. **Business Recommendations** — 3 actionable recommendations
5. **Suggested Next Steps** — What analyses would add the most value
"""
    return _complete(_ANALYST_SYSTEM, prompt, use_cache=True)


def explain_anomalies(anomaly_report: dict, col_stats: dict) -> str:
    """
    Ask the LLM to explain detected anomalies in plain English.
    """
    prompt = f"""Anomaly Report (IQR-based outlier detection):
{json.dumps(anomaly_report, indent=2, default=str)[:2000]}

Column Statistics (for context):
{json.dumps(col_stats, indent=2, default=str)[:1500]}

Please explain:
1. Which columns have the most concerning outliers and why
2. What might be causing these anomalies (data entry errors, genuine extremes, etc.)
3. How they might impact analysis and what to do about them
"""
    return _complete(_ANALYST_SYSTEM, prompt, use_cache=True)


def explain_trends(trend_report: list) -> str:
    """
    Narrate detected trends in business-friendly language.
    """
    if not trend_report:
        return "No statistically significant trends were detected in this dataset."
    prompt = f"""Trend Analysis Report:
{json.dumps(trend_report, indent=2, default=str)[:2000]}

Please explain each trend in plain business language:
1. What the trend means in practical terms
2. How strong/reliable it is
3. What action it might suggest
"""
    return _complete(_ANALYST_SYSTEM, prompt, use_cache=True)


def answer_question(question: str, summary: dict, sample_rows: str) -> str:
    """
    Answer a natural-language question about the dataset.
    Returns an answer and optionally suggests pandas code.
    """
    prompt = f"""Dataset Summary:
{json.dumps(summary, indent=2, default=str)[:2500]}

Sample Data (first few rows):
{sample_rows[:400]}

User Question: {question}

Answer the user's question in plain, conversational English using only the data summary above.
Do NOT include any code. Do NOT use technical jargon. Give a direct, friendly answer."""
    return _complete(_CHAT_SYSTEM, prompt, use_cache=False, max_tokens=600)


def generate_pandas_code(question: str, columns: list[str], dtypes: dict) -> str:
    """
    Generate safe pandas code to answer an analytical question.
    """
    dtype_str = "\n".join(f"  {col}: {dt}" for col, dt in dtypes.items())
    prompt = f"""The user wants to know: {question}

DataFrame columns and dtypes:
{dtype_str}

Write pandas code that stores the answer in `result`."""
    return _complete(_CODE_SYSTEM, prompt, use_cache=False, max_tokens=400)


def generate_report_summary(full_analysis: dict, insights_text: str) -> str:
    """
    Create a concise executive summary for the downloadable report.
    """
    prompt = f"""Analysis Report:
{json.dumps(full_analysis, indent=2, default=str)[:2000]}

LLM Insights Already Generated:
{insights_text[:1000]}

Write a concise (300-word) executive summary suitable for a business report.
Include: dataset overview, top 3 findings, and 2 recommendations.
Use plain English, no jargon."""
    return _complete(_ANALYST_SYSTEM, prompt, use_cache=True)
