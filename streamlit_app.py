# -*- coding: utf-8 -*-
"""
FRED Macro Investment Timing Dashboard (Streamlit)
- GitHub + Streamlit Community Cloud friendly
- iPhone/web access
- Additional Methodology tab:
    1) how each parameter is calculated
    2) detailed meaning / interpretation
    3) regime scoring rules
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots


# =========================================================
# 1) PAGE / USER SETTINGS
# =========================================================
st.set_page_config(
    page_title="FRED Macro Investment Timing Dashboard",
    page_icon="📊",
    layout="wide",
)

APP_TITLE = "FRED Macro Investment Timing Dashboard"
DEFAULT_LOOKBACK_YEARS = 5
REQUEST_TIMEOUT = 20
MAX_RETRIES = 4
BACKOFF_SEC = 1.5
REQUEST_GAP_SEC = 0.35
CACHE_TTL_SEC = 1800  # 30 min

SERIES_META = {
    "WALCL": {"name": "Fed Balance Sheet", "category": "Liquidity", "unit": "Million USD"},
    "WTREGEN": {"name": "Treasury General Account (TGA)", "category": "Liquidity", "unit": "Million USD"},
    "RRPONTSYD": {"name": "Reverse Repo (RRP)", "category": "Liquidity", "unit": "Billion USD"},
    "EFFR": {"name": "Effective Fed Funds Rate", "category": "Rates", "unit": "%"},
    "SOFR": {"name": "SOFR", "category": "Rates", "unit": "%"},
    "DGS10": {"name": "US 10Y Treasury Yield", "category": "Rates", "unit": "%"},
    "DGS2": {"name": "US 2Y Treasury Yield", "category": "Rates", "unit": "%"},
    "T10Y2Y": {"name": "10Y-2Y Yield Spread", "category": "Rates", "unit": "%p"},
    "BAMLH0A0HYM2": {"name": "High Yield OAS", "category": "Credit", "unit": "%"},
    "STLFSI4": {"name": "St. Louis Fed Financial Stress Index", "category": "Credit", "unit": "Index"},
    "UNRATE": {"name": "Unemployment Rate", "category": "Macro", "unit": "%"},
    "INDPRO": {"name": "Industrial Production", "category": "Macro", "unit": "Index"},
    "CPIAUCSL": {"name": "CPI", "category": "Inflation", "unit": "Index"},
    "PCEPI": {"name": "PCE Price Index", "category": "Inflation", "unit": "Index"},
}

SERIES_ORDER = list(SERIES_META.keys())

THEME = {
    "bg": "#0f172a",
    "panel": "#111827",
    "panel_2": "#1f2937",
    "text": "#e5e7eb",
    "muted": "#9ca3af",
    "green": "#22c55e",
    "yellow": "#f59e0b",
    "red": "#ef4444",
    "blue": "#60a5fa",
    "border": "#374151",
}


# =========================================================
# 2) STYLES
# =========================================================
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {THEME["bg"]};
        color: {THEME["text"]};
    }}
    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }}
    .metric-card {{
        background: {THEME["panel"]};
        border: 1px solid {THEME["border"]};
        border-radius: 14px;
        padding: 16px;
        min-height: 120px;
    }}
    .metric-title {{
        color: {THEME["muted"]};
        font-size: 13px;
        margin-bottom: 8px;
    }}
    .metric-value {{
        font-size: 28px;
        font-weight: 700;
        line-height: 1.1;
    }}
    .metric-sub {{
        color: {THEME["muted"]};
        font-size: 12px;
        margin-top: 6px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3) API KEY / FRED
# =========================================================
def load_api_key() -> str:
    try:
        if "FRED_API_KEY" in st.secrets:
            key = str(st.secrets["FRED_API_KEY"]).strip()
            if key:
                return key
    except Exception:
        pass

    env_key = os.getenv("FRED_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    local_path = os.path.join(os.getcwd(), "API_KEY.txt")
    if os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8") as f:
            key = f.read().strip()
            if key:
                return key

    raise FileNotFoundError(
        "FRED API key not found. Use Streamlit Secrets, env var FRED_API_KEY, or local API_KEY.txt."
    )


def fetch_fred_series(
    series_id: str,
    api_key: str,
    observation_start: str = "2000-01-01",
    timeout: int = REQUEST_TIMEOUT,
    max_retries: int = MAX_RETRIES,
    backoff_sec: float = BACKOFF_SEC,
) -> pd.DataFrame:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": observation_start,
        "sort_order": "asc",
    }

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            data = r.json()

            if "observations" not in data:
                raise ValueError(f"Unexpected FRED response for {series_id}: {data}")

            obs = data["observations"]
            if not obs:
                return pd.DataFrame(columns=["date", series_id])

            df = pd.DataFrame(obs)[["date", "value"]].copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["date"]).rename(columns={"value": series_id})
            df = df.sort_values("date").reset_index(drop=True)
            return df

        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(backoff_sec * attempt)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(backoff_sec * attempt)

    raise last_error


def fetch_all_series(api_key: str, start_date: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}

    for sid in SERIES_ORDER:
        try:
            out[sid] = fetch_fred_series(
                sid,
                api_key,
                observation_start=start_date,
                timeout=REQUEST_TIMEOUT,
                max_retries=MAX_RETRIES,
                backoff_sec=BACKOFF_SEC,
            )
        except Exception:
            out[sid] = pd.DataFrame(columns=["date", sid])

        time.sleep(REQUEST_GAP_SEC)

    return out


# =========================================================
# 4) DATA PROCESSING
# =========================================================
def to_weekly_wed(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", value_col])

    x = df.copy().set_index("date").sort_index()
    weekly = x.resample("W-WED").last()
    weekly = weekly.reset_index()
    return weekly


def combine_weekly_data(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    weekly_list = []

    for sid, df in raw.items():
        if df.empty:
            continue
        wk = to_weekly_wed(df, sid)
        weekly_list.append(wk)

    if not weekly_list:
        return pd.DataFrame()

    merged = weekly_list[0].copy()
    for wk in weekly_list[1:]:
        merged = pd.merge(merged, wk, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)

    full_idx = pd.date_range(
        start=merged["date"].min(),
        end=merged["date"].max(),
        freq="W-WED",
    )

    merged = (
        merged.set_index("date")
        .reindex(full_idx)
        .rename_axis("date")
        .reset_index()
    )

    for sid in SERIES_ORDER:
        if sid in merged.columns:
            merged[sid] = merged[sid].ffill()

    return merged


def compute_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    if all(col in x.columns for col in ["WALCL", "WTREGEN", "RRPONTSYD"]):
        x["WALCL_B"] = x["WALCL"] / 1000.0
        x["WTREGEN_B"] = x["WTREGEN"] / 1000.0
        x["NET_LIQUIDITY_B"] = x["WALCL_B"] - x["WTREGEN_B"] - x["RRPONTSYD"]

    if "CPIAUCSL" in x.columns:
        x["CPI_YOY"] = x["CPIAUCSL"].pct_change(52) * 100.0

    if "PCEPI" in x.columns:
        x["PCE_YOY"] = x["PCEPI"].pct_change(52) * 100.0

    if "INDPRO" in x.columns:
        x["INDPRO_YOY"] = x["INDPRO"].pct_change(52) * 100.0

    if "NET_LIQUIDITY_B" in x.columns:
        x["NET_LIQ_13W_CHG"] = x["NET_LIQUIDITY_B"] - x["NET_LIQUIDITY_B"].shift(13)
        x["NET_LIQ_26W_CHG"] = x["NET_LIQUIDITY_B"] - x["NET_LIQUIDITY_B"].shift(26)

    if "BAMLH0A0HYM2" in x.columns:
        x["HY_SPREAD"] = x["BAMLH0A0HYM2"]

    if "T10Y2Y" in x.columns:
        x["YC_USED"] = x["T10Y2Y"]

    if all(col in x.columns for col in ["DGS10", "DGS2"]):
        x["T10Y2Y_CALC"] = x["DGS10"] - x["DGS2"]
        if "YC_USED" in x.columns:
            x["YC_USED"] = x["YC_USED"].fillna(x["T10Y2Y_CALC"])
        else:
            x["YC_USED"] = x["T10Y2Y_CALC"]

    return x


def filter_lookback(df: pd.DataFrame, years: int) -> pd.DataFrame:
    if df.empty:
        return df
    cutoff = df["date"].max() - pd.DateOffset(years=years)
    return df[df["date"] >= cutoff].copy()


# =========================================================
# 5) SIGNAL / INTERPRETATION
# =========================================================
def classify_signal(latest: pd.Series) -> tuple[float, str, str]:
    score = 0.0

    net_liq_chg = latest.get("NET_LIQ_13W_CHG", np.nan)
    yc = latest.get("YC_USED", np.nan)
    hy = latest.get("HY_SPREAD", np.nan)
    stress = latest.get("STLFSI4", np.nan)
    indpro = latest.get("INDPRO_YOY", np.nan)
    unrate = latest.get("UNRATE", np.nan)

    if pd.notna(net_liq_chg) and net_liq_chg > 0:
        score += 1.0

    if pd.notna(yc) and yc > 0:
        score += 1.0

    if pd.notna(hy):
        if hy < 4.0:
            score += 1.0
        elif hy <= 6.0:
            score += 0.5

    if pd.notna(stress):
        if stress < 0:
            score += 1.0
        elif stress <= 1.0:
            score += 0.5

    if pd.notna(indpro) and indpro > 0:
        score += 1.0

    if pd.notna(unrate) and unrate < 4.5:
        score += 1.0

    if score >= 5:
        regime = "RISK ON"
        color = THEME["green"]
    elif score >= 3:
        regime = "NEUTRAL"
        color = THEME["yellow"]
    else:
        regime = "RISK OFF"
        color = THEME["red"]

    return score, regime, color


def interpret_regime(score: float, regime: str) -> tuple[str, str, str]:
    if regime == "RISK ON":
        return (
            "Investment Stance: Aggressive",
            "Risk On means investors are favoring risk assets such as stocks. "
            "It does NOT mean danger warning; it means the macro backdrop is supportive.",
            "Suggested Action: Increase equity exposure / favor growth assets",
        )
    elif regime == "NEUTRAL":
        return (
            "Investment Stance: Balanced",
            "Mixed backdrop. Some indicators are supportive, but not all. "
            "A balanced allocation and phased entry may be more appropriate.",
            "Suggested Action: Partial buying / diversified allocation",
        )
    else:
        return (
            "Investment Stance: Defensive",
            "Risk Off means investors prefer safer assets such as cash, bonds, or gold. "
            "Macro conditions are not friendly for aggressive risk-taking.",
            "Suggested Action: Reduce risk / hold more cash or defensive assets",
        )


def build_status_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Indicator", "Latest", "Rule", "Status"])

    latest = df.iloc[-1]
    rows = []

    def add_row(indicator, latest_value, rule, status):
        rows.append(
            {
                "Indicator": indicator,
                "Latest": latest_value,
                "Rule": rule,
                "Status": status,
            }
        )

    nl = latest.get("NET_LIQ_13W_CHG", np.nan)
    add_row(
        "Net Liquidity 13W Change",
        f"{nl:,.1f} B" if pd.notna(nl) else "N/A",
        "> 0 = bullish",
        "Bullish" if pd.notna(nl) and nl > 0 else "Bearish/Neutral",
    )

    yc = latest.get("YC_USED", np.nan)
    add_row(
        "10Y-2Y Curve",
        f"{yc:.2f}" if pd.notna(yc) else "N/A",
        "> 0 = healthy",
        "Healthy" if pd.notna(yc) and yc > 0 else "Inverted/Flat",
    )

    hy = latest.get("HY_SPREAD", np.nan)
    hy_status = (
        "Benign" if pd.notna(hy) and hy < 4
        else "Stressed" if pd.notna(hy) and hy > 6
        else "Neutral"
    )
    add_row(
        "HY OAS",
        f"{hy:.2f}%" if pd.notna(hy) else "N/A",
        "< 4 good / > 6 bad",
        hy_status,
    )

    stress = latest.get("STLFSI4", np.nan)
    stress_status = (
        "Low" if pd.notna(stress) and stress < 0
        else "High" if pd.notna(stress) and stress > 1
        else "Moderate"
    )
    add_row(
        "Financial Stress",
        f"{stress:.2f}" if pd.notna(stress) else "N/A",
        "< 0 good / > 1 bad",
        stress_status,
    )

    indpro = latest.get("INDPRO_YOY", np.nan)
    add_row(
        "Industrial Production YoY",
        f"{indpro:.1f}%" if pd.notna(indpro) else "N/A",
        "> 0 = expansion",
        "Expansion" if pd.notna(indpro) and indpro > 0 else "Weakening",
    )

    un = latest.get("UNRATE", np.nan)
    add_row(
        "Unemployment",
        f"{un:.1f}%" if pd.notna(un) else "N/A",
        "< 4.5 favorable",
        "Firm labor" if pd.notna(un) and un < 4.5 else "Softening",
    )

    return pd.DataFrame(rows)


# =========================================================
# 6) METHODOLOGY TABLES
# =========================================================
def build_parameter_definition_table() -> pd.DataFrame:
    rows = [
        {
            "Parameter": "WALCL",
            "Meaning": "Federal Reserve total assets (Fed balance sheet).",
            "How Calculated": "Raw FRED series value",
            "Interpretation": "Higher balance sheet can imply more liquidity support.",
        },
        {
            "Parameter": "WTREGEN",
            "Meaning": "Treasury General Account balance.",
            "How Calculated": "Raw FRED series value",
            "Interpretation": "When TGA rises, liquidity is often drained from markets.",
        },
        {
            "Parameter": "RRPONTSYD",
            "Meaning": "Reverse Repo balance.",
            "How Calculated": "Raw FRED series value",
            "Interpretation": "Higher RRP can absorb liquidity from the financial system.",
        },
        {
            "Parameter": "WALCL_B",
            "Meaning": "Fed balance sheet in billions.",
            "How Calculated": "WALCL / 1000",
            "Interpretation": "Unit-converted version of WALCL.",
        },
        {
            "Parameter": "WTREGEN_B",
            "Meaning": "TGA in billions.",
            "How Calculated": "WTREGEN / 1000",
            "Interpretation": "Unit-converted version of TGA.",
        },
        {
            "Parameter": "NET_LIQUIDITY_B",
            "Meaning": "Net system liquidity proxy.",
            "How Calculated": "WALCL_B - WTREGEN_B - RRPONTSYD",
            "Interpretation": "Higher net liquidity is generally supportive for risk assets.",
        },
        {
            "Parameter": "NET_LIQ_13W_CHG",
            "Meaning": "13-week change in net liquidity.",
            "How Calculated": "NET_LIQUIDITY_B - NET_LIQUIDITY_B.shift(13)",
            "Interpretation": "Positive means liquidity improved over the last quarter.",
        },
        {
            "Parameter": "NET_LIQ_26W_CHG",
            "Meaning": "26-week change in net liquidity.",
            "How Calculated": "NET_LIQUIDITY_B - NET_LIQUIDITY_B.shift(26)",
            "Interpretation": "Medium-term liquidity trend.",
        },
        {
            "Parameter": "EFFR",
            "Meaning": "Effective Fed Funds Rate.",
            "How Calculated": "Raw FRED series value",
            "Interpretation": "Policy-sensitive short-term funding rate.",
        },
        {
            "Parameter": "SOFR",
            "Meaning": "Secured Overnight Financing Rate.",
            "How Calculated": "Raw FRED series value",
            "Interpretation": "Broad overnight funding benchmark.",
        },
        {
            "Parameter": "DGS10",
            "Meaning": "US 10-year Treasury yield.",
            "How Calculated": "Raw FRED series value",
            "Interpretation": "Represents long-term rate expectations and growth/inflation outlook.",
        },
        {
            "Parameter": "DGS2",
            "Meaning": "US 2-year Treasury yield.",
            "How Calculated": "Raw FRED series value",
            "Interpretation": "More sensitive to near-term policy expectations.",
        },
        {
            "Parameter": "T10Y2Y",
            "Meaning": "Official 10Y minus 2Y spread from FRED.",
            "How Calculated": "Raw FRED series value",
            "Interpretation": "Positive curve usually implies healthier macro expectations.",
        },
        {
            "Parameter": "T10Y2Y_CALC",
            "Meaning": "Fallback calculated yield curve.",
            "How Calculated": "DGS10 - DGS2",
            "Interpretation": "Used when official T10Y2Y is missing.",
        },
        {
            "Parameter": "YC_USED",
            "Meaning": "Yield curve actually used in the dashboard.",
            "How Calculated": "T10Y2Y if available, else DGS10 - DGS2",
            "Interpretation": "Positive = better macro backdrop, negative = inversion risk.",
        },
        {
            "Parameter": "BAMLH0A0HYM2 / HY_SPREAD",
            "Meaning": "US High Yield Option-Adjusted Spread.",
            "How Calculated": "Raw FRED series value",
            "Interpretation": "Lower spread = benign credit conditions, higher spread = credit stress.",
        },
        {
            "Parameter": "STLFSI4",
            "Meaning": "St. Louis Fed Financial Stress Index.",
            "How Calculated": "Raw FRED series value",
            "Interpretation": "Below 0 is relatively calm, above 1 suggests stress.",
        },
        {
            "Parameter": "UNRATE",
            "Meaning": "Unemployment rate.",
            "How Calculated": "Raw FRED series value",
            "Interpretation": "Lower unemployment generally signals a stronger labor market.",
        },
        {
            "Parameter": "INDPRO",
            "Meaning": "Industrial Production Index.",
            "How Calculated": "Raw FRED series value",
            "Interpretation": "Proxy for production activity in the economy.",
        },
        {
            "Parameter": "INDPRO_YOY",
            "Meaning": "Industrial production year-over-year growth.",
            "How Calculated": "INDPRO.pct_change(52) * 100",
            "Interpretation": "Positive = expansion, negative = weakening production trend.",
        },
        {
            "Parameter": "CPIAUCSL",
            "Meaning": "Consumer Price Index.",
            "How Calculated": "Raw FRED series value",
            "Interpretation": "Broad measure of consumer inflation.",
        },
        {
            "Parameter": "CPI_YOY",
            "Meaning": "CPI year-over-year inflation.",
            "How Calculated": "CPIAUCSL.pct_change(52) * 100",
            "Interpretation": "Tracks inflation trend versus one year earlier.",
        },
        {
            "Parameter": "PCEPI",
            "Meaning": "PCE Price Index.",
            "How Calculated": "Raw FRED series value",
            "Interpretation": "Fed-preferred inflation measure.",
        },
        {
            "Parameter": "PCE_YOY",
            "Meaning": "PCE year-over-year inflation.",
            "How Calculated": "PCEPI.pct_change(52) * 100",
            "Interpretation": "Useful for assessing inflation pressure over time.",
        },
    ]
    return pd.DataFrame(rows)


def build_signal_rule_table() -> pd.DataFrame:
    rows = [
        {
            "Signal Component": "Net Liquidity 13W Change",
            "Rule": "> 0",
            "Score Impact": "+1.0",
            "Meaning": "Liquidity improved over the last 13 weeks.",
        },
        {
            "Signal Component": "Yield Curve (YC_USED)",
            "Rule": "> 0",
            "Score Impact": "+1.0",
            "Meaning": "Positive yield curve is treated as healthier macro structure.",
        },
        {
            "Signal Component": "HY Spread",
            "Rule": "< 4.0",
            "Score Impact": "+1.0",
            "Meaning": "Credit market stress is low.",
        },
        {
            "Signal Component": "HY Spread",
            "Rule": "4.0 to 6.0",
            "Score Impact": "+0.5",
            "Meaning": "Credit conditions are neutral.",
        },
        {
            "Signal Component": "Financial Stress",
            "Rule": "< 0",
            "Score Impact": "+1.0",
            "Meaning": "Stress environment is calm.",
        },
        {
            "Signal Component": "Financial Stress",
            "Rule": "0 to 1.0",
            "Score Impact": "+0.5",
            "Meaning": "Moderate stress environment.",
        },
        {
            "Signal Component": "Industrial Production YoY",
            "Rule": "> 0",
            "Score Impact": "+1.0",
            "Meaning": "Economic production is expanding.",
        },
        {
            "Signal Component": "Unemployment Rate",
            "Rule": "< 4.5",
            "Score Impact": "+1.0",
            "Meaning": "Labor market is considered supportive.",
        },
        {
            "Signal Component": "Total Score",
            "Rule": ">= 5",
            "Score Impact": "RISK ON",
            "Meaning": "Supportive macro backdrop for risk assets.",
        },
        {
            "Signal Component": "Total Score",
            "Rule": ">= 3 and < 5",
            "Score Impact": "NEUTRAL",
            "Meaning": "Mixed environment.",
        },
        {
            "Signal Component": "Total Score",
            "Rule": "< 3",
            "Score Impact": "RISK OFF",
            "Meaning": "Defensive environment.",
        },
    ]
    return pd.DataFrame(rows)


# =========================================================
# 7) FIGURES
# =========================================================
def apply_dark_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        paper_bgcolor=THEME["panel"],
        plot_bgcolor=THEME["panel"],
        font=dict(color=THEME["text"]),
        margin=dict(l=50, r=30, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
        height=420,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=THEME["border"])
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
        linecolor=THEME["border"],
    )
    return fig


def fig_liquidity(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["NET_LIQUIDITY_B"],
            name="Net Liquidity (B USD)",
            mode="lines",
            line=dict(width=2),
        ),
        secondary_y=False,
    )

    if "WALCL_B" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["WALCL_B"],
                name="Fed Balance Sheet (B)",
                mode="lines",
                line=dict(width=1),
                opacity=0.6,
            ),
            secondary_y=True,
        )

    if "WTREGEN_B" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["WTREGEN_B"],
                name="TGA (B)",
                mode="lines",
                line=dict(width=1),
                opacity=0.6,
            ),
            secondary_y=True,
        )

    if "RRPONTSYD" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["RRPONTSYD"],
                name="RRP (B)",
                mode="lines",
                line=dict(width=1),
                opacity=0.6,
            ),
            secondary_y=True,
        )

    fig.update_yaxes(title_text="Net Liquidity (B USD)", secondary_y=False)
    fig.update_yaxes(title_text="Components (B USD)", secondary_y=True)
    return apply_dark_layout(fig, "Liquidity: Net Liquidity / WALCL / TGA / RRP")


def fig_rates(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for col, name in [("EFFR", "EFFR"), ("SOFR", "SOFR"), ("DGS10", "US 10Y"), ("DGS2", "US 2Y")]:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df[col],
                    name=name,
                    mode="lines",
                    line=dict(width=2),
                ),
                secondary_y=False,
            )

    if "YC_USED" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["YC_USED"],
                name="10Y-2Y Spread (used)",
                mode="lines",
                line=dict(width=2, dash="dot"),
            ),
            secondary_y=True,
        )
        fig.add_hline(y=0, line_dash="dash", opacity=0.5, secondary_y=True)

    fig.update_yaxes(title_text="Rates (%)", secondary_y=False)
    fig.update_yaxes(title_text="Curve Spread (%p)", secondary_y=True)
    return apply_dark_layout(fig, "Rates: EFFR / SOFR / 10Y / 2Y / 10Y-2Y")


def fig_credit(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    has_any = False

    if "BAMLH0A0HYM2" in df.columns and df["BAMLH0A0HYM2"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["BAMLH0A0HYM2"],
                name="HY OAS",
                mode="lines",
                line=dict(width=2),
            ),
            secondary_y=False,
        )
        fig.add_hline(y=4, line_dash="dash", opacity=0.4, secondary_y=False)
        fig.add_hline(y=6, line_dash="dash", opacity=0.4, secondary_y=False)
        has_any = True

    if "STLFSI4" in df.columns and df["STLFSI4"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["STLFSI4"],
                name="Financial Stress",
                mode="lines",
                line=dict(width=2, dash="dot"),
            ),
            secondary_y=True,
        )
        fig.add_hline(y=0, line_dash="dash", opacity=0.4, secondary_y=True)
        fig.add_hline(y=1, line_dash="dash", opacity=0.4, secondary_y=True)
        has_any = True

    if not has_any:
        fig.add_annotation(text="Credit data unavailable", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)

    fig.update_yaxes(title_text="HY OAS (%)", secondary_y=False)
    fig.update_yaxes(title_text="Stress Index", secondary_y=True)
    return apply_dark_layout(fig, "Credit: High Yield Spread / Financial Stress")


def fig_macro(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    has_any = False

    if "INDPRO_YOY" in df.columns and df["INDPRO_YOY"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["INDPRO_YOY"],
                name="Industrial Production YoY",
                mode="lines",
                line=dict(width=2),
            ),
            secondary_y=False,
        )
        fig.add_hline(y=0, line_dash="dash", opacity=0.4, secondary_y=False)
        has_any = True

    if "UNRATE" in df.columns and df["UNRATE"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["UNRATE"],
                name="Unemployment Rate",
                mode="lines",
                line=dict(width=2, dash="dot"),
            ),
            secondary_y=True,
        )
        has_any = True

    if not has_any:
        fig.add_annotation(text="Macro data unavailable", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)

    fig.update_yaxes(title_text="Industrial Production YoY (%)", secondary_y=False)
    fig.update_yaxes(title_text="Unemployment (%)", secondary_y=True)
    return apply_dark_layout(fig, "Macro: Industrial Production / Unemployment")


def fig_inflation(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    has_any = False

    if "CPI_YOY" in df.columns and df["CPI_YOY"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["CPI_YOY"],
                name="CPI YoY",
                mode="lines",
                line=dict(width=2),
            )
        )
        has_any = True

    if "PCE_YOY" in df.columns and df["PCE_YOY"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["PCE_YOY"],
                name="PCE YoY",
                mode="lines",
                line=dict(width=2, dash="dot"),
            )
        )
        has_any = True

    if not has_any:
        fig.add_annotation(text="Inflation data unavailable", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)

    fig.add_hline(y=2.0, line_dash="dash", opacity=0.4)
    fig.update_yaxes(title_text="YoY Inflation (%)")
    return apply_dark_layout(fig, "Inflation: CPI YoY / PCE YoY")


def fig_score_gauge(score: float, regime: str, color: str) -> go.Figure:
    subtitle_map = {
        "RISK ON": "Supportive for stocks / risk assets",
        "NEUTRAL": "Mixed macro backdrop",
        "RISK OFF": "Defensive macro backdrop",
    }

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": " / 6"},
            title={
                "text": f"Market Regime: {regime}<br><span style='font-size:14px;color:#9ca3af'>{subtitle_map.get(regime, '')}</span>"
            },
            gauge={
                "axis": {"range": [0, 6]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 2.99], "color": "rgba(239,68,68,0.25)"},
                    {"range": [3, 4.99], "color": "rgba(245,158,11,0.25)"},
                    {"range": [5, 6], "color": "rgba(34,197,94,0.25)"},
                ],
            },
        )
    )
    fig.update_layout(
        paper_bgcolor=THEME["panel"],
        font=dict(color=THEME["text"]),
        margin=dict(l=20, r=20, t=80, b=20),
        height=360,
    )
    return fig


# =========================================================
# 8) CACHE / LOAD
# =========================================================
@st.cache_data(ttl=CACHE_TTL_SEC, show_spinner=True)
def load_dashboard_data(fetch_years: int = 20):
    api_key = load_api_key()
    start_dt = datetime.today() - timedelta(days=int((fetch_years + 3) * 365.25))
    start_date = start_dt.strftime("%Y-%m-%d")

    raw = fetch_all_series(api_key, start_date)
    merged = combine_weekly_data(raw)
    merged = compute_derived_fields(merged)

    return merged, datetime.now()


# =========================================================
# 9) UI HELPERS
# =========================================================
def metric_card(col, title, value, sub="", color=None):
    col.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value" style="color:{color or THEME["text"]};">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 10) UI
# =========================================================
st.title(APP_TITLE)

top1, top2 = st.columns([1, 1])
with top1:
    lookback_years = st.selectbox("Lookback (Years)", [3, 5, 10, 15, 20], index=1)

with top2:
    manual_refresh = st.button("Refresh now")

if manual_refresh:
    st.cache_data.clear()

try:
    data_cache, last_refresh_ts = load_dashboard_data(20)
except Exception as e:
    st.error(f"Initial data load failed: {e}")
    st.stop()

if data_cache.empty:
    st.warning("No data available.")
    st.stop()

plot_df = filter_lookback(data_cache, int(lookback_years))
latest = data_cache.iloc[-1]
score, regime, regime_color = classify_signal(latest)
stance_title, explanation, action = interpret_regime(score, regime)

st.caption(f"Last refresh: {last_refresh_ts:%Y-%m-%d %H:%M:%S}")

tabs = st.tabs(["Dashboard", "Methodology"])

with tabs[0]:
    net_liq = latest.get("NET_LIQUIDITY_B", np.nan)
    yc = latest.get("YC_USED", np.nan)
    hy = latest.get("HY_SPREAD", np.nan)
    stress = latest.get("STLFSI4", np.nan)

    hy_stress_text = "N/A"
    if pd.notna(hy) and pd.notna(stress):
        hy_stress_text = f"{hy:.2f}% / {stress:.2f}"
    elif pd.notna(hy):
        hy_stress_text = f"{hy:.2f}% / N/A"
    elif pd.notna(stress):
        hy_stress_text = f"N/A / {stress:.2f}"

    m1, m2, m3, m4, m5 = st.columns(5)
    metric_card(m1, "Market Regime", regime, f"Score {score:.1f}/6", regime_color)
    metric_card(m2, "Investment Stance", stance_title.replace("Investment Stance: ", ""), action)
    metric_card(m3, "Net Liquidity", f"{net_liq:,.0f} B" if pd.notna(net_liq) else "N/A", "WALCL - TGA - RRP")
    metric_card(m4, "10Y-2Y Curve", f"{yc:.2f}" if pd.notna(yc) else "N/A", "Positive is better")
    metric_card(m5, "HY Spread / Stress", hy_stress_text, "Credit risk check")

    left, right = st.columns([1, 2])

    with left:
        st.plotly_chart(fig_score_gauge(score, regime, regime_color), use_container_width=True)
        st.markdown(f"### {stance_title}")
        st.write(explanation)
        st.markdown(f"**{action}**")

    with right:
        st.markdown("### Investment Status Rules")
        status_df = build_status_table(data_cache)
        st.dataframe(status_df, use_container_width=True, hide_index=True)

    st.plotly_chart(fig_liquidity(plot_df), use_container_width=True)
    st.plotly_chart(fig_rates(plot_df), use_container_width=True)
    st.plotly_chart(fig_credit(plot_df), use_container_width=True)
    st.plotly_chart(fig_macro(plot_df), use_container_width=True)
    st.plotly_chart(fig_inflation(plot_df), use_container_width=True)

with tabs[1]:
    st.subheader("How each parameter is calculated")
    st.write("This tab explains the raw indicators, derived fields, and how they are used in the dashboard.")

    param_df = build_parameter_definition_table()
    st.dataframe(param_df, use_container_width=True, hide_index=True)

    st.subheader("Signal scoring methodology")
    rule_df = build_signal_rule_table()
    st.dataframe(rule_df, use_container_width=True, hide_index=True)

    st.subheader("Key formulas")
    st.latex(r"WALCL\_B = \frac{WALCL}{1000}")
    st.latex(r"WTREGEN\_B = \frac{WTREGEN}{1000}")
    st.latex(r"NET\_LIQUIDITY\_B = WALCL\_B - WTREGEN\_B - RRPONTSYD")
    st.latex(r"NET\_LIQ\_13W\_CHG = NET\_LIQUIDITY\_B(t) - NET\_LIQUIDITY\_B(t-13)")
    st.latex(r"NET\_LIQ\_26W\_CHG = NET\_LIQUIDITY\_B(t) - NET\_LIQUIDITY\_B(t-26)")
    st.latex(r"T10Y2Y\_CALC = DGS10 - DGS2")
    st.latex(r"CPI\_YOY = \left(\frac{CPIAUCSL(t)}{CPIAUCSL(t-52)} - 1\right)\times100")
    st.latex(r"PCE\_YOY = \left(\frac{PCEPI(t)}{PCEPI(t-52)} - 1\right)\times100")
    st.latex(r"INDPRO\_YOY = \left(\frac{INDPRO(t)}{INDPRO(t-52)} - 1\right)\times100")

    st.subheader("Interpretation notes")
    st.markdown(
        """
**1. Liquidity**
- `NET_LIQUIDITY_B` is a simplified market liquidity proxy.
- Rising net liquidity is generally supportive for equities and other risk assets.

**2. Yield Curve**
- `YC_USED` represents the 10Y minus 2Y Treasury spread.
- A positive curve is usually interpreted as healthier than an inverted curve.

**3. Credit Conditions**
- `HY_SPREAD` and `STLFSI4` are used to detect financial stress.
- Wider spreads and higher stress levels usually imply risk-off conditions.

**4. Growth / Macro**
- `INDPRO_YOY` is a production-growth proxy.
- `UNRATE` helps assess labor market resilience.

**5. Inflation**
- `CPI_YOY` and `PCE_YOY` track inflation pressure.
- They are displayed for context, but currently do not directly contribute to the regime score.

**6. Regime Score**
- The score is a rule-based framework from 0 to 6.
- It is not a forecast model, but a structured macro condition indicator.
"""
    )

    st.subheader("Detailed parameter guide")
    with st.expander("Liquidity parameters", expanded=False):
        st.markdown(
            """
- **WALCL**: Fed total assets. Often used as a rough proxy for central bank balance sheet expansion/contraction.
- **WTREGEN**: Treasury cash parked at the Fed. Rising TGA can drain liquidity from markets.
- **RRPONTSYD**: Reverse repo usage. High balances can reflect cash being absorbed out of the system.
- **NET_LIQUIDITY_B**: Simplified market liquidity estimate from the three series above.
"""
        )

    with st.expander("Rates parameters", expanded=False):
        st.markdown(
            """
- **EFFR**: Effective Fed Funds Rate, the realized overnight policy rate.
- **SOFR**: Secured Overnight Financing Rate, an important funding benchmark.
- **DGS10 / DGS2**: 10Y and 2Y Treasury yields.
- **YC_USED**: The yield curve measure used for scoring. Positive is treated as supportive.
"""
        )

    with st.expander("Credit parameters", expanded=False):
        st.markdown(
            """
- **HY_SPREAD**: High yield spread over Treasuries. A wider spread usually means higher credit risk.
- **STLFSI4**: Financial stress index. Higher readings suggest tighter financial conditions.
"""
        )

    with st.expander("Macro parameters", expanded=False):
        st.markdown(
            """
- **UNRATE**: Unemployment rate. Lower values often imply a stronger labor market.
- **INDPRO**: Industrial production level.
- **INDPRO_YOY**: Growth rate of industrial production over the last year.
"""
        )

    with st.expander("Inflation parameters", expanded=False):
        st.markdown(
            """
- **CPIAUCSL / CPI_YOY**: Consumer inflation level and yearly growth.
- **PCEPI / PCE_YOY**: Personal consumption expenditures price index and yearly growth.
- PCE is widely followed because it is closely watched by the Fed.
"""
        )
