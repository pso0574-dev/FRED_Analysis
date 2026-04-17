# streamlit_app.py
# ============================================================
# FRED Macro Risk Dashboard
# - Financial crisis monitoring dashboard using FRED
# - Focused on liquidity / rates / credit / macro / inflation
# - Includes current snapshot, risk interpretation, and charts
#
# Run:
#   streamlit run streamlit_app.py
#
# Install:
#   pip install streamlit pandas numpy plotly requests
#
# Optional:
#   Set FRED_API_KEY in environment variables
#   or create .streamlit/secrets.toml with:
#   FRED_API_KEY="YOUR_API_KEY"
# ============================================================

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="FRED Macro Risk Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 FRED Macro Risk Dashboard")
st.caption("Liquidity / Rates / Credit / Macro / Inflation / Crisis Risk Monitoring")

# ============================================================
# Sidebar
# ============================================================
st.sidebar.header("Settings")

LOOKBACK_YEARS = st.sidebar.slider("Lookback years", 1, 30, 20)
AUTO_REFRESH = st.sidebar.checkbox("Auto refresh every 30 min", value=False)
SHOW_NORMALIZED = st.sidebar.checkbox("Show normalized comparison chart", value=True)
SHOW_RAW_CHARTS = st.sidebar.checkbox("Show raw charts", value=True)
USE_MARKERS_FOR_LOW_FREQ = st.sidebar.checkbox("Show markers for low-frequency series", value=True)

if AUTO_REFRESH:
    st.sidebar.info("Enable browser refresh or use Streamlit rerun logic if needed.")

# ============================================================
# FRED API configuration
# ============================================================
FRED_API_KEY = None
try:
    if "FRED_API_KEY" in st.secrets:
        FRED_API_KEY = st.secrets["FRED_API_KEY"]
except Exception:
    pass

if not FRED_API_KEY:
    FRED_API_KEY = os.getenv("FRED_API_KEY", "")

BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
REQUEST_TIMEOUT = 20
MAX_RETRIES = 4
BACKOFF_SEC = 1.2

# ============================================================
# Series definitions
# ============================================================
SERIES_META = {
    "WALCL": {
        "name": "Fed Balance Sheet",
        "category": "Liquidity",
        "unit": "Million USD",
        "desc": "Federal Reserve total assets. Rapid expansion often signals stress support.",
    },
    "RRPONTSYD": {
        "name": "Reverse Repo",
        "category": "Liquidity",
        "unit": "Billion USD",
        "desc": "Overnight Reverse Repo usage. Reflects liquidity parking / drainage.",
    },
    "WTREGEN": {
        "name": "Treasury General Account",
        "category": "Liquidity",
        "unit": "Million USD",
        "desc": "US Treasury cash balance. Falling TGA can inject liquidity into the system.",
    },
    "DGS10": {
        "name": "10Y Treasury Yield",
        "category": "Rates",
        "unit": "%",
        "desc": "Long-term US Treasury yield.",
    },
    "DGS2": {
        "name": "2Y Treasury Yield",
        "category": "Rates",
        "unit": "%",
        "desc": "Short-term US Treasury yield, sensitive to policy expectations.",
    },
    "T10Y2Y": {
        "name": "10Y-2Y Yield Curve",
        "category": "Rates",
        "unit": "%",
        "desc": "Yield curve spread. Prolonged inversion often precedes recession.",
    },
    "BAMLH0A0HYM2": {
        "name": "High Yield Spread",
        "category": "Credit",
        "unit": "%",
        "desc": "High-yield corporate bond spread. Rising spread signals credit stress.",
    },
    "BAMLC0A0CM": {
        "name": "Investment Grade Spread",
        "category": "Credit",
        "unit": "%",
        "desc": "Investment-grade corporate bond spread.",
    },
    "STLFSI4": {
        "name": "Financial Stress Index",
        "category": "Stress",
        "unit": "Index",
        "desc": "Composite financial stress index. Higher values imply market stress.",
    },
    "INDPRO": {
        "name": "Industrial Production",
        "category": "Macro",
        "unit": "Index",
        "desc": "Industrial activity proxy.",
    },
    "UNRATE": {
        "name": "Unemployment Rate",
        "category": "Macro",
        "unit": "%",
        "desc": "US unemployment rate.",
    },
    "DFF": {
        "name": "Effective Fed Funds Rate",
        "category": "Policy",
        "unit": "%",
        "desc": "Effective federal funds rate.",
    },
    "SOFR": {
        "name": "SOFR",
        "category": "Policy",
        "unit": "%",
        "desc": "Secured Overnight Financing Rate.",
    },
    "CPIAUCSL": {
        "name": "CPI",
        "category": "Inflation",
        "unit": "Index",
        "desc": "Consumer Price Index.",
    },
    "POILWTIUSDM": {
        "name": "Global price of WTI Crude",
        "category": "Inflation",
        "unit": "USD/bbl",
        "desc": "Global price of WTI crude oil.",
    },
    "CPFF": {
        "name": "Commercial Paper Funding Facility",
        "category": "Stress",
        "unit": "Index",
        "desc": "Commercial paper funding stress-related series.",
    },
    "TEDRATE": {
        "name": "TED Spread",
        "category": "Stress",
        "unit": "%",
        "desc": "Bank funding stress proxy.",
    },
    "DRBLACBS": {
        "name": "Bank Lending Tightening",
        "category": "Credit",
        "unit": "%",
        "desc": "Net percentage of domestic banks tightening standards for C&I loans.",
    },
}

DEFAULT_SERIES = [
    "WALCL",
    "RRPONTSYD",
    "WTREGEN",
    "DGS10",
    "DGS2",
    "T10Y2Y",
    "BAMLH0A0HYM2",
    "BAMLC0A0CM",
    "STLFSI4",
    "INDPRO",
    "UNRATE",
    "DFF",
    "SOFR",
    "CPIAUCSL",
    "POILWTIUSDM",
    "CPFF",
    "TEDRATE",
    "DRBLACBS",
]

# ============================================================
# Helper functions
# ============================================================
def safe_float(x) -> float:
    try:
        if x == "." or x is None or x == "":
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def fred_request(params: Dict) -> Dict:
    last_error = None
    for i in range(MAX_RETRIES):
        try:
            res = requests.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
            res.raise_for_status()
            return res.json()
        except Exception as e:
            last_error = e
            time.sleep(BACKOFF_SEC * (i + 1))
    raise RuntimeError(f"FRED request failed after retries: {last_error}")


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_fred_series(series_id: str, start_date: str) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "file_type": "json",
        "observation_start": start_date,
        "sort_order": "asc",
    }
    if FRED_API_KEY:
        params["api_key"] = FRED_API_KEY

    data = fred_request(params)
    obs = data.get("observations", [])
    if not obs:
        return pd.DataFrame(columns=["date", "value"]).set_index("date")

    df = pd.DataFrame(obs)[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = df["value"].apply(safe_float)
    df = df.set_index("date").sort_index()
    return df


def latest_valid_value(series: pd.Series) -> Tuple[Optional[pd.Timestamp], float]:
    s = series.dropna()
    if s.empty:
        return None, np.nan
    return s.index[-1], float(s.iloc[-1])


def prev_valid_value(series: pd.Series, n: int = 1) -> float:
    s = series.dropna()
    if len(s) <= n:
        return np.nan
    return float(s.iloc[-1 - n])


def yoy_change(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 13:
        return np.nan
    current = s.iloc[-1]
    past = s.iloc[-13]
    if pd.isna(current) or pd.isna(past) or past == 0:
        return np.nan
    return (current / past - 1.0) * 100.0


def pct_change_recent(series: pd.Series, periods: int = 20) -> float:
    s = series.dropna()
    if len(s) <= periods:
        return np.nan
    prev = s.iloc[-1 - periods]
    curr = s.iloc[-1]
    if pd.isna(prev) or prev == 0 or pd.isna(curr):
        return np.nan
    return (curr / prev - 1.0) * 100.0


def diff_recent(series: pd.Series, periods: int = 20) -> float:
    s = series.dropna()
    if len(s) <= periods:
        return np.nan
    return float(s.iloc[-1] - s.iloc[-1 - periods])


def normalize_series(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return s
    min_v = s.min()
    max_v = s.max()
    if max_v == min_v:
        return pd.Series(0.0, index=s.index)
    return (s - min_v) / (max_v - min_v)


def annualized_inflation_from_cpi(cpi: pd.Series, months: int = 3) -> float:
    s = cpi.dropna()
    if len(s) <= months:
        return np.nan
    recent = s.iloc[-1]
    past = s.iloc[-1 - months]
    if recent <= 0 or past <= 0:
        return np.nan
    return ((recent / past) ** (12 / months) - 1) * 100.0


def classify_signal(series_id: str, value: float) -> Tuple[str, str]:
    """
    Returns:
        status in {"Low Risk", "Watch", "High Risk", "N/A"}
        brief interpretation
    """
    if pd.isna(value):
        return "N/A", "No recent data"

    if series_id == "BAMLH0A0HYM2":
        if value >= 6:
            return "High Risk", "Credit stress elevated"
        elif value >= 4:
            return "Watch", "Credit conditions weakening"
        else:
            return "Low Risk", "Credit spread contained"

    if series_id == "BAMLC0A0CM":
        if value >= 2.5:
            return "High Risk", "IG spread elevated"
        elif value >= 1.7:
            return "Watch", "IG spread rising"
        else:
            return "Low Risk", "IG spread stable"

    if series_id == "T10Y2Y":
        if value < -0.5:
            return "High Risk", "Deep curve inversion / recession signal"
        elif value < 0:
            return "Watch", "Yield curve inverted"
        else:
            return "Low Risk", "Curve normal or steepening"

    if series_id == "STLFSI4":
        if value >= 1.0:
            return "High Risk", "Financial stress elevated"
        elif value >= 0:
            return "Watch", "Stress above normal"
        else:
            return "Low Risk", "Stress below average"

    if series_id == "UNRATE":
        if value >= 5.0:
            return "High Risk", "Labor market weakening"
        elif value >= 4.3:
            return "Watch", "Unemployment drifting higher"
        else:
            return "Low Risk", "Labor market relatively firm"

    if series_id == "POILWTIUSDM":
        if value >= 100:
            return "High Risk", "Oil shock risk"
        elif value >= 80:
            return "Watch", "Inflation pressure from oil"
        else:
            return "Low Risk", "Oil not yet shock-level"

    if series_id == "TEDRATE":
        if value >= 1.0:
            return "High Risk", "Funding stress elevated"
        elif value >= 0.5:
            return "Watch", "Funding stress rising"
        else:
            return "Low Risk", "Funding stress contained"

    if series_id == "DRBLACBS":
        if value >= 30:
            return "High Risk", "Banks tightening aggressively"
        elif value >= 10:
            return "Watch", "Lending standards tightening"
        else:
            return "Low Risk", "Credit standards not severely tight"

    if series_id == "WALCL":
        return "Watch", "Use together with stress indicators"
    if series_id == "RRPONTSYD":
        return "Watch", "Interpret with liquidity context"
    if series_id == "WTREGEN":
        return "Watch", "Interpret with Treasury cash trends"
    if series_id == "DGS10":
        return "Watch", "Long rate level depends on growth/inflation mix"
    if series_id == "DGS2":
        return "Watch", "Short rate reflects policy expectations"
    if series_id == "DFF":
        return "Watch", "Policy restrictive if high for long"
    if series_id == "SOFR":
        return "Watch", "Short funding rate should stay orderly"
    if series_id == "INDPRO":
        return "Watch", "Watch trend and YoY changes"
    if series_id == "CPIAUCSL":
        return "Watch", "Use YoY / 3M annualized trend"
    if series_id == "CPFF":
        return "Watch", "Interpret as short-term funding stress proxy"

    return "Watch", "Check chart and trend"


def risk_color(status: str) -> str:
    if status == "High Risk":
        return "#ff4b4b"
    if status == "Watch":
        return "#f0ad4e"
    if status == "Low Risk":
        return "#2ca02c"
    return "#9e9e9e"


def is_low_frequency_series(series: pd.Series) -> bool:
    s = series.dropna()
    if len(s) < 3:
        return False

    diffs = pd.Series(s.index).diff().dropna().dt.days
    if diffs.empty:
        return False

    median_gap_days = diffs.median()
    return median_gap_days >= 5


def build_single_chart(df: pd.DataFrame, series_id: str, use_markers: bool = True) -> go.Figure:
    meta = SERIES_META[series_id]
    s = df[series_id].dropna()

    fig = go.Figure()

    if s.empty:
        fig.update_layout(
            title=f"{meta['name']} ({series_id})",
            xaxis_title="Date",
            yaxis_title=meta["unit"],
            height=360,
            margin=dict(l=30, r=20, t=50, b=30),
            annotations=[
                dict(
                    text="No valid data available",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=16),
                )
            ],
        )
        return fig

    low_freq = is_low_frequency_series(s)
    mode = "lines+markers" if (use_markers and low_freq) else "lines"

    fig.add_trace(
        go.Scatter(
            x=s.index,
            y=s.values,
            mode=mode,
            name=meta["name"],
            connectgaps=False,
            marker=dict(size=4) if mode == "lines+markers" else None,
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Value: %{y:,.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"{meta['name']} ({series_id})",
        xaxis_title="Date",
        yaxis_title=meta["unit"],
        height=360,
        margin=dict(l=30, r=20, t=50, b=30),
        hovermode="x unified",
    )
    return fig


def build_single_chart_with_range(
    df: pd.DataFrame,
    series_id: str,
    start_date: pd.Timestamp,
    chart_title: str,
    use_markers: bool = True,
) -> go.Figure:
    meta = SERIES_META[series_id]

    if series_id not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            title=chart_title,
            height=320,
            annotations=[
                dict(
                    text=f"{series_id} not found",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                )
            ],
        )
        return fig

    s = df.loc[df.index >= start_date, series_id].dropna()

    fig = go.Figure()

    if s.empty:
        fig.update_layout(
            title=chart_title,
            xaxis_title="Date",
            yaxis_title=meta["unit"],
            height=320,
            margin=dict(l=30, r=20, t=50, b=30),
            annotations=[
                dict(
                    text="No valid data available in this range",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=16),
                )
            ],
        )
        return fig

    low_freq = is_low_frequency_series(s)
    mode = "lines+markers" if (use_markers and low_freq) else "lines"

    fig.add_trace(
        go.Scatter(
            x=s.index,
            y=s.values,
            mode=mode,
            name=meta["name"],
            connectgaps=False,
            marker=dict(size=4) if mode == "lines+markers" else None,
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Value: %{y:,.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=chart_title,
        xaxis_title="Date",
        yaxis_title=meta["unit"],
        height=320,
        margin=dict(l=30, r=20, t=50, b=30),
        hovermode="x unified",
    )
    return fig


def build_normalized_chart(
    df: pd.DataFrame,
    series_ids: List[str],
    title: str,
    use_markers: bool = False,
) -> go.Figure:
    fig = go.Figure()

    for sid in series_ids:
        if sid not in df.columns:
            continue

        s = df[sid].dropna()
        if s.empty:
            continue

        s_norm = normalize_series(s)
        low_freq = is_low_frequency_series(s)
        mode = "lines+markers" if (use_markers and low_freq) else "lines"

        fig.add_trace(
            go.Scatter(
                x=s_norm.index,
                y=s_norm.values,
                mode=mode,
                name=f"{SERIES_META[sid]['name']} ({sid})",
                connectgaps=False,
                marker=dict(size=3) if mode == "lines+markers" else None,
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Normalized: %{y:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Normalized (0-1)",
        height=420,
        margin=dict(l=30, r=20, t=50, b=30),
        legend=dict(orientation="h"),
        hovermode="x unified",
    )
    return fig


def compute_dashboard_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for sid in data.columns:
        series = data[sid]
        dt, latest = latest_valid_value(series)
        prev = prev_valid_value(series, 1)
        chg = latest - prev if pd.notna(latest) and pd.notna(prev) else np.nan
        yoy = yoy_change(series)
        status, note = classify_signal(sid, latest)

        row = {
            "Category": SERIES_META[sid]["category"],
            "Series ID": sid,
            "Indicator": SERIES_META[sid]["name"],
            "Latest Date": dt.date().isoformat() if dt is not None else "",
            "Latest": latest,
            "1-step Change": chg,
            "YoY %": yoy,
            "Status": status,
            "Interpretation": note,
            "Description": SERIES_META[sid]["desc"],
        }

        if sid == "CPIAUCSL":
            row["3M Annualized %"] = annualized_inflation_from_cpi(series, 3)
        else:
            row["3M Annualized %"] = np.nan

        if sid in ["WALCL", "RRPONTSYD", "WTREGEN", "INDPRO", "POILWTIUSDM"]:
            row["20-period % Change"] = pct_change_recent(series, 20)
        else:
            row["20-period % Change"] = np.nan

        if sid in ["BAMLH0A0HYM2", "BAMLC0A0CM", "STLFSI4", "TEDRATE", "T10Y2Y", "UNRATE"]:
            row["20-period Diff"] = diff_recent(series, 20)
        else:
            row["20-period Diff"] = np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    cat_order = ["Liquidity", "Rates", "Credit", "Stress", "Macro", "Policy", "Inflation"]
    out["CategoryOrder"] = out["Category"].map({c: i for i, c in enumerate(cat_order)})
    out = out.sort_values(["CategoryOrder", "Indicator"]).drop(columns=["CategoryOrder"])
    return out


def style_status_cell(val: str) -> str:
    color = risk_color(val)
    return f"background-color: {color}; color: white; font-weight: 600;"


def infer_overall_risk(table: pd.DataFrame) -> Tuple[str, str]:
    score = 0

    for sid, weight in [
        ("BAMLH0A0HYM2", 3),
        ("T10Y2Y", 2),
        ("STLFSI4", 3),
        ("TEDRATE", 2),
        ("DRBLACBS", 2),
        ("UNRATE", 2),
        ("POILWTIUSDM", 1),
    ]:
        row = table[table["Series ID"] == sid]
        if row.empty:
            continue
        status = row["Status"].iloc[0]
        if status == "High Risk":
            score += 2 * weight
        elif status == "Watch":
            score += 1 * weight

    if score >= 18:
        return "High Risk", "Broad multi-asset stress is building."
    elif score >= 9:
        return "Watch", "Several warning signals are active."
    else:
        return "Low Risk", "Risk signals are mixed or contained."


# ============================================================
# Data loading
# ============================================================
start_date = (datetime.today() - timedelta(days=365 * LOOKBACK_YEARS)).strftime("%Y-%m-%d")

with st.spinner("Loading FRED data..."):
    data_dict = {}
    failed_series = []

    for sid in DEFAULT_SERIES:
        try:
            df_sid = fetch_fred_series(sid, start_date)
            if not df_sid.empty:
                data_dict[sid] = df_sid["value"].rename(sid)
            else:
                failed_series.append(sid)
        except Exception:
            failed_series.append(sid)

    if not data_dict:
        st.error("No FRED data could be loaded. Check API key, internet connection, or series availability.")
        st.stop()

    data = pd.concat(data_dict.values(), axis=1).sort_index()

    if "T10Y2Y" not in data.columns and {"DGS10", "DGS2"}.issubset(data.columns):
        data["T10Y2Y"] = data["DGS10"] - data["DGS2"]

# ============================================================
# Top summary
# ============================================================
table = compute_dashboard_table(data)
overall_status, overall_msg = infer_overall_risk(table)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Overall Risk", overall_status)
c2.metric("Loaded Series", int(data.shape[1]))
c3.metric("Lookback", f"{LOOKBACK_YEARS}Y")

last_valid_date = data.dropna(how="all").index.max()
c4.metric("Last Data Date", last_valid_date.date().isoformat() if pd.notna(last_valid_date) else "N/A")

st.info(overall_msg)

if failed_series:
    st.warning("Some series could not be loaded: " + ", ".join(failed_series))

# ============================================================
# Key signal cards
# ============================================================
st.subheader("Key Crisis Signals")

key_signals = [
    "BAMLH0A0HYM2",
    "T10Y2Y",
    "STLFSI4",
    "TEDRATE",
    "DRBLACBS",
    "UNRATE",
    "POILWTIUSDM",
]

cols = st.columns(len(key_signals))
for col, sid in zip(cols, key_signals):
    row = table[table["Series ID"] == sid]
    if row.empty:
        col.write(f"**{sid}**")
        col.write("N/A")
        continue

    latest = row["Latest"].iloc[0]
    status = row["Status"].iloc[0]
    interp = row["Interpretation"].iloc[0]
    name = row["Indicator"].iloc[0]
    unit = SERIES_META[sid]["unit"]

    latest_text = f"{latest:.2f} {unit}" if pd.notna(latest) else "N/A"

    col.markdown(
        f"""
        <div style="padding:12px; border-radius:14px; background:{risk_color(status)}20; border:1px solid {risk_color(status)};">
            <div style="font-size:14px; font-weight:700;">{name}</div>
            <div style="font-size:22px; font-weight:800; margin-top:6px;">{latest_text}</div>
            <div style="font-size:13px; font-weight:700; color:{risk_color(status)}; margin-top:4px;">{status}</div>
            <div style="font-size:12px; margin-top:6px;">{interp}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Snapshot Table", "Risk Dashboard", "Normalized View", "Raw Charts", "Indicator Guide", "Long-Term Rates"]
)

# ============================================================
# Tab 1: Snapshot Table
# ============================================================
with tab1:
    st.subheader("Current Snapshot Table")

    display_cols = [
        "Category",
        "Series ID",
        "Indicator",
        "Latest Date",
        "Latest",
        "1-step Change",
        "YoY %",
        "3M Annualized %",
        "20-period % Change",
        "20-period Diff",
        "Status",
        "Interpretation",
    ]

    styled = (
        table[display_cols]
        .style.format(
            {
                "Latest": "{:,.2f}",
                "1-step Change": "{:,.2f}",
                "YoY %": "{:,.2f}",
                "3M Annualized %": "{:,.2f}",
                "20-period % Change": "{:,.2f}",
                "20-period Diff": "{:,.2f}",
            },
            na_rep="",
        )
        .map(style_status_cell, subset=["Status"])
    )

    st.dataframe(styled, use_container_width=True)

    st.download_button(
        label="Download snapshot CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="fred_macro_snapshot.csv",
        mime="text/csv",
    )

# ============================================================
# Tab 2: Risk Dashboard
# ============================================================
with tab2:
    st.subheader("Risk Dashboard by Category")

    for category in ["Liquidity", "Rates", "Credit", "Stress", "Macro", "Policy", "Inflation"]:
        sub = table[table["Category"] == category].copy()
        if sub.empty:
            continue

        st.markdown(f"### {category}")
        show = sub[
            ["Indicator", "Series ID", "Latest", "Status", "Interpretation", "Description"]
        ].copy()

        st.dataframe(
            show.style.format({"Latest": "{:,.2f}"}).map(style_status_cell, subset=["Status"]),
            use_container_width=True,
        )

# ============================================================
# Tab 3: Normalized View
# ============================================================
with tab3:
    st.subheader("Normalized Multi-Series Comparison")

    if SHOW_NORMALIZED:
        norm_groups = {
            "Liquidity Comparison": ["WALCL", "RRPONTSYD", "WTREGEN"],
            "Rates Comparison": ["DGS10", "DGS2", "T10Y2Y", "DFF", "SOFR"],
            "Credit and Stress Comparison": ["BAMLH0A0HYM2", "BAMLC0A0CM", "STLFSI4", "TEDRATE", "DRBLACBS"],
            "Macro and Inflation Comparison": ["INDPRO", "UNRATE", "CPIAUCSL", "POILWTIUSDM"],
        }

        for title, sids in norm_groups.items():
            existing = [sid for sid in sids if sid in data.columns]
            if not existing:
                continue
            fig = build_normalized_chart(
                data,
                existing,
                title,
                use_markers=USE_MARKERS_FOR_LOW_FREQ,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Enable normalized comparison in the sidebar.")

# ============================================================
# Tab 4: Raw Charts
# ============================================================
with tab4:
    st.subheader("Raw Charts")

    if SHOW_RAW_CHARTS:
        grouped = {}
        for sid in data.columns:
            cat = SERIES_META[sid]["category"]
            grouped.setdefault(cat, []).append(sid)

        category_order = ["Liquidity", "Rates", "Credit", "Stress", "Macro", "Policy", "Inflation"]
        available_categories = [c for c in category_order if c in grouped]

        selected_category = st.selectbox("Select category", available_categories)

        selected_series = st.multiselect(
            "Select indicators",
            grouped[selected_category],
            default=grouped[selected_category][: min(3, len(grouped[selected_category]))],
            format_func=lambda x: f"{SERIES_META[x]['name']} ({x})",
        )

        if selected_series:
            for sid in selected_series:
                series_non_na = data[sid].dropna()
                freq_note = "Low frequency" if is_low_frequency_series(series_non_na) else "Higher frequency"
                st.caption(
                    f"{SERIES_META[sid]['name']} ({sid}) | valid points: {len(series_non_na)} | {freq_note}"
                )
                fig = build_single_chart(
                    data,
                    sid,
                    use_markers=USE_MARKERS_FOR_LOW_FREQ,
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Choose at least one indicator.")
    else:
        st.info("Enable raw charts in the sidebar.")

# ============================================================
# Tab 5: Indicator Guide
# ============================================================
with tab5:
    st.subheader("Indicator Guide")
    guide_rows = []

    for sid, meta in SERIES_META.items():
        guide_rows.append(
            {
                "Category": meta["category"],
                "Series ID": sid,
                "Indicator": meta["name"],
                "Description": meta["desc"],
                "What to Check": {
                    "WALCL": "Rapid balance sheet expansion can signal emergency liquidity support.",
                    "RRPONTSYD": "Watch for large shifts showing liquidity absorption/release changes.",
                    "WTREGEN": "Falling TGA can add liquidity; rising TGA can drain liquidity.",
                    "DGS10": "Sharp declines may reflect recession fears; sharp rises may reflect inflation.",
                    "DGS2": "Sensitive to Fed expectations and policy pivot pricing.",
                    "T10Y2Y": "Negative values suggest inversion; deep/prolonged inversion is a warning.",
                    "BAMLH0A0HYM2": "Move above 4% watch; above 6% high risk.",
                    "BAMLC0A0CM": "Persistent widening suggests deteriorating corporate credit conditions.",
                    "STLFSI4": "Above 0 watch; above 1 elevated market stress.",
                    "INDPRO": "Look for rolling weakness and negative YoY trend.",
                    "UNRATE": "Rising unemployment often confirms recessionary pressure.",
                    "DFF": "High policy rate for longer increases refinancing pressure.",
                    "SOFR": "Unexpected jumps may reflect short-term funding pressure.",
                    "CPIAUCSL": "Check YoY and 3M annualized trend for inflation persistence.",
                    "POILWTIUSDM": "Oil spikes can reignite inflation and pressure growth.",
                    "CPFF": "Useful as a funding market stress context series.",
                    "TEDRATE": "Funding stress tends to show up when bank confidence weakens.",
                    "DRBLACBS": "Rising tightening means credit availability is worsening.",
                }.get(sid, ""),
                "Risk Rule of Thumb": {
                    "BAMLH0A0HYM2": ">= 6 high risk",
                    "T10Y2Y": "< 0 inversion, < -0.5 deep inversion",
                    "STLFSI4": ">= 1 high stress",
                    "UNRATE": ">= 5 labor weakness",
                    "POILWTIUSDM": ">= 100 oil shock",
                    "TEDRATE": ">= 1 funding stress",
                    "DRBLACBS": ">= 30 aggressive tightening",
                }.get(sid, "Interpret together with trend"),
            }
        )

    guide_df = pd.DataFrame(guide_rows).sort_values(["Category", "Indicator"])
    st.dataframe(guide_df, use_container_width=True)

# ============================================================
# Tab 6: Long-Term Rates
# ============================================================
with tab6:
    st.subheader("Long-Term Rate Change Dashboard")
    st.caption("10Y Treasury Yield (DGS10) shown across multiple horizons at once.")

    if "DGS10" not in data.columns:
        st.warning("DGS10 series is not available.")
    else:
        last_dt = data["DGS10"].dropna().index.max()

        if pd.isna(last_dt):
            st.warning("No valid DGS10 data available.")
        else:
            ranges = [
                ("1 Week", last_dt - pd.Timedelta(days=7)),
                ("1 Month", last_dt - pd.Timedelta(days=30)),
                ("1 Year", last_dt - pd.Timedelta(days=365)),
                ("5 Years", last_dt - pd.Timedelta(days=365 * 5)),
                ("20 Years", last_dt - pd.Timedelta(days=365 * 20)),
            ]

            col1, col2 = st.columns(2)
            target_cols = [col1, col2, col1, col2, col1]

            for (label, dt_start), col in zip(ranges, target_cols):
                with col:
                    fig = build_single_chart_with_range(
                        data,
                        "DGS10",
                        dt_start,
                        chart_title=f"10Y Treasury Yield ({label})",
                        use_markers=USE_MARKERS_FOR_LOW_FREQ,
                    )
                    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Bottom section: focused interpretation
# ============================================================
st.subheader("Quick Interpretation")

def get_val(sid: str) -> float:
    row = table[table["Series ID"] == sid]
    if row.empty:
        return np.nan
    return float(row["Latest"].iloc[0])


hy = get_val("BAMLH0A0HYM2")
yc = get_val("T10Y2Y")
fsi = get_val("STLFSI4")
ted = get_val("TEDRATE")
oil = get_val("POILWTIUSDM")
unr = get_val("UNRATE")

messages = []

if pd.notna(hy):
    if hy >= 6:
        messages.append("High-yield spreads are in a stressed zone.")
    elif hy >= 4:
        messages.append("High-yield spreads are elevated but not yet full-crisis level.")
    else:
        messages.append("High-yield spreads remain relatively contained.")

if pd.notna(yc):
    if yc < -0.5:
        messages.append("The yield curve is deeply inverted, which historically aligns with recession risk.")
    elif yc < 0:
        messages.append("The yield curve is inverted, still a cautionary signal.")
    else:
        messages.append("The yield curve is no longer inverted or is normalizing.")

if pd.notna(fsi):
    if fsi >= 1:
        messages.append("Financial stress is elevated.")
    elif fsi >= 0:
        messages.append("Financial stress is above normal but not extreme.")
    else:
        messages.append("Financial stress remains below historical average.")

if pd.notna(ted):
    if ted >= 1:
        messages.append("Bank funding stress is elevated.")
    elif ted >= 0.5:
        messages.append("Funding stress is rising.")
    else:
        messages.append("Funding stress appears contained.")

if pd.notna(oil):
    if oil >= 100:
        messages.append("Oil is at a shock-like level that can pressure inflation and growth.")
    elif oil >= 80:
        messages.append("Oil is high enough to keep inflation pressure alive.")
    else:
        messages.append("Oil is not currently in a shock zone.")

if pd.notna(unr):
    if unr >= 5:
        messages.append("Unemployment is at a level consistent with a weakening economy.")
    elif unr >= 4.3:
        messages.append("Unemployment is drifting higher and deserves monitoring.")
    else:
        messages.append("Labor market still looks relatively firm.")

for m in messages:
    st.write(f"- {m}")

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption(
    "Tip: For practical market timing, focus first on High Yield Spread, 10Y-2Y curve, Financial Stress Index, TED Spread, Bank Lending Tightening, and Unemployment trend."
)
