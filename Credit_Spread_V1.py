import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ============================================================
# Streamlit page config
# ============================================================
st.set_page_config(
    page_title="Credit & Liquidity Risk Dashboard",
    page_icon="📉",
    layout="wide",
)

st.title("📉 Credit & Liquidity Risk Dashboard")
st.caption("Monitor credit spreads, financial stress, yield curve, and liquidity conditions using FRED data.")

# ============================================================
# Config
# ============================================================
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

SERIES_META = {
    # ---------------- Credit ----------------
    "HY_OAS": {
        "ticker": "BAMLH0A0HYM2",
        "name": "US High Yield OAS",
        "category": "Credit",
        "unit": "%",
        "description": "High yield corporate bond spread. Sensitive risk appetite indicator.",
    },
    "BBB_OAS": {
        "ticker": "BAMLC0A4CBBB",
        "name": "BBB Corporate OAS",
        "category": "Credit",
        "unit": "%",
        "description": "BBB corporate bond spread. Shows stress in lower investment grade credit.",
    },
    "CORP_OAS": {
        "ticker": "BAMLC0A0CM",
        "name": "US Corporate OAS",
        "category": "Credit",
        "unit": "%",
        "description": "Broad US corporate bond spread.",
    },
    "FIN_STRESS": {
        "ticker": "STLFSI4",
        "name": "St. Louis Fed Financial Stress Index",
        "category": "Financial Conditions",
        "unit": "index",
        "description": "Measures stress across the US financial system.",
    },

    # ---------------- Rates ----------------
    "US10Y": {
        "ticker": "DGS10",
        "name": "US 10Y Treasury Yield",
        "category": "Rates",
        "unit": "%",
        "description": "US 10-year Treasury yield.",
    },
    "US2Y": {
        "ticker": "DGS2",
        "name": "US 2Y Treasury Yield",
        "category": "Rates",
        "unit": "%",
        "description": "US 2-year Treasury yield.",
    },
    "US3M": {
        "ticker": "DGS3MO",
        "name": "US 3M Treasury Yield",
        "category": "Rates",
        "unit": "%",
        "description": "US 3-month Treasury yield.",
    },
    "SPREAD_10Y2Y": {
        "ticker": "T10Y2Y",
        "name": "10Y - 2Y Treasury Spread",
        "category": "Yield Curve",
        "unit": "%p",
        "description": "10-year minus 2-year Treasury spread.",
    },
    "SPREAD_10Y3M": {
        "ticker": "T10Y3M",
        "name": "10Y - 3M Treasury Spread",
        "category": "Yield Curve",
        "unit": "%p",
        "description": "10-year minus 3-month Treasury spread.",
    },

    # ---------------- Liquidity ----------------
    "FED_BALANCE_SHEET": {
        "ticker": "WALCL",
        "name": "Fed Balance Sheet",
        "category": "Liquidity",
        "unit": "USD bn",
        "description": "Federal Reserve total assets.",
    },
    "RRP": {
        "ticker": "RRPONTSYD",
        "name": "Reverse Repo",
        "category": "Liquidity",
        "unit": "USD bn",
        "description": "Overnight Reverse Repo usage.",
    },
    "TGA": {
        "ticker": "WTREGEN",
        "name": "Treasury General Account",
        "category": "Liquidity",
        "unit": "USD bn",
        "description": "Treasury cash balance at the Fed.",
    },
    "MMF_RETAIL": {
        "ticker": "WRMFNS",
        "name": "Retail Money Market Funds",
        "category": "Liquidity",
        "unit": "USD bn",
        "description": "Retail money market fund assets.",
    },
}

CREDIT_KEYS = ["HY_OAS", "BBB_OAS", "CORP_OAS", "FIN_STRESS"]
CURVE_KEYS = ["SPREAD_10Y2Y", "SPREAD_10Y3M"]
RATE_KEYS = ["US10Y", "US2Y", "US3M"]
LIQUIDITY_KEYS = ["FED_BALANCE_SHEET", "RRP", "TGA", "MMF_RETAIL"]

LOOKBACK_MAP = {
    "3M": 90,
    "6M": 180,
    "1Y": 365,
    "3Y": 365 * 3,
    "5Y": 365 * 5,
    "10Y": 365 * 10,
    "MAX": None,
}

# ============================================================
# Helpers
# ============================================================
def safe_float(value):
    try:
        if value in [".", "", None]:
            return np.nan
        return float(value)
    except Exception:
        return np.nan


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_fred_series(series_id: str) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "file_type": "json",
        "sort_order": "asc",
    }
    if FRED_API_KEY:
        params["api_key"] = FRED_API_KEY

    retries = 3
    for attempt in range(retries):
        try:
            response = requests.get(FRED_BASE_URL, params=params, timeout=20)
            response.raise_for_status()
            payload = response.json()
            observations = payload.get("observations", [])

            df = pd.DataFrame(observations)
            if df.empty:
                return pd.DataFrame(columns=["date", series_id])

            df["date"] = pd.to_datetime(df["date"])
            df[series_id] = df["value"].apply(safe_float)
            df = df[["date", series_id]].dropna(subset=[series_id]).sort_values("date")
            return df

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                st.error(f"Failed to load FRED series {series_id}: {e}")
                return pd.DataFrame(columns=["date", series_id])


def merge_series(series_keys: list[str]) -> pd.DataFrame:
    merged = None
    for key in series_keys:
        ticker = SERIES_META[key]["ticker"]
        df = fetch_fred_series(ticker).rename(columns={ticker: key})

        if merged is None:
            merged = df.copy()
        else:
            merged = pd.merge(merged, df, on="date", how="outer")

    if merged is None:
        return pd.DataFrame()

    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def filter_by_lookback(df: pd.DataFrame, lookback_label: str) -> pd.DataFrame:
    if df.empty:
        return df

    days = LOOKBACK_MAP[lookback_label]
    if days is None:
        return df.copy()

    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)
    return df[df["date"] >= cutoff].copy()


def latest_valid_value(df: pd.DataFrame, column: str):
    if column not in df.columns:
        return None
    s = df[column].dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def previous_valid_value_by_days(df: pd.DataFrame, column: str, days_back: int = 30):
    if column not in df.columns or df.empty:
        return None

    s = df[["date", column]].dropna().sort_values("date")
    if s.empty:
        return None

    latest_date = s["date"].iloc[-1]
    target_date = latest_date - pd.Timedelta(days=days_back)

    eligible = s[s["date"] <= target_date]
    if eligible.empty:
        return None

    return float(eligible[column].iloc[-1])


def delta_value(current, previous):
    if current is None or previous is None:
        return None
    return current - previous


def convert_units(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # millions -> billions
    for col in ["FED_BALANCE_SHEET", "TGA"]:
        if col in out.columns:
            out[col] = out[col] / 1000.0

    # RRP and MMF_RETAIL are already in billions
    return out


def add_sparse_safe_derived_series(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("date").reset_index(drop=True)

    if all(c in out.columns for c in ["FED_BALANCE_SHEET", "TGA", "RRP"]):
        net_df = out[["date", "FED_BALANCE_SHEET", "TGA", "RRP"]].dropna().copy()
        if not net_df.empty:
            net_df["NET_LIQUIDITY_PROXY"] = (
                net_df["FED_BALANCE_SHEET"] - net_df["TGA"] - net_df["RRP"]
            )
            net_df["NET_LIQUIDITY_4W_CHANGE"] = net_df["NET_LIQUIDITY_PROXY"].diff(4)
            out = out.merge(
                net_df[["date", "NET_LIQUIDITY_PROXY", "NET_LIQUIDITY_4W_CHANGE"]],
                on="date",
                how="left",
            )

    if "MMF_RETAIL" in out.columns:
        mmf = out[["date", "MMF_RETAIL"]].dropna().copy()
        if not mmf.empty:
            mmf["MMF_FLOW_PROXY"] = mmf["MMF_RETAIL"].diff()
            mmf["MMF_FLOW_PROXY_4W_MA"] = mmf["MMF_FLOW_PROXY"].rolling(4).mean()
            out = out.merge(
                mmf[["date", "MMF_FLOW_PROXY", "MMF_FLOW_PROXY_4W_MA"]],
                on="date",
                how="left",
            )

    return out


def prepare_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    out = convert_units(df)
    out = add_sparse_safe_derived_series(out)
    return out


def classify_credit_spread_hy(value):
    if value is None or np.isnan(value):
        return "N/A"
    if value < 3.5:
        return "Low Risk"
    elif value < 5.0:
        return "Normal"
    elif value < 7.0:
        return "Warning"
    return "High Risk"


def classify_credit_spread_bbb(value):
    if value is None or np.isnan(value):
        return "N/A"
    if value < 1.5:
        return "Low Risk"
    elif value < 2.5:
        return "Normal"
    elif value < 3.5:
        return "Warning"
    return "High Risk"


def classify_credit_spread_corp(value):
    if value is None or np.isnan(value):
        return "N/A"
    if value < 1.2:
        return "Low Risk"
    elif value < 2.0:
        return "Normal"
    elif value < 3.0:
        return "Warning"
    return "High Risk"


def classify_fin_stress(value):
    if value is None or np.isnan(value):
        return "N/A"
    if value < 0:
        return "Low Risk"
    elif value < 1.0:
        return "Normal"
    elif value < 2.0:
        return "Warning"
    return "High Risk"


def classify_curve_10y2y(value):
    if value is None or np.isnan(value):
        return "N/A"
    if value > 0.5:
        return "Healthy"
    elif value > 0:
        return "Flattening"
    elif value > -0.5:
        return "Inverted"
    return "Deep Inversion"


def classify_curve_10y3m(value):
    if value is None or np.isnan(value):
        return "N/A"
    if value > 0.75:
        return "Healthy"
    elif value > 0:
        return "Flattening"
    elif value > -0.5:
        return "Inverted"
    return "Deep Inversion"


def classify_liquidity_delta(value):
    if value is None or np.isnan(value):
        return "N/A"
    if value > 200:
        return "Strong Positive"
    elif value > 50:
        return "Positive"
    elif value > -50:
        return "Neutral"
    elif value > -200:
        return "Negative"
    return "Strong Negative"


def get_signal_label(key: str, value):
    if key == "HY_OAS":
        return classify_credit_spread_hy(value)
    if key == "BBB_OAS":
        return classify_credit_spread_bbb(value)
    if key == "CORP_OAS":
        return classify_credit_spread_corp(value)
    if key == "FIN_STRESS":
        return classify_fin_stress(value)
    if key == "SPREAD_10Y2Y":
        return classify_curve_10y2y(value)
    if key == "SPREAD_10Y3M":
        return classify_curve_10y3m(value)
    return "N/A"


def get_signal_comment(key: str, label: str) -> str:
    comments = {
        "HY_OAS": {
            "Low Risk": "Credit market is calm and supportive for risk assets.",
            "Normal": "Credit conditions are broadly normal.",
            "Warning": "Risk appetite is weakening in high yield credit.",
            "High Risk": "Credit stress is elevated and equity volatility risk is high.",
        },
        "BBB_OAS": {
            "Low Risk": "Lower investment grade credit remains stable.",
            "Normal": "Corporate funding conditions are normal.",
            "Warning": "Stress is spreading into lower investment grade credit.",
            "High Risk": "Funding conditions are materially deteriorating.",
        },
        "CORP_OAS": {
            "Low Risk": "Broad corporate credit conditions are stable.",
            "Normal": "Corporate credit environment is normal.",
            "Warning": "Broad corporate spreads are widening.",
            "High Risk": "Corporate credit stress is clearly elevated.",
        },
        "FIN_STRESS": {
            "Low Risk": "The financial system is broadly stable.",
            "Normal": "Financial conditions are within a normal range.",
            "Warning": "Financial market stress is increasing.",
            "High Risk": "System-wide financial stress is high.",
        },
        "SPREAD_10Y2Y": {
            "Healthy": "Yield curve shape is consistent with a healthier backdrop.",
            "Flattening": "Growth expectations are softening or policy pressure is rising.",
            "Inverted": "Policy pressure and slowdown concerns are visible.",
            "Deep Inversion": "Strong recession warning signal.",
        },
        "SPREAD_10Y3M": {
            "Healthy": "Yield curve remains healthy.",
            "Flattening": "Liquidity and growth expectations are softening.",
            "Inverted": "Often interpreted as a recession warning.",
            "Deep Inversion": "Very strong slowdown warning.",
        },
    }
    return comments.get(key, {}).get(label, "")


def infer_overall_risk(latest: dict) -> tuple[str, int]:
    score = 0

    hy = latest.get("HY_OAS")
    bbb = latest.get("BBB_OAS")
    corp = latest.get("CORP_OAS")
    stress = latest.get("FIN_STRESS")
    s10y2y = latest.get("SPREAD_10Y2Y")
    s10y3m = latest.get("SPREAD_10Y3M")

    if hy is not None:
        if hy >= 7.0:
            score += 3
        elif hy >= 5.0:
            score += 2
        elif hy >= 3.5:
            score += 1

    if bbb is not None:
        if bbb >= 3.5:
            score += 3
        elif bbb >= 2.5:
            score += 2
        elif bbb >= 1.5:
            score += 1

    if corp is not None:
        if corp >= 3.0:
            score += 3
        elif corp >= 2.0:
            score += 2
        elif corp >= 1.2:
            score += 1

    if stress is not None:
        if stress >= 2.0:
            score += 3
        elif stress >= 1.0:
            score += 2
        elif stress >= 0:
            score += 1

    if s10y2y is not None:
        if s10y2y <= -0.5:
            score += 2
        elif s10y2y < 0:
            score += 1

    if s10y3m is not None:
        if s10y3m <= -0.5:
            score += 2
        elif s10y3m < 0:
            score += 1

    if score <= 2:
        return "Low Risk", score
    elif score <= 5:
        return "Moderate", score
    elif score <= 8:
        return "Elevated", score
    return "High Risk", score


def make_line_chart(df: pd.DataFrame, y_col: str, title: str, y_label: str):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df[y_col],
            mode="lines",
            name=title,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        height=380,
        margin=dict(l=30, r=20, t=60, b=30),
        hovermode="x unified",
    )
    return fig


def make_bar_line_chart(df: pd.DataFrame, bar_col: str, line_col: str, title: str, y_label: str):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df[bar_col],
            name=bar_col,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df[line_col],
            mode="lines",
            name=line_col,
        )
    )
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        height=420,
        margin=dict(l=30, r=20, t=60, b=30),
        hovermode="x unified",
    )
    return fig


def make_normalized_chart(df: pd.DataFrame, columns: list[str], title: str, use_log_y: bool = False):
    fig = go.Figure()

    color_map = {
        "FED_BALANCE_SHEET": "#4FC3F7",
        "RRP": "#1E88E5",
        "TGA": "#FFB74D",
        "MMF_RETAIL": "#EF5350",
        "NET_LIQUIDITY_PROXY": "#66BB6A",
        "HY_OAS": "#AB47BC",
        "BBB_OAS": "#FFA726",
        "CORP_OAS": "#26C6DA",
        "FIN_STRESS": "#EC407A",
    }

    pretty_name = {
        "FED_BALANCE_SHEET": "Fed Balance Sheet",
        "RRP": "Reverse Repo",
        "TGA": "Treasury General Account",
        "MMF_RETAIL": "Retail Money Market Funds",
        "NET_LIQUIDITY_PROXY": "Net Liquidity Proxy",
        "HY_OAS": "US High Yield OAS",
        "BBB_OAS": "BBB Corporate OAS",
        "CORP_OAS": "US Corporate OAS",
        "FIN_STRESS": "Financial Stress Index",
    }

    valid_trace_count = 0

    for col in columns:
        if col not in df.columns:
            continue

        s = df[["date", col]].dropna().copy()
        if s.empty:
            continue

        base = s[col].iloc[0]
        if pd.isna(base) or base == 0:
            continue

        s["normalized"] = s[col] / base

        if use_log_y:
            s = s[s["normalized"] > 0].copy()

        if s.empty:
            continue

        valid_trace_count += 1
        fig.add_trace(
            go.Scatter(
                x=s["date"],
                y=s["normalized"],
                mode="lines",
                name=pretty_name.get(col, col),
                line=dict(width=3, color=color_map.get(col, None)),
                hovertemplate="%{x|%b %Y}<br>%{y:.2f}x<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Normalized (start = 1)",
        height=460,
        margin=dict(l=30, r=20, t=60, b=30),
        legend=dict(orientation="v"),
        hovermode="x unified",
    )

    if use_log_y:
        fig.update_yaxes(type="log")

    return fig, valid_trace_count


def make_liquidity_dual_axis_chart(df: pd.DataFrame):
    fig = go.Figure()

    color_map = {
        "FED_BALANCE_SHEET": "#4FC3F7",
        "RRP": "#1E88E5",
        "TGA": "#FFB74D",
    }

    if "FED_BALANCE_SHEET" in df.columns:
        s = df[["date", "FED_BALANCE_SHEET"]].dropna()
        if not s.empty:
            fig.add_trace(
                go.Scatter(
                    x=s["date"],
                    y=s["FED_BALANCE_SHEET"],
                    mode="lines",
                    name="Fed Balance Sheet",
                    line=dict(width=3, color=color_map["FED_BALANCE_SHEET"]),
                    yaxis="y1",
                )
            )

    if "RRP" in df.columns:
        s = df[["date", "RRP"]].dropna()
        if not s.empty:
            fig.add_trace(
                go.Scatter(
                    x=s["date"],
                    y=s["RRP"],
                    mode="lines",
                    name="Reverse Repo",
                    line=dict(width=3, color=color_map["RRP"]),
                    yaxis="y2",
                )
            )

    if "TGA" in df.columns:
        s = df[["date", "TGA"]].dropna()
        if not s.empty:
            fig.add_trace(
                go.Scatter(
                    x=s["date"],
                    y=s["TGA"],
                    mode="lines",
                    name="Treasury General Account",
                    line=dict(width=3, color=color_map["TGA"]),
                    yaxis="y2",
                )
            )

    fig.update_layout(
        title="Liquidity Components (Raw Values, Excluding Retail MMF)",
        xaxis=dict(title="Date"),
        yaxis=dict(
            title="Fed Balance Sheet (USD bn)",
            side="left",
            showgrid=True,
        ),
        yaxis2=dict(
            title="RRP / TGA (USD bn)",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        height=500,
        margin=dict(l=30, r=30, t=60, b=30),
        hovermode="x unified",
        legend=dict(orientation="v"),
    )

    return fig

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("Settings")
    lookback = st.selectbox("Select lookback", list(LOOKBACK_MAP.keys()), index=2)
    show_normalized = st.checkbox("Show normalized comparison", value=True)
    show_summary_table = st.checkbox("Show summary table", value=True)
    show_liquidity = st.checkbox("Show liquidity section", value=True)

    st.markdown("---")
    st.subheader("Core Series")
    for key in CREDIT_KEYS + CURVE_KEYS + RATE_KEYS + LIQUIDITY_KEYS:
        meta = SERIES_META[key]
        st.write(f"**{meta['name']}**")
        st.caption(f"{meta['ticker']} · {meta['description']}")

# ============================================================
# Load data
# ============================================================
all_series = list(SERIES_META.keys())
raw_df = merge_series(all_series)
raw_df = prepare_liquidity_features(raw_df)
df = filter_by_lookback(raw_df, lookback)

if df.empty:
    st.warning("No data available.")
    st.stop()

# ============================================================
# Latest snapshot
# ============================================================
latest = {}
for key in SERIES_META.keys():
    latest[key] = latest_valid_value(df, key)

latest["NET_LIQUIDITY_PROXY"] = latest_valid_value(df, "NET_LIQUIDITY_PROXY")
latest["MMF_FLOW_PROXY"] = latest_valid_value(df, "MMF_FLOW_PROXY")
latest["MMF_FLOW_PROXY_4W_MA"] = latest_valid_value(df, "MMF_FLOW_PROXY_4W_MA")
latest["NET_LIQUIDITY_4W_CHANGE"] = latest_valid_value(df, "NET_LIQUIDITY_4W_CHANGE")

overall_risk, risk_score = infer_overall_risk(latest)

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Overall Risk Signal")
    if overall_risk == "Low Risk":
        st.success(f"Overall Risk: {overall_risk} | Score: {risk_score}")
    elif overall_risk == "Moderate":
        st.info(f"Overall Risk: {overall_risk} | Score: {risk_score}")
    elif overall_risk == "Elevated":
        st.warning(f"Overall Risk: {overall_risk} | Score: {risk_score}")
    else:
        st.error(f"Overall Risk: {overall_risk} | Score: {risk_score}")

    st.write(
        "This dashboard tracks whether credit markets and liquidity conditions are improving or tightening "
        "before equities fully react. Widening credit spreads, rising financial stress, shrinking liquidity, "
        "and weak MMF flow behavior deserve more caution."
    )

with col2:
    st.subheader("Quick Interpretation")
    quick_lines = []
    for key in ["HY_OAS", "BBB_OAS", "FIN_STRESS", "SPREAD_10Y2Y", "SPREAD_10Y3M"]:
        value = latest.get(key)
        label = get_signal_label(key, value)
        quick_lines.append(f"- **{SERIES_META[key]['name']}**: {label}")

    quick_lines.append(
        f"- **Net Liquidity 4W Change**: {classify_liquidity_delta(latest.get('NET_LIQUIDITY_4W_CHANGE'))}"
    )
    quick_lines.append(
        f"- **MMF Flow 4W MA**: {classify_liquidity_delta(latest.get('MMF_FLOW_PROXY_4W_MA'))}"
    )
    st.markdown("\n".join(quick_lines))

# ============================================================
# Credit metrics row
# ============================================================
st.subheader("Credit / Curve Snapshot")

metric_keys = ["HY_OAS", "BBB_OAS", "CORP_OAS", "FIN_STRESS", "SPREAD_10Y2Y", "SPREAD_10Y3M"]
metric_cols = st.columns(len(metric_keys))

for i, key in enumerate(metric_keys):
    current = latest_valid_value(df, key)
    previous = previous_valid_value_by_days(df, key, days_back=30)
    delta = delta_value(current, previous)
    unit = SERIES_META[key]["unit"]
    label = get_signal_label(key, current)

    value_text = "N/A" if current is None else f"{current:.2f} {unit}"
    delta_text = None if delta is None else f"{delta:+.2f} {unit}"

    metric_cols[i].metric(
        label=SERIES_META[key]["name"],
        value=value_text,
        delta=delta_text,
    )
    metric_cols[i].caption(label)

# ============================================================
# Liquidity metrics row
# ============================================================
if show_liquidity:
    st.subheader("Liquidity Snapshot")

    liq_metric_cols = st.columns(6)
    liquidity_metric_config = [
        ("FED_BALANCE_SHEET", "Fed Balance Sheet"),
        ("RRP", "Reverse Repo"),
        ("TGA", "TGA"),
        ("NET_LIQUIDITY_PROXY", "Net Liquidity Proxy"),
        ("MMF_FLOW_PROXY", "MMF Weekly Flow Proxy"),
        ("MMF_FLOW_PROXY_4W_MA", "MMF Flow 4W MA"),
    ]

    for i, (key, label_name) in enumerate(liquidity_metric_config):
        current = latest_valid_value(df, key)
        previous = previous_valid_value_by_days(df, key, days_back=30)
        delta = delta_value(current, previous)

        value_text = "N/A" if current is None else f"{current:,.1f}"
        delta_text = None if delta is None else f"{delta:+,.1f}"

        liq_metric_cols[i].metric(
            label=label_name,
            value=value_text,
            delta=delta_text,
        )

# ============================================================
# Summary table
# ============================================================
if show_summary_table:
    st.subheader("Signal Summary Table")

    summary_rows = []
    for key in metric_keys:
        current = latest_valid_value(df, key)
        previous = previous_valid_value_by_days(df, key, days_back=30)
        delta = delta_value(current, previous)
        label = get_signal_label(key, current)
        comment = get_signal_comment(key, label)

        summary_rows.append(
            {
                "Series": SERIES_META[key]["name"],
                "Ticker": SERIES_META[key]["ticker"],
                "Latest": None if current is None else round(current, 3),
                "1M Change": None if delta is None else round(delta, 3),
                "Unit": SERIES_META[key]["unit"],
                "Signal": label,
                "Interpretation": comment,
            }
        )

    if show_liquidity:
        for key in ["FED_BALANCE_SHEET", "RRP", "TGA", "MMF_RETAIL"]:
            current = latest_valid_value(df, key)
            previous = previous_valid_value_by_days(df, key, days_back=30)
            delta = delta_value(current, previous)

            summary_rows.append(
                {
                    "Series": SERIES_META[key]["name"],
                    "Ticker": SERIES_META[key]["ticker"],
                    "Latest": None if current is None else round(current, 1),
                    "1M Change": None if delta is None else round(delta, 1),
                    "Unit": SERIES_META[key]["unit"],
                    "Signal": "Liquidity",
                    "Interpretation": SERIES_META[key]["description"],
                }
            )

        derived_liquidity_items = [
            ("NET_LIQUIDITY_PROXY", "Derived internal liquidity proxy."),
            ("NET_LIQUIDITY_4W_CHANGE", "4-week change in net liquidity proxy."),
            ("MMF_FLOW_PROXY", "Weekly proxy for retail money market flow."),
            ("MMF_FLOW_PROXY_4W_MA", "4-week moving average of MMF flow proxy."),
        ]

        for key, desc in derived_liquidity_items:
            current = latest_valid_value(df, key)
            previous = previous_valid_value_by_days(df, key, days_back=30)
            delta = delta_value(current, previous)

            summary_rows.append(
                {
                    "Series": key,
                    "Ticker": "Derived",
                    "Latest": None if current is None else round(current, 1),
                    "1M Change": None if delta is None else round(delta, 1),
                    "Unit": "USD bn",
                    "Signal": classify_liquidity_delta(current),
                    "Interpretation": desc,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True)

# ============================================================
# Credit charts
# ============================================================
st.subheader("Credit Signal Charts")

for key in metric_keys:
    if key not in df.columns:
        continue
    chart_df = df[["date", key]].dropna()
    if not chart_df.empty:
        fig = make_line_chart(
            chart_df,
            y_col=key,
            title=f"{SERIES_META[key]['name']} ({SERIES_META[key]['ticker']})",
            y_label=SERIES_META[key]["unit"],
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Treasury rates chart
# ============================================================
st.subheader("Treasury Yields")

rate_chart_df = df[["date"] + RATE_KEYS].dropna(subset=RATE_KEYS, how="all")
if not rate_chart_df.empty:
    fig = go.Figure()
    for col in RATE_KEYS:
        s = rate_chart_df[["date", col]].dropna()
        if s.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=s["date"],
                y=s[col],
                mode="lines",
                name=SERIES_META[col]["name"],
            )
        )
    fig.update_layout(
        title="US Treasury Yields",
        xaxis_title="Date",
        yaxis_title="%",
        height=420,
        margin=dict(l=30, r=20, t=60, b=30),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Yield curve chart
# ============================================================
st.subheader("Yield Curve Monitoring")

curve_chart_df = df[["date"] + CURVE_KEYS].dropna(subset=CURVE_KEYS, how="all")
if not curve_chart_df.empty:
    fig = go.Figure()
    for col in CURVE_KEYS:
        s = curve_chart_df[["date", col]].dropna()
        if s.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=s["date"],
                y=s[col],
                mode="lines",
                name=SERIES_META[col]["name"],
            )
        )
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(
        title="Yield Curve Spreads",
        xaxis_title="Date",
        yaxis_title="%p",
        height=420,
        margin=dict(l=30, r=20, t=60, b=30),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Liquidity charts
# ============================================================
if show_liquidity:
    st.subheader("Liquidity Charts")

    # Individual liquidity charts
    liquidity_single_chart_order = [
        "FED_BALANCE_SHEET",
        "RRP",
        "TGA",
        "MMF_RETAIL",
        "NET_LIQUIDITY_PROXY",
    ]

    for key in liquidity_single_chart_order:
        if key in df.columns:
            chart_df = df[["date", key]].dropna()
            if not chart_df.empty:
                y_label = "USD bn"
                if key in SERIES_META:
                    chart_title = f"{SERIES_META[key]['name']} ({SERIES_META[key]['ticker']})"
                elif key == "NET_LIQUIDITY_PROXY":
                    chart_title = "Net Liquidity Proxy (Fed Balance Sheet - TGA - RRP)"
                else:
                    chart_title = key

                fig = make_line_chart(
                    chart_df,
                    y_col=key,
                    title=chart_title,
                    y_label=y_label,
                )
                st.plotly_chart(fig, use_container_width=True)

    # Raw combined liquidity chart without normalization, excluding Retail MMF
    st.subheader("Liquidity Components (Raw, Excluding Retail MMF)")

    liq_raw_cols = ["FED_BALANCE_SHEET", "RRP", "TGA"]
    liq_raw_df = df[["date"] + liq_raw_cols].dropna(subset=liq_raw_cols, how="all")

    if not liq_raw_df.empty:
        fig = make_liquidity_dual_axis_chart(liq_raw_df)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No raw liquidity component data available for the selected lookback.")

    # MMF flow proxy chart
    if all(c in df.columns for c in ["MMF_FLOW_PROXY", "MMF_FLOW_PROXY_4W_MA"]):
        mmf_df = df[["date", "MMF_FLOW_PROXY", "MMF_FLOW_PROXY_4W_MA"]].dropna(
            subset=["MMF_FLOW_PROXY", "MMF_FLOW_PROXY_4W_MA"],
            how="all",
        )
        if not mmf_df.empty:
            fig = make_bar_line_chart(
                mmf_df,
                bar_col="MMF_FLOW_PROXY",
                line_col="MMF_FLOW_PROXY_4W_MA",
                title="Money Market Fund Flow Proxy",
                y_label="USD bn",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No MMF flow data available for the selected lookback.")

# ============================================================
# Normalized comparison
# ============================================================
if show_normalized:
    st.subheader("Normalized Credit / Stress / Liquidity Comparison")

    norm_cols = ["HY_OAS", "BBB_OAS", "CORP_OAS", "FIN_STRESS"]
    if show_liquidity and "NET_LIQUIDITY_PROXY" in df.columns:
        norm_cols.append("NET_LIQUIDITY_PROXY")

    norm_df = df[["date"] + norm_cols].dropna(subset=norm_cols, how="all")
    if not norm_df.empty:
        fig, trace_count = make_normalized_chart(
            norm_df,
            columns=norm_cols,
            title="Normalized Credit / Stress / Liquidity Comparison",
            use_log_y=False,
        )
        if trace_count > 0:
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Dashboard reading guide
# ============================================================
st.subheader("How to Read This Dashboard")

st.markdown(
    """
**Main idea**
- Credit often weakens before equity headlines fully reflect the problem.
- Liquidity often supports or suppresses the magnitude of the move.
- Widening credit spreads + rising financial stress + falling liquidity is a weaker backdrop.
- Tightening spreads + stable stress + improving liquidity is a more supportive backdrop.

**Typical warning combination**
- High Yield OAS rising
- BBB OAS rising
- Financial Stress rising
- Yield curve flat or inverted
- Net Liquidity Proxy falling
- MMF flow weak or turning negative

**Typical supportive combination**
- Credit spreads narrowing
- Financial stress falling
- Net liquidity rising
- Money market cash starting to redeploy
"""
)

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption(
    f"Last updated in app runtime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
    "Data source: Federal Reserve Economic Data (FRED)"
)
