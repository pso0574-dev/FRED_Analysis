# -*- coding: utf-8 -*-
"""
US Macro / Liquidity / Recession Dashboard (Streamlit)

Features
--------
1) Liquidity
   - Net Liquidity = WALCL - TGA - RRP
   - Components chart
2) Rates
   - SOFR, EFFR, 10Y, 2Y, 3M, Yield Curve
3) Credit
   - High Yield Spread, Financial Stress
4) Market
   - S&P 500, Nasdaq, QQQ, VIX
5) Risk Signal
   - Risk On / Neutral / Risk Off scoring
6) Methodology
   - Detailed explanation of each metric

Run locally
-----------
streamlit run app.py

Packages
--------
pip install streamlit pandas numpy plotly requests yfinance
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
import yfinance as yf

# =========================================================
# 1) APP CONFIG
# =========================================================
st.set_page_config(
    page_title="US Macro / Liquidity Dashboard",
    page_icon="📊",
    layout="wide",
)

APP_TITLE = "US Macro / Liquidity / Recession Dashboard"
DEFAULT_LOOKBACK_YEARS = 10
REQUEST_TIMEOUT = 20
MAX_RETRIES = 4
BACKOFF_SEC = 1.5

# FRED API KEY
# Streamlit Cloud 배포 시 Secrets에 저장 권장:
# [secrets]
# FRED_API_KEY="YOUR_KEY"
FRED_API_KEY = st.secrets["FRED_API_KEY"] if "FRED_API_KEY" in st.secrets else os.getenv("FRED_API_KEY", "")

if not FRED_API_KEY:
    st.warning("FRED_API_KEY가 설정되지 않았습니다. Streamlit Cloud의 Secrets 또는 환경변수에 넣어주세요.")

# =========================================================
# 2) FRED SERIES MAP
# =========================================================
FRED_SERIES = {
    # Liquidity
    "WALCL": "WALCL",          # Fed Balance Sheet (Million USD)
    "TGA": "WTREGEN",          # Treasury General Account (Million USD)
    "RRP": "RRPONTSYD",        # ON RRP Total Take-up (Billion USD, daily)

    # Rates
    "SOFR": "SOFR",
    "EFFR": "EFFR",
    "DGS10": "DGS10",          # 10Y Treasury
    "DGS2": "DGS2",            # 2Y Treasury
    "TB3MS": "TB3MS",          # 3M Treasury Bill secondary market rate

    # Credit / Stress
    "HY_SPREAD": "BAMLH0A0HYM2",   # ICE BofA US High Yield Index OAS
    "FSI": "STLFSI4",              # St. Louis Fed Financial Stress Index
    "INDPRO": "INDPRO",            # Industrial Production Index
    "CPI": "CPIAUCSL",             # CPI
    "UNRATE": "UNRATE",            # Unemployment Rate
}

MARKET_TICKERS = {
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "QQQ": "QQQ",
    "VIX": "^VIX",
}

# =========================================================
# 3) STYLES
# =========================================================
st.markdown(
    """
    <style>
    .metric-card {
        padding: 12px 16px;
        border-radius: 14px;
        background-color: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 10px;
    }
    .small-note {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 4) DATA HELPERS
# =========================================================
def _fred_get_json(series_id: str, start_date: str) -> dict:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start_date,
        "sort_order": "asc",
    }

    last_err = None
    for i in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(BACKOFF_SEC * (i + 1))
    raise RuntimeError(f"Failed to load {series_id}: {last_err}")


@st.cache_data(ttl=1800, show_spinner=False)
def load_fred_series(series_id: str, start_date: str) -> pd.Series:
    data = _fred_get_json(series_id, start_date)
    obs = data.get("observations", [])
    if not obs:
        return pd.Series(dtype=float)

    rows = []
    for x in obs:
        date_str = x.get("date")
        val = x.get("value", ".")
        if val == ".":
            continue
        try:
            rows.append((pd.to_datetime(date_str), float(val)))
        except Exception:
            pass

    if not rows:
        return pd.Series(dtype=float)

    s = pd.Series({d: v for d, v in rows}).sort_index()
    s.name = series_id
    return s


@st.cache_data(ttl=1800, show_spinner=False)
def load_market_data(ticker: str, start_date: str) -> pd.Series:
    df = yf.download(ticker, start=start_date, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.Series(dtype=float)

    col = "Close"
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance sometimes returns multiindex
        if ("Close", ticker) in df.columns:
            s = df[("Close", ticker)].copy()
        else:
            s = df.xs("Close", axis=1, level=0).iloc[:, 0].copy()
    else:
        s = df[col].copy()

    s.index = pd.to_datetime(s.index)
    s.name = ticker
    return s.dropna()


def to_billions(series: pd.Series, source_name: str) -> pd.Series:
    """
    Convert to billion USD if needed.
    WALCL, WTREGEN are usually in million USD.
    RRPONTSYD is usually billion USD already.
    """
    s = series.copy()

    if source_name in ["WALCL", "WTREGEN"]:
        s = s / 1000.0  # million -> billion
    elif source_name in ["RRPONTSYD"]:
        s = s  # already in billion
    return s


def resample_daily_ffill(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    full_idx = pd.date_range(series.index.min(), series.index.max(), freq="D")
    return series.reindex(full_idx).ffill()


def align_series_dict(series_dict: dict[str, pd.Series]) -> pd.DataFrame:
    clean = {}
    for k, s in series_dict.items():
        if s is not None and not s.empty:
            clean[k] = resample_daily_ffill(s)
    if not clean:
        return pd.DataFrame()
    return pd.concat(clean, axis=1).sort_index().ffill()


def zscore_last(series: pd.Series, window: int = 252) -> float:
    s = series.dropna()
    if len(s) < max(30, window // 4):
        return np.nan
    tail = s.iloc[-window:] if len(s) >= window else s
    mu = tail.mean()
    sd = tail.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float((s.iloc[-1] - mu) / sd)


def yoy(series: pd.Series) -> pd.Series:
    return series.pct_change(12) * 100


def build_net_liquidity(start_date: str) -> pd.DataFrame:
    walcl = load_fred_series(FRED_SERIES["WALCL"], start_date)
    tga = load_fred_series(FRED_SERIES["TGA"], start_date)
    rrp = load_fred_series(FRED_SERIES["RRP"], start_date)

    walcl_b = to_billions(walcl, "WALCL")
    tga_b = to_billions(tga, "WTREGEN")
    rrp_b = to_billions(rrp, "RRPONTSYD")

    df = align_series_dict({
        "WALCL": walcl_b,
        "TGA": tga_b,
        "RRP": rrp_b,
    })
    if df.empty:
        return df

    df["NetLiquidity"] = df["WALCL"] - df["TGA"] - df["RRP"]
    return df


def build_rates_df(start_date: str) -> pd.DataFrame:
    d = {
        "SOFR": load_fred_series(FRED_SERIES["SOFR"], start_date),
        "EFFR": load_fred_series(FRED_SERIES["EFFR"], start_date),
        "10Y": load_fred_series(FRED_SERIES["DGS10"], start_date),
        "2Y": load_fred_series(FRED_SERIES["DGS2"], start_date),
        "3M": load_fred_series(FRED_SERIES["TB3MS"], start_date),
    }
    df = align_series_dict(d)
    if df.empty:
        return df

    df["10Y-2Y"] = df["10Y"] - df["2Y"]
    df["10Y-3M"] = df["10Y"] - df["3M"]
    return df


def build_credit_df(start_date: str) -> pd.DataFrame:
    d = {
        "HY Spread": load_fred_series(FRED_SERIES["HY_SPREAD"], start_date),
        "FSI": load_fred_series(FRED_SERIES["FSI"], start_date),
        "INDPRO": load_fred_series(FRED_SERIES["INDPRO"], start_date),
        "CPI": load_fred_series(FRED_SERIES["CPI"], start_date),
        "UNRATE": load_fred_series(FRED_SERIES["UNRATE"], start_date),
    }
    df = align_series_dict(d)
    if df.empty:
        return df

    # 월간 지표의 경우 YoY 보조 계산
    if "INDPRO" in df.columns:
        df["INDPRO_YoY"] = yoy(df["INDPRO"])
    if "CPI" in df.columns:
        df["CPI_YoY"] = yoy(df["CPI"])
    return df


def build_market_df(start_date: str) -> pd.DataFrame:
    d = {}
    for name, ticker in MARKET_TICKERS.items():
        d[name] = load_market_data(ticker, start_date)
    df = align_series_dict(d)
    return df


def latest_change(series: pd.Series, days: int = 20) -> float:
    s = series.dropna()
    if len(s) < days + 1:
        return np.nan
    return float((s.iloc[-1] / s.iloc[-days - 1] - 1) * 100)


# =========================================================
# 5) SIGNAL ENGINE
# =========================================================
def compute_risk_signal(liq_df: pd.DataFrame, rates_df: pd.DataFrame, credit_df: pd.DataFrame, market_df: pd.DataFrame):
    score = 0
    reasons = []

    # 1) Net Liquidity trend
    if not liq_df.empty and "NetLiquidity" in liq_df.columns:
        nl_3m = latest_change(liq_df["NetLiquidity"], 63)
        if not np.isnan(nl_3m):
            if nl_3m > 2:
                score += 1
                reasons.append(f"Net Liquidity 3M 변화율 개선 ({nl_3m:.1f}%)")
            elif nl_3m < -2:
                score -= 1
                reasons.append(f"Net Liquidity 3M 변화율 악화 ({nl_3m:.1f}%)")

    # 2) Yield curve
    if not rates_df.empty and "10Y-3M" in rates_df.columns:
        yc = rates_df["10Y-3M"].dropna()
        if len(yc) > 0:
            last_yc = yc.iloc[-1]
            if last_yc > 0.5:
                score += 1
                reasons.append(f"10Y-3M 정상/가파름 ({last_yc:.2f})")
            elif last_yc < 0:
                score -= 1
                reasons.append(f"10Y-3M 역전 ({last_yc:.2f})")

    # 3) HY Spread
    if not credit_df.empty and "HY Spread" in credit_df.columns:
        hy = credit_df["HY Spread"].dropna()
        if len(hy) > 0:
            last_hy = hy.iloc[-1]
            hy_z = zscore_last(hy, 252)
            if last_hy < 4.0 and (np.isnan(hy_z) or hy_z < 0.5):
                score += 1
                reasons.append(f"HY Spread 안정 ({last_hy:.2f}%)")
            elif last_hy > 5.5 or (not np.isnan(hy_z) and hy_z > 1.0):
                score -= 1
                reasons.append(f"HY Spread 확대 ({last_hy:.2f}%)")

    # 4) Financial Stress
    if not credit_df.empty and "FSI" in credit_df.columns:
        fsi = credit_df["FSI"].dropna()
        if len(fsi) > 0:
            last_fsi = fsi.iloc[-1]
            if last_fsi < 0:
                score += 1
                reasons.append(f"금융 스트레스 낮음 ({last_fsi:.2f})")
            elif last_fsi > 1.0:
                score -= 1
                reasons.append(f"금융 스트레스 높음 ({last_fsi:.2f})")

    # 5) Market trend
    if not market_df.empty and "S&P 500" in market_df.columns:
        spx_3m = latest_change(market_df["S&P 500"], 63)
        if not np.isnan(spx_3m):
            if spx_3m > 3:
                score += 1
                reasons.append(f"S&P 500 3M 상승 ({spx_3m:.1f}%)")
            elif spx_3m < -5:
                score -= 1
                reasons.append(f"S&P 500 3M 하락 ({spx_3m:.1f}%)")

    if score >= 3:
        regime = "RISK ON"
        color = "green"
    elif score <= -2:
        regime = "RISK OFF"
        color = "red"
    else:
        regime = "NEUTRAL"
        color = "orange"

    return score, regime, color, reasons


# =========================================================
# 6) CHART HELPERS
# =========================================================
def make_liquidity_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if df.empty:
        return fig

    fig.add_trace(go.Scatter(
        x=df.index, y=df["NetLiquidity"],
        mode="lines", name="Net Liquidity (B USD)",
        yaxis="y1"
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["WALCL"],
        mode="lines", name="Fed Balance Sheet (B)",
        yaxis="y2"
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["TGA"],
        mode="lines", name="TGA (B)",
        yaxis="y2"
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["RRP"],
        mode="lines", name="RRP (B)",
        yaxis="y2"
    ))

    fig.update_layout(
        title="Liquidity: Net Liquidity / WALCL / TGA / RRP",
        height=520,
        xaxis=dict(title="Date"),
        yaxis=dict(title="Net Liquidity (B USD)", side="left"),
        yaxis2=dict(title="Components (B USD)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(l=30, r=30, t=70, b=30),
    )
    return fig


def make_rates_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        return fig

    for c in ["SOFR", "EFFR", "10Y", "2Y", "3M"]:
        if c in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=c))

    fig.update_layout(
        title="Rates: SOFR / EFFR / Treasury Yields",
        height=500,
        xaxis_title="Date",
        yaxis_title="Rate (%)",
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(l=30, r=30, t=70, b=30),
    )
    return fig


def make_curve_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        return fig

    for c in ["10Y-2Y", "10Y-3M"]:
        if c in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=c))

    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(
        title="Yield Curve: 10Y-2Y / 10Y-3M",
        height=420,
        xaxis_title="Date",
        yaxis_title="Spread (%)",
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(l=30, r=30, t=70, b=30),
    )
    return fig


def make_credit_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        return fig

    if "HY Spread" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["HY Spread"], mode="lines", name="HY Spread",
            yaxis="y1"
        ))
    if "FSI" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["FSI"], mode="lines", name="Financial Stress",
            yaxis="y2"
        ))

    fig.update_layout(
        title="Credit / Stress: HY Spread / STLFSI4",
        height=500,
        xaxis_title="Date",
        yaxis=dict(title="HY Spread (%)", side="left"),
        yaxis2=dict(title="FSI", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(l=30, r=30, t=70, b=30),
    )
    return fig


def make_market_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        return fig

    for c in ["S&P 500", "Nasdaq", "QQQ"]:
        if c in df.columns:
            base = df[c].dropna()
            if not base.empty:
                norm = base / base.iloc[0] * 100
                fig.add_trace(go.Scatter(x=norm.index, y=norm, mode="lines", name=f"{c} (Normalized=100)"))

    fig.update_layout(
        title="Market Performance (Normalized)",
        height=500,
        xaxis_title="Date",
        yaxis_title="Normalized Index",
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(l=30, r=30, t=70, b=30),
    )
    return fig


def make_vix_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty or "VIX" not in df.columns:
        return fig

    fig.add_trace(go.Scatter(x=df.index, y=df["VIX"], mode="lines", name="VIX"))
    fig.update_layout(
        title="VIX",
        height=350,
        xaxis_title="Date",
        yaxis_title="Index",
        margin=dict(l=30, r=30, t=70, b=30),
    )
    return fig


# =========================================================
# 7) SIDEBAR
# =========================================================
st.sidebar.title("Dashboard Settings")

lookback_years = st.sidebar.slider("Lookback Years", min_value=3, max_value=20, value=DEFAULT_LOOKBACK_YEARS)
start_date = (datetime.today() - timedelta(days=365 * lookback_years)).strftime("%Y-%m-%d")

auto_refresh = st.sidebar.checkbox("Auto Refresh (every 30 min)", value=False)
show_raw_data = st.sidebar.checkbox("Show Raw Data", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("Data Sources")
st.sidebar.caption("- FRED")
st.sidebar.caption("- Yahoo Finance")

if auto_refresh:
    st.sidebar.info("자동 새로고침 활성화")
    st.markdown(
        f"""
        <script>
            setTimeout(function(){{
                window.location.reload();
            }}, {30*60*1000});
        </script>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# 8) LOAD DATA
# =========================================================
with st.spinner("Loading macro and market data..."):
    liq_df = build_net_liquidity(start_date)
    rates_df = build_rates_df(start_date)
    credit_df = build_credit_df(start_date)
    market_df = build_market_df(start_date)

score, regime, regime_color, reasons = compute_risk_signal(liq_df, rates_df, credit_df, market_df)

# =========================================================
# 9) HEADER
# =========================================================
st.title(APP_TITLE)
st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

c1, c2, c3, c4 = st.columns(4)

with c1:
    latest_nl = liq_df["NetLiquidity"].dropna().iloc[-1] if not liq_df.empty else np.nan
    st.metric("Net Liquidity (B USD)", f"{latest_nl:,.0f}" if not np.isnan(latest_nl) else "N/A")

with c2:
    latest_yc = rates_df["10Y-3M"].dropna().iloc[-1] if not rates_df.empty and "10Y-3M" in rates_df.columns else np.nan
    st.metric("10Y-3M", f"{latest_yc:.2f}" if not np.isnan(latest_yc) else "N/A")

with c3:
    latest_hy = credit_df["HY Spread"].dropna().iloc[-1] if not credit_df.empty and "HY Spread" in credit_df.columns else np.nan
    st.metric("HY Spread", f"{latest_hy:.2f}%" if not np.isnan(latest_hy) else "N/A")

with c4:
    st.metric("Macro Risk Regime", regime)

st.markdown("---")

# =========================================================
# 10) TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Liquidity",
    "Rates",
    "Credit",
    "Market",
    "Risk Signal",
    "Methodology",
])

# ---------------------------------------------------------
# TAB 1: LIQUIDITY
# ---------------------------------------------------------
with tab1:
    st.subheader("Liquidity Dashboard")
    st.plotly_chart(make_liquidity_chart(liq_df), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        v = liq_df["WALCL"].dropna().iloc[-1] if not liq_df.empty else np.nan
        st.metric("WALCL (B)", f"{v:,.0f}" if not np.isnan(v) else "N/A")
    with c2:
        v = liq_df["TGA"].dropna().iloc[-1] if not liq_df.empty else np.nan
        st.metric("TGA (B)", f"{v:,.0f}" if not np.isnan(v) else "N/A")
    with c3:
        v = liq_df["RRP"].dropna().iloc[-1] if not liq_df.empty else np.nan
        st.metric("RRP (B)", f"{v:,.0f}" if not np.isnan(v) else "N/A")
    with c4:
        nl_3m = latest_change(liq_df["NetLiquidity"], 63) if not liq_df.empty else np.nan
        st.metric("Net Liquidity 3M %", f"{nl_3m:.1f}%" if not np.isnan(nl_3m) else "N/A")

    st.markdown("""
    **Interpretation**
    - **WALCL 상승**: Fed 유동성 공급 증가
    - **TGA 상승**: 시장 내 현금 흡수
    - **RRP 하락**: 시장 유동성 증가에 우호적
    - **Net Liquidity 상승**: 일반적으로 위험자산에 우호적
    """)

# ---------------------------------------------------------
# TAB 2: RATES
# ---------------------------------------------------------
with tab2:
    st.subheader("Rates Dashboard")
    st.plotly_chart(make_rates_chart(rates_df), use_container_width=True)
    st.plotly_chart(make_curve_chart(rates_df), use_container_width=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    rate_names = ["SOFR", "EFFR", "10Y", "2Y", "3M"]
    for col, name in zip([c1, c2, c3, c4, c5], rate_names):
        with col:
            val = rates_df[name].dropna().iloc[-1] if not rates_df.empty and name in rates_df.columns else np.nan
            st.metric(name, f"{val:.2f}%" if not np.isnan(val) else "N/A")

    c6, c7 = st.columns(2)
    with c6:
        val = rates_df["10Y-2Y"].dropna().iloc[-1] if not rates_df.empty and "10Y-2Y" in rates_df.columns else np.nan
        st.metric("10Y-2Y", f"{val:.2f}" if not np.isnan(val) else "N/A")
    with c7:
        val = rates_df["10Y-3M"].dropna().iloc[-1] if not rates_df.empty and "10Y-3M" in rates_df.columns else np.nan
        st.metric("10Y-3M", f"{val:.2f}" if not np.isnan(val) else "N/A")

# ---------------------------------------------------------
# TAB 3: CREDIT
# ---------------------------------------------------------
with tab3:
    st.subheader("Credit / Financial Stress")
    st.plotly_chart(make_credit_chart(credit_df), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        val = credit_df["HY Spread"].dropna().iloc[-1] if not credit_df.empty and "HY Spread" in credit_df.columns else np.nan
        st.metric("HY Spread", f"{val:.2f}%" if not np.isnan(val) else "N/A")
    with c2:
        val = credit_df["FSI"].dropna().iloc[-1] if not credit_df.empty and "FSI" in credit_df.columns else np.nan
        st.metric("Financial Stress", f"{val:.2f}" if not np.isnan(val) else "N/A")
    with c3:
        val = credit_df["INDPRO_YoY"].dropna().iloc[-1] if not credit_df.empty and "INDPRO_YoY" in credit_df.columns else np.nan
        st.metric("INDPRO YoY", f"{val:.2f}%" if not np.isnan(val) else "N/A")
    with c4:
        val = credit_df["UNRATE"].dropna().iloc[-1] if not credit_df.empty and "UNRATE" in credit_df.columns else np.nan
        st.metric("Unemployment", f"{val:.2f}%" if not np.isnan(val) else "N/A")

# ---------------------------------------------------------
# TAB 4: MARKET
# ---------------------------------------------------------
with tab4:
    st.subheader("Market Dashboard")
    st.plotly_chart(make_market_chart(market_df), use_container_width=True)
    st.plotly_chart(make_vix_chart(market_df), use_container_width=True)

    cols = st.columns(4)
    for col, name in zip(cols, ["S&P 500", "Nasdaq", "QQQ", "VIX"]):
        with col:
            val = market_df[name].dropna().iloc[-1] if not market_df.empty and name in market_df.columns else np.nan
            st.metric(name, f"{val:,.2f}" if not np.isnan(val) else "N/A")

# ---------------------------------------------------------
# TAB 5: RISK SIGNAL
# ---------------------------------------------------------
with tab5:
    st.subheader("Macro Risk Signal")

    if regime == "RISK ON":
        st.success(f"Current Regime: {regime} | Score = {score}")
    elif regime == "RISK OFF":
        st.error(f"Current Regime: {regime} | Score = {score}")
    else:
        st.warning(f"Current Regime: {regime} | Score = {score}")

    st.markdown("### Why this signal?")
    if reasons:
        for r in reasons:
            st.write(f"- {r}")
    else:
        st.write("충분한 데이터가 없어 신호 설명이 제한됩니다.")

    st.markdown("### Practical Interpretation")
    st.markdown("""
    - **RISK ON**  
      유동성/신용/시장 추세가 위험자산에 우호적인 상태  
      → 성장주, 지수 ETF, 레버리지 일부 검토 가능

    - **NEUTRAL**  
      방향성이 약하거나 상충되는 상태  
      → 현금 비중 유지, 분할 매수/리밸런싱 적합

    - **RISK OFF**  
      유동성 악화, 신용 스프레드 확대, 금융 스트레스 증가  
      → 방어적 포지션, 현금/단기채/금 선호 가능
    """)

# ---------------------------------------------------------
# TAB 6: METHODOLOGY
# ---------------------------------------------------------
with tab6:
    st.subheader("How each parameter is calculated")
    st.markdown("""
    ### 1) Net Liquidity
    **Formula**

    `Net Liquidity = WALCL - TGA - RRP`

    - **WALCL**: Federal Reserve total assets
    - **TGA**: Treasury General Account
    - **RRP**: Reverse Repo usage

    **Meaning**
    - WALCL 상승 → 유동성 공급 증가
    - TGA 상승 → 재무부가 현금 흡수
    - RRP 상승 → 시장 유동성 흡수
    - Net Liquidity 상승 → 위험자산에 대체로 우호적

    ---

    ### 2) Yield Curve
    **Formulas**
    - `10Y-2Y = DGS10 - DGS2`
    - `10Y-3M = DGS10 - TB3MS`

    **Meaning**
    - 양(+)의 값: 정상적인 경기 확장 환경
    - 음(-)의 값: 역전, 침체 선행 신호 가능

    ---

    ### 3) SOFR / EFFR
    - **SOFR**: 담보 기반 초단기 자금금리
    - **EFFR**: 연방기금 실효금리

    **Meaning**
    - 정책금리 및 단기 금융여건을 보여줌

    ---

    ### 4) HY Spread
    - `BAMLH0A0HYM2`

    **Meaning**
    - 하이일드 채권의 추가 위험 프리미엄
    - 확대되면 신용리스크 증가
    - 축소되면 위험 선호 회복

    ---

    ### 5) STLFSI4
    - St. Louis Fed Financial Stress Index

    **Meaning**
    - 금융시장의 스트레스 수준
    - 높을수록 위험회피 환경

    ---

    ### 6) INDPRO / CPI / UNRATE
    - **INDPRO**: 산업생산
    - **CPI**: 소비자물가
    - **UNRATE**: 실업률

    **Meaning**
    - 경기/물가/고용의 기본 축
    - 리세션 및 스태그플레이션 여부 판단 보조

    ---

    ### 7) Risk Signal Logic
    현재 버전에서는 아래 항목을 점수화합니다.
    - Net Liquidity 3개월 변화율
    - 10Y-3M Yield Curve
    - HY Spread 수준
    - Financial Stress 수준
    - S&P 500 3개월 추세

    **Example**
    - 점수 높음 → RISK ON
    - 중간 → NEUTRAL
    - 낮음 → RISK OFF
    """)

# =========================================================
# 11) RAW DATA
# =========================================================
if show_raw_data:
    st.markdown("---")
    st.subheader("Raw Data")
    st.write("### Liquidity")
    st.dataframe(liq_df.tail(20), use_container_width=True)
    st.write("### Rates")
    st.dataframe(rates_df.tail(20), use_container_width=True)
    st.write("### Credit")
    st.dataframe(credit_df.tail(20), use_container_width=True)
    st.write("### Market")
    st.dataframe(market_df.tail(20), use_container_width=True)
