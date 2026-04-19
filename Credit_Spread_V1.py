# streamlit_app.py
# ============================================================
# Macro Dashboard: Liquidity + Credit
# - Liquidity raw chart (log scale)
# - Liquidity normalized chart (linear scale)
# - Credit monitoring section
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# Streamlit setup
# ============================================================
st.set_page_config(
    page_title="Macro Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Macro Dashboard")
st.caption("Liquidity components and credit-related monitoring charts.")


# ============================================================
# Labels
# ============================================================
SERIES_LABELS = {
    # Liquidity
    "FED_BALANCE_SHEET": "Fed Balance Sheet",
    "RRP": "Reverse Repo",
    "TGA": "Treasury General Account",
    "MMF_RETAIL": "Retail Money Market Fund Proxy",

    # Credit
    "BAMLH0A0HYM2": "High Yield OAS",
    "BAMLC0A0CM": "Investment Grade OAS",
    "STLFSI4": "St. Louis Fed Financial Stress Index",
    "CPFF": "Commercial Paper Funding Facility Rate",
    "TEDRATE": "TED Spread",
}


# ============================================================
# Optional sample data
# Replace this with your real FRED data loading logic
# ============================================================
@st.cache_data
def load_sample_data() -> pd.DataFrame:
    dates = pd.date_range(start="2020-01-01", periods=260, freq="W")
    np.random.seed(42)

    # Liquidity
    fed = 4200 + np.cumsum(np.random.normal(8, 12, len(dates)))
    rrp = 1800 + np.cumsum(np.random.normal(-3, 15, len(dates)))
    tga = 700 + np.cumsum(np.random.normal(1, 10, len(dates)))
    mmf = 900 + np.cumsum(np.random.normal(2, 8, len(dates)))

    # Credit
    hy = 4.2 + np.cumsum(np.random.normal(0.00, 0.05, len(dates)))
    ig = 1.4 + np.cumsum(np.random.normal(0.00, 0.02, len(dates)))
    stress = -0.5 + np.cumsum(np.random.normal(0.002, 0.03, len(dates)))
    cpff = 0.8 + np.cumsum(np.random.normal(0.001, 0.015, len(dates)))
    ted = 0.25 + np.cumsum(np.random.normal(0.000, 0.01, len(dates)))

    df = pd.DataFrame(
        {
            "date": dates,
            "FED_BALANCE_SHEET": fed * 1e3,
            "RRP": np.maximum(rrp * 1e3, 1),
            "TGA": np.maximum(tga * 1e3, 1),
            "MMF_RETAIL": np.maximum(mmf * 1e3, 1),
            "BAMLH0A0HYM2": np.maximum(hy, 0.01),
            "BAMLC0A0CM": np.maximum(ig, 0.01),
            "STLFSI4": stress,
            "CPFF": np.maximum(cpff, 0.01),
            "TEDRATE": ted,
        }
    )

    # Missing values for realism
    df.loc[10:12, "RRP"] = np.nan
    df.loc[40, "TGA"] = np.nan
    df.loc[80:82, "MMF_RETAIL"] = np.nan
    df.loc[30:31, "BAMLH0A0HYM2"] = np.nan
    df.loc[100, "TEDRATE"] = np.nan

    return df


# ============================================================
# Utility
# ============================================================
def compute_change(series: pd.Series, periods: int) -> float | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) <= periods:
        return None
    current = s.iloc[-1]
    prev = s.iloc[-(periods + 1)]
    if pd.isna(current) or pd.isna(prev) or prev == 0:
        return None
    return ((current / prev) - 1.0) * 100.0


def compute_diff(series: pd.Series, periods: int) -> float | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) <= periods:
        return None
    current = s.iloc[-1]
    prev = s.iloc[-(periods + 1)]
    if pd.isna(current) or pd.isna(prev):
        return None
    return current - prev


# ============================================================
# Raw multi-line chart
# ============================================================
def make_raw_chart(
    df: pd.DataFrame,
    columns: list[str],
    title: str = "",
    use_log_y: bool = False,
    yaxis_title: str = "Value",
) -> go.Figure:
    plot_df = df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"], errors="coerce")

    fig = go.Figure()

    for col in columns:
        if col not in plot_df.columns:
            continue

        s = pd.to_numeric(plot_df[col], errors="coerce")

        if use_log_y:
            s = s.where(s > 0, np.nan)

        if s.notna().sum() == 0:
            continue

        fig.add_trace(
            go.Scatter(
                x=plot_df["date"],
                y=s,
                mode="lines",
                name=SERIES_LABELS.get(col, col),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.2f}<extra>%{fullData.name}</extra>",
            )
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        legend_title_text="Series",
        xaxis_title="Date",
        yaxis_title=yaxis_title,
        height=520,
        margin=dict(l=40, r=20, t=60, b=40),
    )

    if use_log_y:
        fig.update_yaxes(type="log")

    return fig


# ============================================================
# Normalized chart
# ============================================================
def make_normalized_chart(
    df: pd.DataFrame,
    columns: list[str],
    title: str = "",
    base_value: float = 100.0,
    use_log_y: bool = False,
) -> go.Figure:
    plot_df = df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"], errors="coerce")

    norm_df = pd.DataFrame()
    norm_df["date"] = plot_df["date"]

    for col in columns:
        if col not in plot_df.columns:
            continue

        s = pd.to_numeric(plot_df[col], errors="coerce")

        valid = s.dropna()
        valid = valid[valid > 0]

        if valid.empty:
            norm_df[col] = np.nan
            continue

        base = valid.iloc[0]
        if pd.isna(base) or base == 0:
            norm_df[col] = np.nan
            continue

        normalized = (s / base) * base_value

        if use_log_y:
            normalized = normalized.where(normalized > 0, np.nan)

        norm_df[col] = normalized

    fig = go.Figure()

    for col in columns:
        if col not in norm_df.columns:
            continue

        s = pd.to_numeric(norm_df[col], errors="coerce")
        if s.notna().sum() == 0:
            continue

        fig.add_trace(
            go.Scatter(
                x=norm_df["date"],
                y=s,
                mode="lines",
                name=SERIES_LABELS.get(col, col),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.2f}<extra>%{fullData.name}</extra>",
            )
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        legend_title_text="Series",
        xaxis_title="Date",
        yaxis_title=f"Indexed to {base_value}",
        height=520,
        margin=dict(l=40, r=20, t=60, b=40),
    )

    if use_log_y:
        fig.update_yaxes(type="log")

    return fig


# ============================================================
# Credit summary table
# ============================================================
def build_credit_summary(df: pd.DataFrame, credit_cols: list[str]) -> pd.DataFrame:
    rows = []

    for col in credit_cols:
        if col not in df.columns:
            continue

        s = pd.to_numeric(df[col], errors="coerce")
        latest = s.dropna().iloc[-1] if s.dropna().shape[0] > 0 else np.nan
        diff_4w = compute_diff(s, 4)
        diff_13w = compute_diff(s, 13)
        change_13w = compute_change(s, 13)

        if col == "BAMLH0A0HYM2":
            signal = (
                "High Risk" if pd.notna(latest) and latest >= 6
                else "Watch" if pd.notna(latest) and latest >= 4.5
                else "Normal"
            )
        elif col == "BAMLC0A0CM":
            signal = (
                "High Risk" if pd.notna(latest) and latest >= 2.5
                else "Watch" if pd.notna(latest) and latest >= 1.8
                else "Normal"
            )
        elif col == "STLFSI4":
            signal = (
                "High Stress" if pd.notna(latest) and latest >= 1.0
                else "Watch" if pd.notna(latest) and latest >= 0.0
                else "Low Stress"
            )
        elif col == "CPFF":
            signal = (
                "Funding Stress" if pd.notna(latest) and latest >= 2.0
                else "Watch" if pd.notna(latest) and latest >= 1.0
                else "Normal"
            )
        elif col == "TEDRATE":
            signal = (
                "Funding Stress" if pd.notna(latest) and latest >= 0.75
                else "Watch" if pd.notna(latest) and latest >= 0.40
                else "Normal"
            )
        else:
            signal = "N/A"

        rows.append(
            {
                "Series": SERIES_LABELS.get(col, col),
                "Latest": round(latest, 3) if pd.notna(latest) else np.nan,
                "4W Diff": round(diff_4w, 3) if diff_4w is not None else np.nan,
                "13W Diff": round(diff_13w, 3) if diff_13w is not None else np.nan,
                "13W % Change": round(change_13w, 2) if change_13w is not None else np.nan,
                "Signal": signal,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Main app
# Replace with your real data loading section
# ============================================================
df = load_sample_data()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values("date")

# ============================================================
# Liquidity section
# ============================================================
st.header("Liquidity Monitoring")

liq_cols = ["FED_BALANCE_SHEET", "RRP", "TGA", "MMF_RETAIL"]
available_liq_cols = [col for col in liq_cols if col in df.columns]

if "date" not in df.columns:
    st.error("The DataFrame must contain a 'date' column.")
else:
    if not available_liq_cols:
        st.warning("No liquidity columns found.")
    else:
        liq_df = df[["date"] + available_liq_cols].copy()

        for col in available_liq_cols:
            liq_df[col] = pd.to_numeric(liq_df[col], errors="coerce")

        raw_df = liq_df.dropna(how="all", subset=available_liq_cols)
        norm_df = liq_df.dropna(how="all", subset=available_liq_cols)

        st.subheader("Raw Liquidity Components (Log Scale)")
        if raw_df.empty:
            st.info("No data available for the raw liquidity chart.")
        else:
            fig_raw_liq = make_raw_chart(
                raw_df,
                columns=available_liq_cols,
                title="Raw Liquidity Components (Log Scale)",
                use_log_y=True,
                yaxis_title="Level",
            )
            st.plotly_chart(fig_raw_liq, use_container_width=True)

        st.subheader("Normalized Liquidity Components")
        if norm_df.empty:
            st.info("No data available for the normalized liquidity chart.")
        else:
            fig_norm_liq = make_normalized_chart(
                norm_df,
                columns=available_liq_cols,
                title="Normalized Liquidity Components",
                base_value=100.0,
                use_log_y=False,
            )
            st.plotly_chart(fig_norm_liq, use_container_width=True)


# ============================================================
# Credit section
# ============================================================
st.header("Credit Monitoring")

credit_cols = ["BAMLH0A0HYM2", "BAMLC0A0CM", "STLFSI4", "CPFF", "TEDRATE"]
available_credit_cols = [col for col in credit_cols if col in df.columns]

if not available_credit_cols:
    st.warning("No credit-related columns found.")
else:
    credit_df = df[["date"] + available_credit_cols].copy()
    for col in available_credit_cols:
        credit_df[col] = pd.to_numeric(credit_df[col], errors="coerce")

    st.subheader("Credit Summary Table")
    credit_summary = build_credit_summary(credit_df, available_credit_cols)
    st.dataframe(credit_summary, use_container_width=True)

    spread_cols = [c for c in ["BAMLH0A0HYM2", "BAMLC0A0CM"] if c in credit_df.columns]
    funding_cols = [c for c in ["CPFF", "TEDRATE"] if c in credit_df.columns]
    stress_cols = [c for c in ["STLFSI4"] if c in credit_df.columns]

    if spread_cols:
        st.subheader("Credit Spreads")
        fig_spreads = make_raw_chart(
            credit_df.dropna(how="all", subset=spread_cols),
            columns=spread_cols,
            title="Credit Spreads",
            use_log_y=False,
            yaxis_title="Spread (%)",
        )
        st.plotly_chart(fig_spreads, use_container_width=True)

    if funding_cols:
        st.subheader("Funding Stress Indicators")
        fig_funding = make_raw_chart(
            credit_df.dropna(how="all", subset=funding_cols),
            columns=funding_cols,
            title="Funding Stress Indicators",
            use_log_y=False,
            yaxis_title="Rate / Spread (%)",
        )
        st.plotly_chart(fig_funding, use_container_width=True)

    if stress_cols:
        st.subheader("Financial Stress Index")
        fig_stress = make_raw_chart(
            credit_df.dropna(how="all", subset=stress_cols),
            columns=stress_cols,
            title="Financial Stress Index",
            use_log_y=False,
            yaxis_title="Index Level",
        )
        st.plotly_chart(fig_stress, use_container_width=True)

    st.subheader("Normalized Credit Indicators")
    fig_norm_credit = make_normalized_chart(
        credit_df.dropna(how="all", subset=available_credit_cols),
        columns=available_credit_cols,
        title="Normalized Credit Indicators",
        base_value=100.0,
        use_log_y=False,
    )
    st.plotly_chart(fig_norm_credit, use_container_width=True)


# ============================================================
# Data preview
# ============================================================
st.header("Latest Data Preview")
preview_cols = ["date"] + available_liq_cols + available_credit_cols
st.dataframe(df[preview_cols].tail(12), use_container_width=True)
