# streamlit_app.py
# ============================================================
# SCHD Dividend -> QQQ / SPY Reinvestment Dashboard
# Single-page dashboard style
#
# Updated:
# - Final summary updates when the month range changes
# - Summary follows the selected month window (df_view)
#
# Run:
#   streamlit run streamlit_app.py
#
# Install:
#   pip install streamlit pandas numpy plotly
# ============================================================

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="SCHD Reinvestment Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("📈 SCHD Dividend Reinvestment Dashboard")
st.caption(
    "Single-page dashboard for SCHD dividend cash flow, QQQ/SPY reinvestment, "
    "and long-term compounded asset growth."
)

# ============================================================
# Constants
# ============================================================
EOK_KRW = 100_000_000
MAN_KRW = 10_000

# ============================================================
# Data class
# ============================================================
@dataclass
class SimulationInputs:
    initial_krw: float
    fx_rate: float
    schd_yield_annual: float
    tax_rate: float
    schd_dividend_growth_annual: float
    qqq_weight: float
    spy_weight: float
    qqq_cagr: float
    spy_cagr: float
    horizon_years: int
    payout_mode: str


# ============================================================
# Utility functions
# ============================================================
def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def annual_to_monthly_rate(annual_rate: float) -> float:
    return (1 + annual_rate) ** (1 / 12) - 1


def krw_to_usd(krw: float, fx_rate: float) -> float:
    return safe_div(krw, fx_rate)


def usd_to_krw(usd: float, fx_rate: float) -> float:
    return usd * fx_rate


def krw_to_eok(krw: float) -> float:
    return krw / EOK_KRW


def krw_to_man(krw: float) -> float:
    return krw / MAN_KRW


def format_krw(value: float) -> str:
    return f"₩{value:,.0f}"


def format_usd(value: float) -> str:
    return f"${value:,.2f}"


def format_eok(value: float) -> str:
    return f"{value:,.2f} 억원"


def format_man(value: float) -> str:
    return f"{value:,.1f} 만원"


# ============================================================
# Simulation engine
# ============================================================
def build_simulation_dataframe(inputs: SimulationInputs) -> pd.DataFrame:
    months = inputs.horizon_years * 12
    initial_usd = krw_to_usd(inputs.initial_krw, inputs.fx_rate)

    qqq_monthly_rate = annual_to_monthly_rate(inputs.qqq_cagr)
    spy_monthly_rate = annual_to_monthly_rate(inputs.spy_cagr)
    schd_div_growth_monthly = annual_to_monthly_rate(inputs.schd_dividend_growth_annual)

    qqq_value = 0.0
    spy_value = 0.0
    cumulative_net_dividend = 0.0
    cumulative_tax = 0.0

    rows = []

    for month in range(1, months + 1):
        effective_annual_yield = inputs.schd_yield_annual * ((1 + schd_div_growth_monthly) ** (month - 1))

        if inputs.payout_mode == "Monthly normalized":
            gross_dividend_usd = initial_usd * effective_annual_yield / 12.0
        else:
            gross_dividend_usd = initial_usd * effective_annual_yield / 4.0 if month % 3 == 0 else 0.0

        tax_usd = gross_dividend_usd * inputs.tax_rate
        net_dividend_usd = gross_dividend_usd - tax_usd

        qqq_contribution_usd = net_dividend_usd * inputs.qqq_weight
        spy_contribution_usd = net_dividend_usd * inputs.spy_weight

        qqq_value = qqq_value * (1 + qqq_monthly_rate) + qqq_contribution_usd
        spy_value = spy_value * (1 + spy_monthly_rate) + spy_contribution_usd

        reinvested_value_usd = qqq_value + spy_value
        schd_principal_usd = initial_usd
        total_asset_usd = schd_principal_usd + reinvested_value_usd

        cumulative_net_dividend += net_dividend_usd
        cumulative_tax += tax_usd

        rows.append(
            {
                "Month": month,
                "Year": math.ceil(month / 12),

                "SCHD Principal (USD)": schd_principal_usd,
                "SCHD Principal (KRW)": usd_to_krw(schd_principal_usd, inputs.fx_rate),
                "SCHD Principal (Eok KRW)": krw_to_eok(usd_to_krw(schd_principal_usd, inputs.fx_rate)),

                "Gross Dividend (USD)": gross_dividend_usd,
                "Gross Dividend (KRW)": usd_to_krw(gross_dividend_usd, inputs.fx_rate),
                "Gross Dividend (Man KRW)": krw_to_man(usd_to_krw(gross_dividend_usd, inputs.fx_rate)),

                "Tax (USD)": tax_usd,
                "Tax (KRW)": usd_to_krw(tax_usd, inputs.fx_rate),

                "Net Dividend (USD)": net_dividend_usd,
                "Net Dividend (KRW)": usd_to_krw(net_dividend_usd, inputs.fx_rate),
                "Net Dividend (Man KRW)": krw_to_man(usd_to_krw(net_dividend_usd, inputs.fx_rate)),

                "Cumulative Net Dividend (USD)": cumulative_net_dividend,
                "Cumulative Net Dividend (KRW)": usd_to_krw(cumulative_net_dividend, inputs.fx_rate),
                "Cumulative Tax (USD)": cumulative_tax,
                "Cumulative Tax (KRW)": usd_to_krw(cumulative_tax, inputs.fx_rate),

                "QQQ Contribution (USD)": qqq_contribution_usd,
                "QQQ Contribution (KRW)": usd_to_krw(qqq_contribution_usd, inputs.fx_rate),

                "SPY Contribution (USD)": spy_contribution_usd,
                "SPY Contribution (KRW)": usd_to_krw(spy_contribution_usd, inputs.fx_rate),

                "QQQ Value (USD)": qqq_value,
                "QQQ Value (KRW)": usd_to_krw(qqq_value, inputs.fx_rate),
                "QQQ Value (Eok KRW)": krw_to_eok(usd_to_krw(qqq_value, inputs.fx_rate)),

                "SPY Value (USD)": spy_value,
                "SPY Value (KRW)": usd_to_krw(spy_value, inputs.fx_rate),
                "SPY Value (Eok KRW)": krw_to_eok(usd_to_krw(spy_value, inputs.fx_rate)),

                "Reinvested Value (USD)": reinvested_value_usd,
                "Reinvested Value (KRW)": usd_to_krw(reinvested_value_usd, inputs.fx_rate),
                "Reinvested Value (Eok KRW)": krw_to_eok(usd_to_krw(reinvested_value_usd, inputs.fx_rate)),

                "Total Asset (USD)": total_asset_usd,
                "Total Asset (KRW)": usd_to_krw(total_asset_usd, inputs.fx_rate),
                "Total Asset (Eok KRW)": krw_to_eok(usd_to_krw(total_asset_usd, inputs.fx_rate)),

                "Monthly Dividend Cash Flow (USD)": net_dividend_usd,
                "Monthly Dividend Cash Flow (KRW)": usd_to_krw(net_dividend_usd, inputs.fx_rate),
                "Monthly Dividend Cash Flow (Man KRW)": krw_to_man(usd_to_krw(net_dividend_usd, inputs.fx_rate)),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Summary helpers
# ============================================================
def build_summary(df: pd.DataFrame) -> Dict[str, float]:
    last = df.iloc[-1]
    return {
        "month": int(last["Month"]),
        "total_asset_usd": float(last["Total Asset (USD)"]),
        "total_asset_krw": float(last["Total Asset (KRW)"]),
        "total_asset_eok": float(last["Total Asset (Eok KRW)"]),
        "monthly_dividend_usd": float(last["Monthly Dividend Cash Flow (USD)"]),
        "monthly_dividend_krw": float(last["Monthly Dividend Cash Flow (KRW)"]),
        "monthly_dividend_man": float(last["Monthly Dividend Cash Flow (Man KRW)"]),
    }


# ============================================================
# Chart builders
# ============================================================
def make_total_asset_chart(df: pd.DataFrame, display_mode: str) -> go.Figure:
    fig = go.Figure()

    if display_mode == "KRW":
        fig.add_trace(
            go.Bar(
                x=df["Month"],
                y=df["SCHD Principal (Eok KRW)"],
                name="SCHD Principal",
            )
        )
        fig.add_trace(
            go.Bar(
                x=df["Month"],
                y=df["Reinvested Value (Eok KRW)"],
                name="Reinvested Assets",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["Month"],
                y=df["Monthly Dividend Cash Flow (Man KRW)"],
                mode="lines",
                name="Monthly Dividend",
                yaxis="y2",
                line=dict(width=3),
            )
        )
        fig.update_layout(
            title="Total Asset Growth (Eok KRW)",
            yaxis=dict(title="Asset Value (Eok KRW)"),
            yaxis2=dict(
                title="Monthly Dividend (Man KRW)",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
        )
    else:
        fig.add_trace(
            go.Bar(
                x=df["Month"],
                y=df["SCHD Principal (USD)"],
                name="SCHD Principal",
            )
        )
        fig.add_trace(
            go.Bar(
                x=df["Month"],
                y=df["Reinvested Value (USD)"],
                name="Reinvested Assets",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["Month"],
                y=df["Monthly Dividend Cash Flow (USD)"],
                mode="lines",
                name="Monthly Dividend",
                yaxis="y2",
                line=dict(width=3),
            )
        )
        fig.update_layout(
            title="Total Asset Growth (USD)",
            yaxis=dict(title="Asset Value (USD)"),
            yaxis2=dict(
                title="Monthly Dividend (USD)",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
        )

    fig.update_layout(
        barmode="stack",
        height=500,
        template="plotly_white",
        margin=dict(l=50, r=70, t=70, b=50),
        legend=dict(orientation="v", x=1.02, y=1),
        xaxis=dict(title="Month"),
    )
    return fig


def make_dividend_chart(df: pd.DataFrame, display_mode: str) -> go.Figure:
    fig = go.Figure()

    if display_mode == "KRW":
        fig.add_trace(
            go.Scatter(
                x=df["Month"],
                y=df["Monthly Dividend Cash Flow (Man KRW)"],
                mode="lines",
                name="Monthly Dividend",
                line=dict(width=3),
            )
        )
        fig.update_layout(
            title="Monthly Dividend Cash Flow (Man KRW)",
            yaxis=dict(title="Cash Flow (Man KRW)"),
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df["Month"],
                y=df["Monthly Dividend Cash Flow (USD)"],
                mode="lines",
                name="Monthly Dividend",
                line=dict(width=3),
            )
        )
        fig.update_layout(
            title="Monthly Dividend Cash Flow (USD)",
            yaxis=dict(title="Cash Flow (USD)"),
        )

    fig.update_layout(
        height=340,
        template="plotly_white",
        margin=dict(l=50, r=20, t=70, b=50),
        xaxis=dict(title="Month"),
        showlegend=False,
    )
    return fig


# ============================================================
# Tables
# ============================================================
def build_display_tables(df: pd.DataFrame, display_mode: str) -> dict[str, pd.DataFrame]:
    if display_mode == "KRW":
        main_table = df[
            [
                "Month",
                "Year",
                "SCHD Principal (Eok KRW)",
                "Reinvested Value (Eok KRW)",
                "Total Asset (Eok KRW)",
                "Monthly Dividend Cash Flow (Man KRW)",
                "QQQ Value (Eok KRW)",
                "SPY Value (Eok KRW)",
            ]
        ].copy()

        dividend_table = df[
            [
                "Month",
                "Year",
                "Gross Dividend (Man KRW)",
                "Tax (KRW)",
                "Net Dividend (Man KRW)",
                "Cumulative Net Dividend (KRW)",
            ]
        ].copy()
    else:
        main_table = df[
            [
                "Month",
                "Year",
                "SCHD Principal (USD)",
                "Reinvested Value (USD)",
                "Total Asset (USD)",
                "Monthly Dividend Cash Flow (USD)",
                "QQQ Value (USD)",
                "SPY Value (USD)",
            ]
        ].copy()

        dividend_table = df[
            [
                "Month",
                "Year",
                "Gross Dividend (USD)",
                "Tax (USD)",
                "Net Dividend (USD)",
                "Cumulative Net Dividend (USD)",
            ]
        ].copy()

    for table in [main_table, dividend_table]:
        numeric_cols = table.select_dtypes(include=[np.number]).columns
        table[numeric_cols] = table[numeric_cols].round(4)

    return {"main": main_table, "dividend": dividend_table}


# ============================================================
# Sidebar controls
# ============================================================
st.sidebar.header("Controls")

display_mode = st.sidebar.radio(
    "Display Mode",
    ["KRW", "USD"],
    index=0,
    horizontal=True,
)

preset = st.sidebar.selectbox(
    "Scenario Preset",
    ["Custom", "Conservative", "Base", "Optimistic"],
    index=1,
)

default_qqq_cagr = 0.07
default_spy_cagr = 0.06

if preset == "Base":
    default_qqq_cagr = 0.10
    default_spy_cagr = 0.08
elif preset == "Optimistic":
    default_qqq_cagr = 0.13
    default_spy_cagr = 0.10

initial_krw = st.sidebar.number_input(
    "Initial SCHD Amount (KRW)",
    min_value=1_000_000,
    value=100_000_000,
    step=1_000_000,
)

fx_rate = st.sidebar.number_input(
    "USD/KRW Exchange Rate",
    min_value=500.0,
    max_value=3000.0,
    value=1483.9,
    step=1.0,
)

schd_yield_annual_pct = st.sidebar.number_input(
    "SCHD Annual Dividend Yield (%)",
    min_value=0.0,
    max_value=20.0,
    value=3.44,
    step=0.01,
)

tax_rate_pct = st.sidebar.number_input(
    "Dividend Tax Rate (%)",
    min_value=0.0,
    max_value=50.0,
    value=15.0,
    step=0.1,
)

schd_dividend_growth_annual_pct = st.sidebar.number_input(
    "SCHD Dividend Growth Rate (%)",
    min_value=0.0,
    max_value=20.0,
    value=0.0,
    step=0.1,
)

payout_mode = st.sidebar.radio(
    "Dividend Payout Mode",
    ["Monthly normalized", "Actual quarterly payout"],
    index=0,
)

qqq_weight_pct = st.sidebar.slider(
    "QQQ Allocation (%)",
    min_value=0,
    max_value=100,
    value=50,
    step=1,
)

spy_weight_pct = 100 - qqq_weight_pct
st.sidebar.metric("SPY Allocation (%)", f"{spy_weight_pct}%")

qqq_cagr_pct = st.sidebar.number_input(
    "QQQ CAGR (%)",
    min_value=0.0,
    max_value=30.0,
    value=default_qqq_cagr * 100,
    step=0.1,
)

spy_cagr_pct = st.sidebar.number_input(
    "SPY CAGR (%)",
    min_value=0.0,
    max_value=30.0,
    value=default_spy_cagr * 100,
    step=0.1,
)

horizon_years = st.sidebar.selectbox(
    "Horizon",
    [10, 20, 30],
    index=2,
)

show_tables = st.sidebar.checkbox("Show raw tables", value=True)

# ============================================================
# Compute full data
# ============================================================
inputs = SimulationInputs(
    initial_krw=float(initial_krw),
    fx_rate=float(fx_rate),
    schd_yield_annual=float(schd_yield_annual_pct) / 100.0,
    tax_rate=float(tax_rate_pct) / 100.0,
    schd_dividend_growth_annual=float(schd_dividend_growth_annual_pct) / 100.0,
    qqq_weight=float(qqq_weight_pct) / 100.0,
    spy_weight=float(spy_weight_pct) / 100.0,
    qqq_cagr=float(qqq_cagr_pct) / 100.0,
    spy_cagr=float(spy_cagr_pct) / 100.0,
    horizon_years=int(horizon_years),
    payout_mode=payout_mode,
)

df = build_simulation_dataframe(inputs)

# ============================================================
# Month range slider
# ============================================================
max_month = inputs.horizon_years * 12
month_range = st.slider(
    "Month Range",
    min_value=1,
    max_value=max_month,
    value=(1, max_month),
    step=1,
)

selected_start, selected_end = month_range

df_view = df[(df["Month"] >= selected_start) & (df["Month"] <= selected_end)].copy()

# Summary now follows the selected range
summary = build_summary(df_view)

# Tables
tables = build_display_tables(df, display_mode)
tables_view = build_display_tables(df_view, display_mode)

# ============================================================
# Top control-like header
# ============================================================
top_left, top_mid, top_right = st.columns([1.4, 1.2, 1.0])

with top_mid:
    mode_col1, mode_col2 = st.columns(2)
    with mode_col1:
        st.button("한국어 (KRW)", disabled=(display_mode == "KRW"), use_container_width=True)
    with mode_col2:
        st.button("English (USD)", disabled=(display_mode == "USD"), use_container_width=True)

with top_right:
    if display_mode == "KRW":
        summary_text = (
            f"**Selected Range: {selected_start}m ~ {selected_end}m**\n\n"
            f"Final Asset: {format_eok(summary['total_asset_eok'])}\n\n"
            f"Monthly Dividend: {format_man(summary['monthly_dividend_man'])}"
        )
    else:
        summary_text = (
            f"**Selected Range: {selected_start}m ~ {selected_end}m**\n\n"
            f"Final Asset: {format_usd(summary['total_asset_usd'])}\n\n"
            f"Monthly Dividend: {format_usd(summary['monthly_dividend_usd'])}"
        )

    st.markdown(
        f"""
        <div style="
            border:1px solid #777;
            border-radius:2px;
            padding:12px;
            background-color:white;
            font-size:16px;
            line-height:1.6;
            text-align:center;
        ">
            {summary_text.replace(chr(10), '<br>')}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# Main charts
# ============================================================
fig_asset = make_total_asset_chart(df_view, display_mode)
st.plotly_chart(fig_asset, use_container_width=True)

fig_dividend = make_dividend_chart(df_view, display_mode)
st.plotly_chart(fig_dividend, use_container_width=True)

# ============================================================
# Raw tables
# ============================================================
if show_tables:
    with st.expander("Raw Tables", expanded=False):
        st.markdown("### Asset Table")
        st.dataframe(tables_view["main"], use_container_width=True, height=300)

        st.markdown("### Dividend Table")
        st.dataframe(tables_view["dividend"], use_container_width=True, height=300)

# ============================================================
# CSV download
# ============================================================
st.subheader("Download CSV")

csv_main = tables["main"].to_csv(index=False).encode("utf-8")
csv_dividend = tables["dividend"].to_csv(index=False).encode("utf-8")

d1, d2 = st.columns(2)
with d1:
    st.download_button(
        "Download Asset Table CSV",
        data=csv_main,
        file_name="asset_table.csv",
        mime="text/csv",
        use_container_width=True,
    )
with d2:
    st.download_button(
        "Download Dividend Table CSV",
        data=csv_dividend,
        file_name="dividend_table.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ============================================================
# Notes
# ============================================================
with st.expander("Model Notes / Assumptions"):
    st.markdown(
        """
- SCHD principal is kept constant in this version.
- Dividends are generated from SCHD and reinvested into QQQ / SPY.
- Reinvested assets are compounded monthly using the QQQ / SPY CAGR assumptions.
- Total Asset = SCHD Principal + Reinvested Assets.
- The top summary box updates based on the selected month range.
- KRW mode uses:
  - **Eok KRW** for asset values
  - **Man KRW** for monthly dividends
- USD mode uses:
  - **USD** for both asset values and monthly dividends
- This is a scenario planning dashboard, not a live market pricing tool.
        """
    )
