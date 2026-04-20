# streamlit_app.py
# ============================================================
# SCHD Dividend -> QQQ/SPY DCA Simulator
# Full English-based code
#
# Features
# - Adjustable SCHD initial amount in KRW
# - USD conversion via editable FX rate
# - SCHD monthly / quarterly dividend simulation
# - Tax-adjusted dividend cash flow
# - QQQ / SPY allocation slider
# - 10Y / 20Y / 30Y horizon comparison
# - Separate Dashboard / Charts / Tables tabs
# - Dual-axis charts (USD left / KRW right)
# - Plot titles placed outside charts to avoid overlap
# - Related table below each chart
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
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="SCHD Dividend DCA Simulator",
    page_icon="💵",
    layout="wide",
)

st.title("💵 SCHD Dividend → QQQ / SPY DCA Simulator")
st.caption(
    "Plan SCHD dividend cash flow, cumulative contributions, and long-term "
    "QQQ / SPY compounding with adjustable assumptions."
)

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


def format_currency(value: float, currency: str = "USD") -> str:
    if currency == "USD":
        return f"${value:,.2f}"
    if currency == "KRW":
        return f"₩{value:,.0f}"
    return f"{value:,.2f}"


# ============================================================
# Core simulation
# ============================================================
def build_simulation_dataframe(inputs: SimulationInputs) -> pd.DataFrame:
    months = inputs.horizon_years * 12
    initial_usd = krw_to_usd(inputs.initial_krw, inputs.fx_rate)

    qqq_monthly_rate = annual_to_monthly_rate(inputs.qqq_cagr)
    spy_monthly_rate = annual_to_monthly_rate(inputs.spy_cagr)
    schd_div_growth_monthly = annual_to_monthly_rate(inputs.schd_dividend_growth_annual)

    qqq_value = 0.0
    spy_value = 0.0
    cumulative_gross_dividend = 0.0
    cumulative_tax = 0.0
    cumulative_net_dividend = 0.0
    cumulative_qqq_contribution = 0.0
    cumulative_spy_contribution = 0.0

    rows = []

    for month in range(1, months + 1):
        effective_annual_yield = inputs.schd_yield_annual * ((1 + schd_div_growth_monthly) ** (month - 1))

        if inputs.payout_mode == "Monthly normalized":
            gross_dividend_usd = initial_usd * effective_annual_yield / 12.0
        else:
            gross_dividend_usd = initial_usd * effective_annual_yield / 4.0 if month % 3 == 0 else 0.0

        tax_usd = gross_dividend_usd * inputs.tax_rate
        net_dividend_usd = gross_dividend_usd - tax_usd

        qqq_contribution = net_dividend_usd * inputs.qqq_weight
        spy_contribution = net_dividend_usd * inputs.spy_weight

        qqq_value = qqq_value * (1 + qqq_monthly_rate) + qqq_contribution
        spy_value = spy_value * (1 + spy_monthly_rate) + spy_contribution

        cumulative_gross_dividend += gross_dividend_usd
        cumulative_tax += tax_usd
        cumulative_net_dividend += net_dividend_usd
        cumulative_qqq_contribution += qqq_contribution
        cumulative_spy_contribution += spy_contribution

        total_portfolio_value = qqq_value + spy_value
        total_contribution = cumulative_qqq_contribution + cumulative_spy_contribution
        total_gain = total_portfolio_value - total_contribution
        total_return_pct = safe_div(total_gain, total_contribution) * 100 if total_contribution > 0 else 0.0

        rows.append(
            {
                "Month": month,
                "Year": math.ceil(month / 12),
                "Initial Principal (KRW)": inputs.initial_krw,
                "Initial Principal (USD)": initial_usd,
                "SCHD Annual Yield (%)": effective_annual_yield * 100,
                "Gross Dividend (USD)": gross_dividend_usd,
                "Gross Dividend (KRW)": usd_to_krw(gross_dividend_usd, inputs.fx_rate),
                "Tax (USD)": tax_usd,
                "Tax (KRW)": usd_to_krw(tax_usd, inputs.fx_rate),
                "Net Dividend (USD)": net_dividend_usd,
                "Net Dividend (KRW)": usd_to_krw(net_dividend_usd, inputs.fx_rate),
                "Cumulative Gross Dividend (USD)": cumulative_gross_dividend,
                "Cumulative Gross Dividend (KRW)": usd_to_krw(cumulative_gross_dividend, inputs.fx_rate),
                "Cumulative Tax (USD)": cumulative_tax,
                "Cumulative Tax (KRW)": usd_to_krw(cumulative_tax, inputs.fx_rate),
                "Cumulative Net Dividend (USD)": cumulative_net_dividend,
                "Cumulative Net Dividend (KRW)": usd_to_krw(cumulative_net_dividend, inputs.fx_rate),
                "QQQ Contribution (USD)": qqq_contribution,
                "QQQ Contribution (KRW)": usd_to_krw(qqq_contribution, inputs.fx_rate),
                "SPY Contribution (USD)": spy_contribution,
                "SPY Contribution (KRW)": usd_to_krw(spy_contribution, inputs.fx_rate),
                "Cumulative QQQ Contribution (USD)": cumulative_qqq_contribution,
                "Cumulative QQQ Contribution (KRW)": usd_to_krw(cumulative_qqq_contribution, inputs.fx_rate),
                "Cumulative SPY Contribution (USD)": cumulative_spy_contribution,
                "Cumulative SPY Contribution (KRW)": usd_to_krw(cumulative_spy_contribution, inputs.fx_rate),
                "QQQ Value (USD)": qqq_value,
                "QQQ Value (KRW)": usd_to_krw(qqq_value, inputs.fx_rate),
                "SPY Value (USD)": spy_value,
                "SPY Value (KRW)": usd_to_krw(spy_value, inputs.fx_rate),
                "Total Portfolio Value (USD)": total_portfolio_value,
                "Total Portfolio Value (KRW)": usd_to_krw(total_portfolio_value, inputs.fx_rate),
                "Total Contribution (USD)": total_contribution,
                "Total Contribution (KRW)": usd_to_krw(total_contribution, inputs.fx_rate),
                "Total Gain (USD)": total_gain,
                "Total Gain (KRW)": usd_to_krw(total_gain, inputs.fx_rate),
                "Total Return (%)": total_return_pct,
            }
        )

    df = pd.DataFrame(rows)

    df["Total Monthly Contribution (USD)"] = df["QQQ Contribution (USD)"] + df["SPY Contribution (USD)"]
    df["Total Monthly Contribution (KRW)"] = df["QQQ Contribution (KRW)"] + df["SPY Contribution (KRW)"]
    df["Total Cumulative Contribution (USD)"] = (
        df["Cumulative QQQ Contribution (USD)"] + df["Cumulative SPY Contribution (USD)"]
    )
    df["Total Cumulative Contribution (KRW)"] = (
        df["Cumulative QQQ Contribution (KRW)"] + df["Cumulative SPY Contribution (KRW)"]
    )

    return df


def build_summary_metrics(df: pd.DataFrame) -> Dict[str, float]:
    first_year = df[df["Year"] == 1]
    last_row = df.iloc[-1]

    return {
        "initial_principal_usd": float(df["Initial Principal (USD)"].iloc[0]),
        "initial_principal_krw": float(df["Initial Principal (KRW)"].iloc[0]),
        "first_year_gross_dividend_usd": float(first_year["Gross Dividend (USD)"].sum()),
        "first_year_net_dividend_usd": float(first_year["Net Dividend (USD)"].sum()),
        "first_year_net_dividend_krw": float(first_year["Net Dividend (KRW)"].sum()),
        "monthly_avg_net_dividend_usd": float(first_year["Net Dividend (USD)"].mean()),
        "monthly_avg_net_dividend_krw": float(first_year["Net Dividend (KRW)"].mean()),
        "final_cumulative_net_dividend_usd": float(last_row["Cumulative Net Dividend (USD)"]),
        "final_cumulative_net_dividend_krw": float(last_row["Cumulative Net Dividend (KRW)"]),
        "final_portfolio_value_usd": float(last_row["Total Portfolio Value (USD)"]),
        "final_portfolio_value_krw": float(last_row["Total Portfolio Value (KRW)"]),
        "final_total_gain_usd": float(last_row["Total Gain (USD)"]),
        "final_total_gain_krw": float(last_row["Total Gain (KRW)"]),
        "final_total_return_pct": float(last_row["Total Return (%)"]),
    }


def build_horizon_comparison(
    base_inputs: SimulationInputs,
    horizons: Tuple[int, ...] = (10, 20, 30),
) -> pd.DataFrame:
    rows = []

    for horizon in horizons:
        temp_inputs = SimulationInputs(
            initial_krw=base_inputs.initial_krw,
            fx_rate=base_inputs.fx_rate,
            schd_yield_annual=base_inputs.schd_yield_annual,
            tax_rate=base_inputs.tax_rate,
            schd_dividend_growth_annual=base_inputs.schd_dividend_growth_annual,
            qqq_weight=base_inputs.qqq_weight,
            spy_weight=base_inputs.spy_weight,
            qqq_cagr=base_inputs.qqq_cagr,
            spy_cagr=base_inputs.spy_cagr,
            horizon_years=horizon,
            payout_mode=base_inputs.payout_mode,
        )

        temp_df = build_simulation_dataframe(temp_inputs)
        last_row = temp_df.iloc[-1]

        rows.append(
            {
                "Horizon (Years)": horizon,
                "Cumulative Net Dividend (USD)": last_row["Cumulative Net Dividend (USD)"],
                "Cumulative Net Dividend (KRW)": last_row["Cumulative Net Dividend (KRW)"],
                "Total Contribution (USD)": last_row["Total Contribution (USD)"],
                "Total Contribution (KRW)": last_row["Total Contribution (KRW)"],
                "QQQ Value (USD)": last_row["QQQ Value (USD)"],
                "QQQ Value (KRW)": last_row["QQQ Value (KRW)"],
                "SPY Value (USD)": last_row["SPY Value (USD)"],
                "SPY Value (KRW)": last_row["SPY Value (KRW)"],
                "Total Portfolio Value (USD)": last_row["Total Portfolio Value (USD)"],
                "Total Portfolio Value (KRW)": last_row["Total Portfolio Value (KRW)"],
                "Total Gain (USD)": last_row["Total Gain (USD)"],
                "Total Gain (KRW)": last_row["Total Gain (KRW)"],
                "Total Return (%)": last_row["Total Return (%)"],
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Table helpers
# ============================================================
def prepare_plot_related_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    temp = df.copy()

    dividend_table = temp[
        [
            "Month",
            "Year",
            "Net Dividend (USD)",
            "Net Dividend (KRW)",
            "Cumulative Net Dividend (USD)",
            "Cumulative Net Dividend (KRW)",
        ]
    ].copy()

    contribution_table = temp[
        [
            "Month",
            "Year",
            "Total Monthly Contribution (USD)",
            "Total Monthly Contribution (KRW)",
            "Total Cumulative Contribution (USD)",
            "Total Cumulative Contribution (KRW)",
        ]
    ].copy()

    qqq_table = temp[
        [
            "Month",
            "Year",
            "QQQ Contribution (USD)",
            "QQQ Contribution (KRW)",
            "Cumulative QQQ Contribution (USD)",
            "Cumulative QQQ Contribution (KRW)",
        ]
    ].copy()

    spy_table = temp[
        [
            "Month",
            "Year",
            "SPY Contribution (USD)",
            "SPY Contribution (KRW)",
            "Cumulative SPY Contribution (USD)",
            "Cumulative SPY Contribution (KRW)",
        ]
    ].copy()

    growth_table = temp[
        [
            "Month",
            "Year",
            "QQQ Value (USD)",
            "QQQ Value (KRW)",
            "SPY Value (USD)",
            "SPY Value (KRW)",
            "Total Portfolio Value (USD)",
            "Total Portfolio Value (KRW)",
            "Total Contribution (USD)",
            "Total Contribution (KRW)",
            "Total Gain (USD)",
            "Total Gain (KRW)",
            "Total Return (%)",
        ]
    ].copy()

    for tdf in [dividend_table, contribution_table, qqq_table, spy_table, growth_table]:
        numeric_cols = tdf.select_dtypes(include=[np.number]).columns
        tdf[numeric_cols] = tdf[numeric_cols].round(2)

    return {
        "dividend": dividend_table,
        "contribution": contribution_table,
        "qqq": qqq_table,
        "spy": spy_table,
        "growth": growth_table,
    }


def show_related_table(
    title: str,
    table_df: pd.DataFrame,
    show_full_table: bool = False,
    default_rows: int = 12,
    height: int = 260,
) -> None:
    st.markdown(f"**{title}**")
    if show_full_table:
        st.dataframe(table_df, use_container_width=True, height=height)
    else:
        st.dataframe(table_df.tail(default_rows), use_container_width=True, height=height)


# ============================================================
# Chart functions
# ============================================================
def make_monthly_bar_line_dual_axis_chart(
    df: pd.DataFrame,
    x_col: str,
    bar_usd_col: str,
    line_usd_col: str,
    bar_krw_col: str,
    line_krw_col: str,
    title: Optional[str] = None,
    left_title: str = "USD",
    right_title: str = "KRW",
    show_legend: bool = True,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=df[x_col],
            y=df[bar_usd_col],
            name=bar_usd_col,
            opacity=0.55,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[line_usd_col],
            mode="lines",
            name=line_usd_col,
            line=dict(width=2.5),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(
            x=df[x_col],
            y=df[bar_krw_col],
            name=bar_krw_col,
            opacity=0.22,
            visible="legendonly",
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[line_krw_col],
            mode="lines",
            name=line_krw_col,
            line=dict(width=2, dash="dot"),
            visible="legendonly",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=title,
        height=430,
        template="plotly_white",
        barmode="overlay",
        showlegend=show_legend,
        margin=dict(l=60, r=70, t=25 if title is None else 80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
    )

    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text=left_title, secondary_y=False)
    fig.update_yaxes(title_text=right_title, secondary_y=True)

    return fig


def make_dual_axis_growth_chart(
    df: pd.DataFrame,
    title: Optional[str] = None,
    show_legend: bool = True,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=df["Month"],
            y=df["QQQ Value (USD)"],
            name="QQQ Value (USD)",
            opacity=0.35,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(
            x=df["Month"],
            y=df["SPY Value (USD)"],
            name="SPY Value (USD)",
            opacity=0.35,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df["Month"],
            y=df["Total Portfolio Value (USD)"],
            mode="lines",
            name="Total Portfolio Value (USD)",
            line=dict(width=3),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df["Month"],
            y=df["Total Portfolio Value (KRW)"],
            mode="lines",
            name="Total Portfolio Value (KRW)",
            line=dict(width=2, dash="dot"),
            visible="legendonly",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=title,
        height=430,
        template="plotly_white",
        barmode="group",
        showlegend=show_legend,
        margin=dict(l=60, r=70, t=25 if title is None else 80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
    )

    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="USD", secondary_y=False)
    fig.update_yaxes(title_text="KRW", secondary_y=True)

    return fig


def make_horizon_bar_line_dual_axis_chart(
    horizon_df: pd.DataFrame,
    title: Optional[str] = None,
    show_legend: bool = True,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=horizon_df["Horizon (Years)"],
            y=horizon_df["Total Contribution (USD)"],
            name="Total Contribution (USD)",
            opacity=0.55,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(
            x=horizon_df["Horizon (Years)"],
            y=horizon_df["Total Portfolio Value (USD)"],
            name="Total Portfolio Value (USD)",
            opacity=0.55,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=horizon_df["Horizon (Years)"],
            y=horizon_df["Total Portfolio Value (USD)"],
            mode="lines+markers",
            name="Final Value Trend (USD)",
            line=dict(width=3),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=horizon_df["Horizon (Years)"],
            y=horizon_df["Total Portfolio Value (KRW)"],
            mode="lines+markers",
            name="Final Value Trend (KRW)",
            line=dict(width=2, dash="dot"),
            visible="legendonly",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=title,
        height=430,
        template="plotly_white",
        barmode="group",
        showlegend=show_legend,
        margin=dict(l=60, r=70, t=25 if title is None else 80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
    )

    fig.update_xaxes(title_text="Horizon (Years)")
    fig.update_yaxes(title_text="USD", secondary_y=False)
    fig.update_yaxes(title_text="KRW", secondary_y=True)

    return fig


# ============================================================
# Sidebar
# ============================================================
st.sidebar.header("Input Parameters")

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

st.sidebar.markdown("---")
st.sidebar.subheader("Allocation")

qqq_weight_pct = st.sidebar.slider(
    "QQQ Allocation (%)",
    min_value=0,
    max_value=100,
    value=50,
    step=1,
)

spy_weight_pct = 100 - qqq_weight_pct
st.sidebar.metric("SPY Allocation (%)", f"{spy_weight_pct}%")

st.sidebar.markdown("---")
st.sidebar.subheader("Expected Return Assumptions")

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
    "Main Horizon",
    options=[10, 20, 30],
    index=0,
)

show_usd = st.sidebar.checkbox("Show USD tables", value=True)
show_krw = st.sidebar.checkbox("Show KRW tables", value=True)
show_full_table = st.sidebar.checkbox("Show full monthly tables", value=False)

# ============================================================
# Build inputs
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

# ============================================================
# Compute
# ============================================================
df = build_simulation_dataframe(inputs)
summary = build_summary_metrics(df)
horizon_df = build_horizon_comparison(inputs, horizons=(10, 20, 30))
plot_tables = prepare_plot_related_tables(df)

for table_df in [horizon_df]:
    numeric_cols = table_df.select_dtypes(include=[np.number]).columns
    table_df[numeric_cols] = table_df[numeric_cols].round(2)

# ============================================================
# Summary metrics
# ============================================================
st.subheader("Summary")

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("Initial SCHD (USD)", format_currency(summary["initial_principal_usd"], "USD"))
col2.metric("Initial SCHD (KRW)", format_currency(summary["initial_principal_krw"], "KRW"))
col3.metric("1Y Net Dividend (USD)", format_currency(summary["first_year_net_dividend_usd"], "USD"))
col4.metric("1Y Net Dividend (KRW)", format_currency(summary["first_year_net_dividend_krw"], "KRW"))
col5.metric(f"{inputs.horizon_years}Y Final Value (USD)", format_currency(summary["final_portfolio_value_usd"], "USD"))
col6.metric(f"{inputs.horizon_years}Y Return", f"{summary['final_total_return_pct']:.2f}%")

col7, col8, col9, col10 = st.columns(4)
col7.metric("Monthly Avg Net Dividend (USD)", format_currency(summary["monthly_avg_net_dividend_usd"], "USD"))
col8.metric("Monthly Avg Net Dividend (KRW)", format_currency(summary["monthly_avg_net_dividend_krw"], "KRW"))
col9.metric(f"{inputs.horizon_years}Y Cum. Net Dividend (USD)", format_currency(summary["final_cumulative_net_dividend_usd"], "USD"))
col10.metric(f"{inputs.horizon_years}Y Final Gain (USD)", format_currency(summary["final_total_gain_usd"], "USD"))

# ============================================================
# Main tables for dedicated table tab
# ============================================================
dividend_table = df[
    [
        "Month",
        "Year",
        "Initial Principal (USD)",
        "SCHD Annual Yield (%)",
        "Gross Dividend (USD)",
        "Tax (USD)",
        "Net Dividend (USD)",
        "Cumulative Net Dividend (USD)",
        "Gross Dividend (KRW)",
        "Tax (KRW)",
        "Net Dividend (KRW)",
        "Cumulative Net Dividend (KRW)",
    ]
].copy()

allocation_table = df[
    [
        "Month",
        "Year",
        "Net Dividend (USD)",
        "Net Dividend (KRW)",
        "QQQ Contribution (USD)",
        "QQQ Contribution (KRW)",
        "SPY Contribution (USD)",
        "SPY Contribution (KRW)",
        "Cumulative QQQ Contribution (USD)",
        "Cumulative QQQ Contribution (KRW)",
        "Cumulative SPY Contribution (USD)",
        "Cumulative SPY Contribution (KRW)",
        "Total Monthly Contribution (USD)",
        "Total Monthly Contribution (KRW)",
        "Total Cumulative Contribution (USD)",
        "Total Cumulative Contribution (KRW)",
    ]
].copy()

compound_table = df[
    [
        "Month",
        "Year",
        "QQQ Contribution (USD)",
        "QQQ Contribution (KRW)",
        "SPY Contribution (USD)",
        "SPY Contribution (KRW)",
        "QQQ Value (USD)",
        "QQQ Value (KRW)",
        "SPY Value (USD)",
        "SPY Value (KRW)",
        "Total Portfolio Value (USD)",
        "Total Portfolio Value (KRW)",
        "Total Contribution (USD)",
        "Total Contribution (KRW)",
        "Total Gain (USD)",
        "Total Gain (KRW)",
        "Total Return (%)",
    ]
].copy()

for table_df in [dividend_table, allocation_table, compound_table]:
    numeric_cols = table_df.select_dtypes(include=[np.number]).columns
    table_df[numeric_cols] = table_df[numeric_cols].round(2)

# ============================================================
# Tabs
# ============================================================
main_tab1, main_tab2, main_tab3 = st.tabs(
    [
        "Dashboard",
        "Charts",
        "Tables / Download",
    ]
)

# ============================================================
# Dashboard tab
# ============================================================
with main_tab1:
    st.markdown("### Quick View")

    dash_col1, dash_col2 = st.columns(2)

    with dash_col1:
        st.markdown("#### Monthly Net Dividend + Cumulative Net Dividend")
        fig1 = make_monthly_bar_line_dual_axis_chart(
            df=df,
            x_col="Month",
            bar_usd_col="Net Dividend (USD)",
            line_usd_col="Cumulative Net Dividend (USD)",
            bar_krw_col="Net Dividend (KRW)",
            line_krw_col="Cumulative Net Dividend (KRW)",
            title=None,
            show_legend=False,
        )
        st.plotly_chart(fig1, use_container_width=True)

        show_related_table(
            "Dividend Table (latest months)",
            plot_tables["dividend"],
            show_full_table=show_full_table,
            default_rows=12,
        )

    with dash_col2:
        st.markdown("#### Monthly Contribution + Cumulative Contribution")
        fig2 = make_monthly_bar_line_dual_axis_chart(
            df=df,
            x_col="Month",
            bar_usd_col="Total Monthly Contribution (USD)",
            line_usd_col="Total Cumulative Contribution (USD)",
            bar_krw_col="Total Monthly Contribution (KRW)",
            line_krw_col="Total Cumulative Contribution (KRW)",
            title=None,
            show_legend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

        show_related_table(
            "Contribution Table (latest months)",
            plot_tables["contribution"],
            show_full_table=show_full_table,
            default_rows=12,
        )

# ============================================================
# Charts tab
# ============================================================
with main_tab2:
    st.markdown("### Detailed Charts")

    chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs(
        [
            "Dividend Flow",
            "Allocation Flow",
            "Compound Growth",
            "Horizon Comparison",
        ]
    )

    with chart_tab1:
        st.markdown("#### SCHD Monthly Net Dividend and Cumulative Net Dividend")
        fig_dividend = make_monthly_bar_line_dual_axis_chart(
            df=df,
            x_col="Month",
            bar_usd_col="Net Dividend (USD)",
            line_usd_col="Cumulative Net Dividend (USD)",
            bar_krw_col="Net Dividend (KRW)",
            line_krw_col="Cumulative Net Dividend (KRW)",
            title=None,
            show_legend=True,
        )
        st.plotly_chart(fig_dividend, use_container_width=True)

        show_related_table(
            "Dividend Table",
            plot_tables["dividend"],
            show_full_table=show_full_table,
            default_rows=24,
        )

    with chart_tab2:
        alloc_col1, alloc_col2 = st.columns(2)

        with alloc_col1:
            st.markdown("#### QQQ Monthly Contribution and Cumulative Contribution")
            fig_alloc_qqq = make_monthly_bar_line_dual_axis_chart(
                df=df,
                x_col="Month",
                bar_usd_col="QQQ Contribution (USD)",
                line_usd_col="Cumulative QQQ Contribution (USD)",
                bar_krw_col="QQQ Contribution (KRW)",
                line_krw_col="Cumulative QQQ Contribution (KRW)",
                title=None,
                show_legend=True,
            )
            st.plotly_chart(fig_alloc_qqq, use_container_width=True)

            show_related_table(
                "QQQ Allocation Table",
                plot_tables["qqq"],
                show_full_table=show_full_table,
                default_rows=24,
            )

        with alloc_col2:
            st.markdown("#### SPY Monthly Contribution and Cumulative Contribution")
            fig_alloc_spy = make_monthly_bar_line_dual_axis_chart(
                df=df,
                x_col="Month",
                bar_usd_col="SPY Contribution (USD)",
                line_usd_col="Cumulative SPY Contribution (USD)",
                bar_krw_col="SPY Contribution (KRW)",
                line_krw_col="Cumulative SPY Contribution (KRW)",
                title=None,
                show_legend=True,
            )
            st.plotly_chart(fig_alloc_spy, use_container_width=True)

            show_related_table(
                "SPY Allocation Table",
                plot_tables["spy"],
                show_full_table=show_full_table,
                default_rows=24,
            )

    with chart_tab3:
        st.markdown(f"#### QQQ / SPY Compounded Growth ({inputs.horizon_years} Years)")
        fig_growth = make_dual_axis_growth_chart(
            df=df,
            title=None,
            show_legend=True,
        )
        st.plotly_chart(fig_growth, use_container_width=True)

        show_related_table(
            "Compound Growth Table",
            plot_tables["growth"],
            show_full_table=show_full_table,
            default_rows=24,
        )

    with chart_tab4:
        st.markdown("#### Horizon Comparison: Contribution and Final Value")
        fig_horizon = make_horizon_bar_line_dual_axis_chart(
            horizon_df=horizon_df,
            title=None,
            show_legend=True,
        )
        st.plotly_chart(fig_horizon, use_container_width=True)

        st.markdown("**Horizon Comparison Table**")
        st.dataframe(horizon_df, use_container_width=True, height=220)

# ============================================================
# Tables / download tab
# ============================================================
with main_tab3:
    table_tab1, table_tab2, table_tab3, table_tab4 = st.tabs(
        [
            "SCHD Dividend Table",
            "Allocation Table",
            "Compound Table",
            "Download CSV",
        ]
    )

    with table_tab1:
        st.markdown("### SCHD Principal and Dividend Table")
        if show_usd and show_krw:
            st.dataframe(
                dividend_table if show_full_table else dividend_table.head(120),
                use_container_width=True,
            )
        elif show_usd:
            usd_cols = [
                "Month",
                "Year",
                "Initial Principal (USD)",
                "SCHD Annual Yield (%)",
                "Gross Dividend (USD)",
                "Tax (USD)",
                "Net Dividend (USD)",
                "Cumulative Net Dividend (USD)",
            ]
            st.dataframe(
                dividend_table[usd_cols] if show_full_table else dividend_table[usd_cols].head(120),
                use_container_width=True,
            )
        elif show_krw:
            krw_cols = [
                "Month",
                "Year",
                "Gross Dividend (KRW)",
                "Tax (KRW)",
                "Net Dividend (KRW)",
                "Cumulative Net Dividend (KRW)",
            ]
            st.dataframe(
                dividend_table[krw_cols] if show_full_table else dividend_table[krw_cols].head(120),
                use_container_width=True,
            )

    with table_tab2:
        st.markdown("### Dividend Allocation to QQQ / SPY")
        st.dataframe(
            allocation_table if show_full_table else allocation_table.head(120),
            use_container_width=True,
        )

    with table_tab3:
        st.markdown("### QQQ / SPY Compound Growth Table")
        st.dataframe(
            compound_table if show_full_table else compound_table.head(120),
            use_container_width=True,
        )

        st.markdown("### 10Y / 20Y / 30Y Comparison")
        st.dataframe(horizon_df, use_container_width=True)

    with table_tab4:
        csv_dividend = dividend_table.to_csv(index=False).encode("utf-8")
        csv_allocation = allocation_table.to_csv(index=False).encode("utf-8")
        csv_compound = compound_table.to_csv(index=False).encode("utf-8")
        csv_horizon = horizon_df.to_csv(index=False).encode("utf-8")

        d1, d2, d3, d4 = st.columns(4)

        with d1:
            st.download_button(
                "Download Dividend Table",
                data=csv_dividend,
                file_name="schd_dividend_table.csv",
                mime="text/csv",
            )

        with d2:
            st.download_button(
                "Download Allocation Table",
                data=csv_allocation,
                file_name="schd_allocation_table.csv",
                mime="text/csv",
            )

        with d3:
            st.download_button(
                "Download Compound Table",
                data=csv_compound,
                file_name="schd_compound_growth_table.csv",
                mime="text/csv",
            )

        with d4:
            st.download_button(
                "Download Horizon Comparison",
                data=csv_horizon,
                file_name="schd_horizon_comparison.csv",
                mime="text/csv",
            )

# ============================================================
# Notes
# ============================================================
with st.expander("Model Notes / Assumptions"):
    st.markdown(
        """
- This version keeps the **SCHD principal constant**.
- SCHD dividend cash flow is estimated from:
  - annual dividend yield
  - tax rate
  - optional dividend growth rate
- QQQ and SPY are modeled with **monthly compounding** from annual CAGR assumptions.
- In **Monthly normalized** mode, dividends are spread evenly over 12 months.
- In **Actual quarterly payout** mode, dividends are paid only in months 3, 6, 9, and 12.
- In Quick View, chart titles are displayed **outside** the plots to avoid overlap.
- Each chart has a related table shown directly below it.
- USD is displayed on the left axis and KRW on the right axis.
- Some KRW traces are set to **legend only** by default to reduce clutter.
        """
    )
