# streamlit_app.py
# ============================================================
# SCHD Dividend -> QQQ/SPY DCA Simulator
# English-based full code
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
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
    "Simulate SCHD dividend cash flow, cumulative dividend income, "
    "and long-term QQQ/SPY compounding with adjustable allocation."
)

# ============================================================
# Helper dataclass
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
    """
    Convert annual CAGR into monthly compounded rate.
    Example: 10% annual -> monthly rate = (1.10)^(1/12) - 1
    """
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
        # Growing dividend assumption:
        # SCHD principal is kept constant in this version.
        # Dividend cash flow can grow annually by an input growth rate.
        effective_annual_yield = inputs.schd_yield_annual * ((1 + schd_div_growth_monthly) ** (month - 1))

        if inputs.payout_mode == "Monthly normalized":
            gross_dividend_usd = initial_usd * effective_annual_yield / 12.0
        else:
            # Actual quarterly payout approximation:
            # Pay only on months 3, 6, 9, 12...
            if month % 3 == 0:
                gross_dividend_usd = initial_usd * effective_annual_yield / 4.0
            else:
                gross_dividend_usd = 0.0

        tax_usd = gross_dividend_usd * inputs.tax_rate
        net_dividend_usd = gross_dividend_usd - tax_usd

        qqq_contribution = net_dividend_usd * inputs.qqq_weight
        spy_contribution = net_dividend_usd * inputs.spy_weight

        # Grow previous portfolio balances first, then add monthly contribution
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
                "SPY Contribution (USD)": spy_contribution,
                "QQQ Contribution (KRW)": usd_to_krw(qqq_contribution, inputs.fx_rate),
                "SPY Contribution (KRW)": usd_to_krw(spy_contribution, inputs.fx_rate),
                "Cumulative QQQ Contribution (USD)": cumulative_qqq_contribution,
                "Cumulative SPY Contribution (USD)": cumulative_spy_contribution,
                "QQQ Value (USD)": qqq_value,
                "SPY Value (USD)": spy_value,
                "QQQ Value (KRW)": usd_to_krw(qqq_value, inputs.fx_rate),
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

    return pd.DataFrame(rows)


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
        df = build_simulation_dataframe(temp_inputs)
        last_row = df.iloc[-1]

        rows.append(
            {
                "Horizon (Years)": horizon,
                "Cumulative Net Dividend (USD)": last_row["Cumulative Net Dividend (USD)"],
                "Cumulative Net Dividend (KRW)": last_row["Cumulative Net Dividend (KRW)"],
                "Total Contribution (USD)": last_row["Total Contribution (USD)"],
                "Total Contribution (KRW)": last_row["Total Contribution (KRW)"],
                "QQQ Value (USD)": last_row["QQQ Value (USD)"],
                "SPY Value (USD)": last_row["SPY Value (USD)"],
                "Total Portfolio Value (USD)": last_row["Total Portfolio Value (USD)"],
                "Total Portfolio Value (KRW)": last_row["Total Portfolio Value (KRW)"],
                "Total Gain (USD)": last_row["Total Gain (USD)"],
                "Total Gain (KRW)": last_row["Total Gain (KRW)"],
                "Total Return (%)": last_row["Total Return (%)"],
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Chart functions
# ============================================================
def make_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    title: str,
    y_axis_title: str,
) -> go.Figure:
    fig = go.Figure()
    for col in y_cols:
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[col],
                mode="lines",
                name=col,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_axis_title,
        height=420,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def make_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    title: str,
    y_axis_title: str,
) -> go.Figure:
    fig = go.Figure()
    for col in y_cols:
        fig.add_trace(
            go.Bar(
                x=df[x_col],
                y=df[col],
                name=col,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_axis_title,
        barmode="group",
        height=420,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ============================================================
# Sidebar inputs
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
show_full_table = st.sidebar.checkbox("Show Full Monthly Tables", value=False)

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

# ============================================================
# Summary
# ============================================================
st.subheader("Summary")

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric(
    "Initial SCHD (USD)",
    format_currency(summary["initial_principal_usd"], "USD"),
)
col2.metric(
    "Initial SCHD (KRW)",
    format_currency(summary["initial_principal_krw"], "KRW"),
)
col3.metric(
    "1Y Net Dividend (USD)",
    format_currency(summary["first_year_net_dividend_usd"], "USD"),
)
col4.metric(
    "1Y Net Dividend (KRW)",
    format_currency(summary["first_year_net_dividend_krw"], "KRW"),
)
col5.metric(
    f"{inputs.horizon_years}Y Final Value (USD)",
    format_currency(summary["final_portfolio_value_usd"], "USD"),
)
col6.metric(
    f"{inputs.horizon_years}Y Return",
    f"{summary['final_total_return_pct']:.2f}%",
)

col7, col8, col9, col10 = st.columns(4)
col7.metric(
    "Monthly Avg Net Dividend (USD)",
    format_currency(summary["monthly_avg_net_dividend_usd"], "USD"),
)
col8.metric(
    "Monthly Avg Net Dividend (KRW)",
    format_currency(summary["monthly_avg_net_dividend_krw"], "KRW"),
)
col9.metric(
    f"{inputs.horizon_years}Y Cum. Net Dividend (USD)",
    format_currency(summary["final_cumulative_net_dividend_usd"], "USD"),
)
col10.metric(
    f"{inputs.horizon_years}Y Final Gain (USD)",
    format_currency(summary["final_total_gain_usd"], "USD"),
)

# ============================================================
# Charts
# ============================================================
st.subheader("Charts")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig_dividend = make_line_chart(
        df,
        x_col="Month",
        y_cols=["Net Dividend (USD)", "Cumulative Net Dividend (USD)"],
        title="SCHD Net Dividend and Cumulative Net Dividend (USD)",
        y_axis_title="USD",
    )
    st.plotly_chart(fig_dividend, use_container_width=True)

with chart_col2:
    fig_contrib = make_line_chart(
        df,
        x_col="Month",
        y_cols=["Cumulative QQQ Contribution (USD)", "Cumulative SPY Contribution (USD)"],
        title="Cumulative QQQ / SPY Contributions (USD)",
        y_axis_title="USD",
    )
    st.plotly_chart(fig_contrib, use_container_width=True)

chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    fig_growth = make_line_chart(
        df,
        x_col="Month",
        y_cols=["QQQ Value (USD)", "SPY Value (USD)", "Total Portfolio Value (USD)"],
        title=f"Compounded Portfolio Growth ({inputs.horizon_years} Years, USD)",
        y_axis_title="USD",
    )
    st.plotly_chart(fig_growth, use_container_width=True)

with chart_col4:
    fig_horizon = make_bar_chart(
        horizon_df,
        x_col="Horizon (Years)",
        y_cols=["Total Contribution (USD)", "Total Portfolio Value (USD)"],
        title="Horizon Comparison: Contribution vs Final Value (USD)",
        y_axis_title="USD",
    )
    st.plotly_chart(fig_horizon, use_container_width=True)

# ============================================================
# Horizon comparison table
# ============================================================
st.subheader("10Y / 20Y / 30Y Comparison")

display_horizon_df = horizon_df.copy()
numeric_cols = display_horizon_df.select_dtypes(include=[np.number]).columns
display_horizon_df[numeric_cols] = display_horizon_df[numeric_cols].round(2)
st.dataframe(display_horizon_df, use_container_width=True)

# ============================================================
# Tables
# ============================================================
st.subheader("Tables")

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
        "QQQ Contribution (USD)",
        "SPY Contribution (USD)",
        "Cumulative QQQ Contribution (USD)",
        "Cumulative SPY Contribution (USD)",
        "Net Dividend (KRW)",
        "QQQ Contribution (KRW)",
        "SPY Contribution (KRW)",
    ]
].copy()

compound_table = df[
    [
        "Month",
        "Year",
        "QQQ Contribution (USD)",
        "SPY Contribution (USD)",
        "QQQ Value (USD)",
        "SPY Value (USD)",
        "Total Portfolio Value (USD)",
        "Total Contribution (USD)",
        "Total Gain (USD)",
        "Total Return (%)",
        "QQQ Value (KRW)",
        "SPY Value (KRW)",
        "Total Portfolio Value (KRW)",
        "Total Contribution (KRW)",
        "Total Gain (KRW)",
    ]
].copy()

for table_df in [dividend_table, allocation_table, compound_table]:
    num_cols = table_df.select_dtypes(include=[np.number]).columns
    table_df[num_cols] = table_df[num_cols].round(2)

tab1, tab2, tab3 = st.tabs(
    [
        "1) SCHD Dividend Table",
        "2) Allocation Table",
        "3) Compound Growth Table",
    ]
)

with tab1:
    st.markdown("### SCHD Principal and Dividend Table")
    if show_usd and show_krw:
        st.dataframe(dividend_table if show_full_table else dividend_table.head(120), use_container_width=True)
    elif show_usd:
        st.dataframe(
            dividend_table[
                [
                    "Month",
                    "Year",
                    "Initial Principal (USD)",
                    "SCHD Annual Yield (%)",
                    "Gross Dividend (USD)",
                    "Tax (USD)",
                    "Net Dividend (USD)",
                    "Cumulative Net Dividend (USD)",
                ]
            ] if show_full_table else dividend_table[
                [
                    "Month",
                    "Year",
                    "Initial Principal (USD)",
                    "SCHD Annual Yield (%)",
                    "Gross Dividend (USD)",
                    "Tax (USD)",
                    "Net Dividend (USD)",
                    "Cumulative Net Dividend (USD)",
                ]
            ].head(120),
            use_container_width=True,
        )
    elif show_krw:
        st.dataframe(
            dividend_table[
                [
                    "Month",
                    "Year",
                    "Gross Dividend (KRW)",
                    "Tax (KRW)",
                    "Net Dividend (KRW)",
                    "Cumulative Net Dividend (KRW)",
                ]
            ] if show_full_table else dividend_table[
                [
                    "Month",
                    "Year",
                    "Gross Dividend (KRW)",
                    "Tax (KRW)",
                    "Net Dividend (KRW)",
                    "Cumulative Net Dividend (KRW)",
                ]
            ].head(120),
            use_container_width=True,
        )

with tab2:
    st.markdown("### Dividend Allocation to QQQ / SPY")
    st.dataframe(allocation_table if show_full_table else allocation_table.head(120), use_container_width=True)

with tab3:
    st.markdown("### QQQ / SPY Compound Growth Table")
    st.dataframe(compound_table if show_full_table else compound_table.head(120), use_container_width=True)

# ============================================================
# Downloads
# ============================================================
st.subheader("Download CSV")

csv_dividend = dividend_table.to_csv(index=False).encode("utf-8")
csv_allocation = allocation_table.to_csv(index=False).encode("utf-8")
csv_compound = compound_table.to_csv(index=False).encode("utf-8")
csv_horizon = display_horizon_df.to_csv(index=False).encode("utf-8")

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
- SCHD dividend cash flow is estimated using:
  - annual dividend yield
  - tax rate
  - optional dividend growth rate
- QQQ/SPY growth is modeled using **monthly compounding** from annual CAGR inputs.
- In **Monthly normalized** mode, dividend is spread evenly across 12 months.
- In **Actual quarterly payout** mode, dividend is paid only in months 3, 6, 9, 12.
- This is a planning / scenario tool, not a real-time market pricing engine.
        """
    )
