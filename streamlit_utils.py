import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def plot_strategy_results(strategy_equity, benchmark_equity):
    """
    Plot strategy results using Streamlit.

    Parameters
    ----------
    strategy_equity : pd.DataFrame or pd.Series
        Strategy equity curve DataFrame or Series
    benchmark_equity : pd.DataFrame or pd.Series
        Benchmark equity curve DataFrame or Series
    """
    # If DataFrames are passed, get the 'Equity' column
    if isinstance(strategy_equity, pd.DataFrame):
        strategy_equity = strategy_equity["Equity"]
    if isinstance(benchmark_equity, pd.DataFrame):
        benchmark_equity = benchmark_equity["Equity"]

    # Ensure we have datetime indices
    if not isinstance(strategy_equity.index, pd.DatetimeIndex):
        strategy_equity.index = pd.to_datetime(strategy_equity.index)
    if not isinstance(benchmark_equity.index, pd.DatetimeIndex):
        benchmark_equity.index = pd.to_datetime(benchmark_equity.index)

    # Create returns data normalized to 100
    returns_data = pd.DataFrame(
        {
            "Strategy": (strategy_equity / strategy_equity.iloc[0] - 1) * 100,
            "Benchmark": (benchmark_equity / benchmark_equity.iloc[0] - 1) * 100,
        }
    )
    fig_returns = px.line(
        returns_data,
        title="Cumulative Returns (%)",
    )
    fig_returns.update_layout(yaxis_title="Return (%)")
    st.plotly_chart(fig_returns)

    # Calculate returns
    strategy_returns = strategy_equity.pct_change().dropna()
    benchmark_returns = benchmark_equity.pct_change().dropna()

    # Calculate drawdowns
    strategy_drawdown = strategy_equity / strategy_equity.cummax() - 1
    benchmark_drawdown = benchmark_equity / benchmark_equity.cummax() - 1

    # Create drawdown chart
    drawdown_data = pd.DataFrame(
        {
            "Strategy": strategy_drawdown * 100,
            "Benchmark": benchmark_drawdown * 100,
        }
    )

    fig_drawdown = px.line(
        drawdown_data,
        title="Drawdown (%)",
    )
    fig_drawdown.update_layout(yaxis_title="Drawdown (%)", showlegend=True, height=300)
    st.plotly_chart(fig_drawdown)

    # Display returns analysis
    plot_returns_analysis(strategy_returns)

    # Display metrics
    display_metrics(
        strategy_equity, benchmark_equity, strategy_returns, benchmark_returns
    )


def display_metrics(
    strategy_equity, benchmark_equity, strategy_returns, benchmark_returns
):
    """Display performance metrics table."""
    st.subheader("Performance Metrics")

    metrics = calculate_metrics(
        strategy_equity, benchmark_equity, strategy_returns, benchmark_returns
    )
    formatted_metrics = format_metrics(metrics)
    st.dataframe(formatted_metrics)


def calculate_metrics(
    strategy_equity, benchmark_equity, strategy_returns, benchmark_returns
):
    """Calculate performance metrics."""
    total_days = len(strategy_returns)
    ann_factor = np.sqrt(252)

    # Calculate CAGR
    total_return_strategy = (strategy_equity.iloc[-1] / strategy_equity.iloc[0]) - 1
    total_return_benchmark = (benchmark_equity.iloc[-1] / benchmark_equity.iloc[0]) - 1
    years = total_days / 252
    cagr_strategy = (1 + total_return_strategy) ** (1 / years) - 1
    cagr_benchmark = (1 + total_return_benchmark) ** (1 / years) - 1

    # Calculate Sortino Ratio
    downside_returns_strat = strategy_returns[strategy_returns < 0]
    downside_returns_bench = benchmark_returns[benchmark_returns < 0]
    downside_std_strat = np.sqrt(252) * np.sqrt(np.mean(downside_returns_strat**2))
    downside_std_bench = np.sqrt(252) * np.sqrt(np.mean(downside_returns_bench**2))
    sortino_ratio_strat = (strategy_returns.mean() * 252) / downside_std_strat
    sortino_ratio_bench = (benchmark_returns.mean() * 252) / downside_std_bench

    # Calculate drawdowns
    strategy_drawdown = strategy_equity / strategy_equity.cummax() - 1
    benchmark_drawdown = benchmark_equity / benchmark_equity.cummax() - 1

    return pd.DataFrame(
        {
            "Strategy": [
                total_return_strategy,
                cagr_strategy,
                strategy_returns.mean() * 252 / (strategy_returns.std() * ann_factor),
                sortino_ratio_strat,
                strategy_returns.std() * ann_factor,
                strategy_drawdown.min(),
                get_drawdown_duration(strategy_equity),
            ],
            "Benchmark": [
                total_return_benchmark,
                cagr_benchmark,
                benchmark_returns.mean() * 252 / (benchmark_returns.std() * ann_factor),
                sortino_ratio_bench,
                benchmark_returns.std() * ann_factor,
                benchmark_drawdown.min(),
                get_drawdown_duration(benchmark_equity),
            ],
        },
        index=[
            "Total Return",
            "CAGR",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Annual Volatility",
            "Max Daily Drawdown",
            "Max Drawdown Duration (Days)",
        ],
    )


def format_metrics(metrics):
    """Format metrics for display."""
    formatted_metrics = metrics.copy()

    # Format percentages
    for col in formatted_metrics.columns:
        formatted_metrics.loc["Total Return", col] = (
            f"{metrics.loc['Total Return', col]:.2%}"
        )
        formatted_metrics.loc["CAGR", col] = f"{metrics.loc['CAGR', col]:.2%}"
        formatted_metrics.loc["Annual Volatility", col] = (
            f"{metrics.loc['Annual Volatility', col]:.2%}"
        )
        formatted_metrics.loc["Max Daily Drawdown", col] = (
            f"{metrics.loc['Max Daily Drawdown', col]:.2%}"
        )

    # Format ratios
    formatted_metrics.loc["Sharpe Ratio"] = metrics.loc["Sharpe Ratio"].map(
        "{:.2f}".format
    )
    formatted_metrics.loc["Sortino Ratio"] = metrics.loc["Sortino Ratio"].map(
        "{:.2f}".format
    )

    # Format duration
    formatted_metrics.loc["Max Drawdown Duration (Days)"] = metrics.loc[
        "Max Drawdown Duration (Days)"
    ].map("{:.0f}".format)

    return formatted_metrics


def plot_returns_analysis(returns):
    """Plot monthly and yearly returns analysis."""
    # Ensure we have a datetime index
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns.index = pd.to_datetime(returns.index)

    st.subheader("Returns Analysis")
    col1, col2 = st.columns(2)

    with col1:
        plot_monthly_returns(returns)

    with col2:
        plot_yearly_returns(returns)


def plot_monthly_returns(returns):
    """Plot monthly returns heatmap."""
    st.subheader("Monthly Returns (%)")

    monthly_returns = returns.resample("M").apply(lambda x: (1 + x).prod() - 1) * 100
    n_years = len(monthly_returns) // 12
    monthly_values = monthly_returns.values[: n_years * 12]

    monthly_matrix = pd.DataFrame(
        monthly_values.reshape(n_years, 12),
        index=monthly_returns.index.year.unique()[:n_years],
        columns=[
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
    )

    fig_monthly = px.imshow(
        monthly_matrix,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        title="Monthly Returns Heatmap (%)",
    )
    fig_monthly.update_traces(text=monthly_matrix.round(1), texttemplate="%{text}")
    fig_monthly.update_layout(
        coloraxis_colorbar_title="Return (%)",
        xaxis_title="",
        yaxis_title="Year",
        height=400,
    )
    st.plotly_chart(fig_monthly)


def plot_yearly_returns(returns):
    """Plot yearly returns bar chart."""
    st.subheader("Yearly Returns (%)")

    yearly_returns = returns.resample("Y").apply(lambda x: (1 + x).prod() - 1) * 100
    yearly_returns.index = yearly_returns.index.year

    fig_yearly = px.bar(
        x=yearly_returns.index,
        y=yearly_returns.values,
        title="Yearly Returns (%)",
        labels={"x": "Year", "y": "Return (%)"},
    )
    fig_yearly.update_layout(showlegend=False, height=400)
    fig_yearly.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    st.plotly_chart(fig_yearly)


def get_drawdown_duration(equity_series):
    """Calculate maximum drawdown duration in days."""
    drawdown = equity_series / equity_series.cummax() - 1
    is_drawdown = drawdown < 0

    if not is_drawdown.any():
        return 0

    drawdown_starts = (is_drawdown != is_drawdown.shift()).cumsum()[is_drawdown]
    drawdown_durations = drawdown_starts.groupby(drawdown_starts).size()
    return int(drawdown_durations.max())
