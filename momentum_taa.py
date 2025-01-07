from dotenv import load_dotenv

load_dotenv()
import operator
import os

import numpy as np
import pandas as pd
import plotly.express as px
import pytz
import streamlit as st
from qstrader.alpha_model.alpha_model import AlphaModel
from qstrader.alpha_model.fixed_signals import FixedSignalsAlphaModel
from qstrader.asset.equity import Equity
from qstrader.asset.universe.dynamic import DynamicUniverse
from qstrader.asset.universe.static import StaticUniverse
from qstrader.data.backtest_data_handler import BacktestDataHandler
from qstrader.data.daily_bar_csv import CSVDailyBarDataSource
from qstrader.signals.momentum import MomentumSignal
from qstrader.signals.signals_collection import SignalsCollection
from qstrader.statistics.tearsheet import TearsheetStatistics
from qstrader.trading.backtest import BacktestTradingSession


class TopNMomentumAlphaModel(AlphaModel):

    def __init__(self, signals, mom_lookback, mom_top_n, universe, data_handler):
        """
        Initialise the TopNMomentumAlphaModel

        Parameters
        ----------
        signals : `SignalsCollection`
            The entity for interfacing with various pre-calculated
            signals. In this instance we want to use 'momentum'.
        mom_lookback : `integer`
            The number of business days to calculate momentum
            lookback over.
        mom_top_n : `integer`
            The number of assets to include in the portfolio,
            ranking from highest momentum descending.
        universe : `Universe`
            The collection of assets utilised for signal generation.
        data_handler : `DataHandler`
            The interface to the CSV data.

        Returns
        -------
        None
        """
        self.signals = signals
        self.mom_lookback = mom_lookback
        self.mom_top_n = mom_top_n
        self.universe = universe
        self.data_handler = data_handler

    def _highest_momentum_asset(self, dt):
        """
        Calculates the ordered list of highest performing momentum
        assets restricted to the 'Top N', for a particular datetime.

        Parameters
        ----------
        dt : `pd.Timestamp`
            The datetime for which the highest momentum assets
            should be calculated.

        Returns
        -------
        `list[str]`
            Ordered list of highest performing momentum assets
            restricted to the 'Top N'.
        """
        assets = self.signals["momentum"].assets

        # Calculate the holding-period return momenta for each asset,
        # for the particular provided momentum lookback period
        all_momenta = {
            asset: self.signals["momentum"](asset, self.mom_lookback)
            for asset in assets
        }

        # Obtain a list of the top performing assets by momentum
        # restricted by the provided number of desired assets to
        # trade per month
        return [
            asset[0]
            for asset in sorted(
                all_momenta.items(), key=operator.itemgetter(1), reverse=True
            )
        ][: self.mom_top_n]

    def _generate_signals(self, dt, weights):
        """
        Calculate the highest performing momentum for each
        asset then assign 1 / N of the signal weight to each
        of these assets.

        Parameters
        ----------
        dt : `pd.Timestamp`
            The datetime for which the signal weights
            should be calculated.
        weights : `dict{str: float}`
            The current signal weights dictionary.

        Returns
        -------
        `dict{str: float}`
            The newly created signal weights dictionary.
        """
        top_assets = self._highest_momentum_asset(dt)
        for asset in top_assets:
            weights[asset] = 1.0 / self.mom_top_n
        return weights

    def __call__(self, dt):
        """
        Calculates the signal weights for the top N
        momentum alpha model, assuming that there is
        sufficient data to begin calculating momentum
        on the desired assets.

        Parameters
        ----------
        dt : `pd.Timestamp`
            The datetime for which the signal weights
            should be calculated.

        Returns
        -------
        `dict{str: float}`
            The newly created signal weights dictionary.
        """
        assets = self.universe.get_assets(dt)
        weights = {asset: 0.0 for asset in assets}

        # Only generate weights if the current time exceeds the
        # momentum lookback period
        if self.signals.warmup >= self.mom_lookback:
            weights = self._generate_signals(dt, weights)
        return weights


@st.cache_data(ttl=3600, persist="disk")
def run_strategy(end_dt):
    """
    Runs the momentum strategy and returns the results.
    Results are cached to avoid recomputing on every Streamlit rerun.
    Cache persists to disk and lasts for 1 hour.

    Parameters
    ----------
    end_dt : pd.Timestamp, The end date for the backtest.
    """
    # Duration of the backtest
    start_dt = pd.Timestamp("1998-12-22 14:30:00", tz=pytz.UTC)
    burn_in_dt = pd.Timestamp("1999-12-22 14:30:00", tz=pytz.UTC)

    # Model parameters
    mom_lookback = 126  # Six months worth of business days
    mom_top_n = 3  # Number of assets to include at any one time

    # Construct the symbols and assets necessary for the backtest
    # This utilises the SPDR US sector ETFs, all beginning with XL
    strategy_symbols = ["XL%s" % sector for sector in "BCEFIKPUVY"]
    assets = ["EQ:%s" % symbol for symbol in strategy_symbols]

    # As this is a dynamic universe of assets (XLC is added later)
    # we need to tell QSTrader when XLC can be included. This is
    # achieved using an asset dates dictionary
    asset_dates = {asset: start_dt for asset in assets}
    asset_dates["EQ:XLC"] = pd.Timestamp("2018-06-18 00:00:00", tz=pytz.UTC)
    strategy_universe = DynamicUniverse(asset_dates)

    # To avoid loading all CSV files in the directory, set the
    # data source to load only those provided symbols
    csv_dir = os.environ.get("QSTRADER_CSV_DATA_DIR", ".")
    strategy_data_source = CSVDailyBarDataSource(
        csv_dir, Equity, csv_symbols=strategy_symbols
    )
    strategy_data_handler = BacktestDataHandler(
        strategy_universe, data_sources=[strategy_data_source]
    )

    # Generate the signals (in this case holding-period return based
    # momentum) used in the top-N momentum alpha model
    momentum = MomentumSignal(start_dt, strategy_universe, lookbacks=[mom_lookback])
    signals = SignalsCollection({"momentum": momentum}, strategy_data_handler)

    # Generate the alpha model instance for the top-N momentum alpha model
    strategy_alpha_model = TopNMomentumAlphaModel(
        signals, mom_lookback, mom_top_n, strategy_universe, strategy_data_handler
    )

    # Construct the strategy backtest and run it
    strategy_backtest = BacktestTradingSession(
        start_dt,
        end_dt,
        strategy_universe,
        strategy_alpha_model,
        signals=signals,
        rebalance="end_of_month",
        long_only=True,
        cash_buffer_percentage=0.01,
        burn_in_dt=burn_in_dt,
        data_handler=strategy_data_handler,
    )
    strategy_backtest.run()

    # Construct benchmark assets (buy & hold SPY)
    benchmark_symbols = ["SPY"]
    benchmark_assets = ["EQ:SPY"]
    benchmark_universe = StaticUniverse(benchmark_assets)
    benchmark_data_source = CSVDailyBarDataSource(
        csv_dir, Equity, csv_symbols=benchmark_symbols
    )
    benchmark_data_handler = BacktestDataHandler(
        benchmark_universe, data_sources=[benchmark_data_source]
    )

    # Construct a benchmark Alpha Model that provides
    # 100% static allocation to the SPY ETF, with no rebalance
    benchmark_alpha_model = FixedSignalsAlphaModel({"EQ:SPY": 1.0})
    benchmark_backtest = BacktestTradingSession(
        burn_in_dt,
        end_dt,
        benchmark_universe,
        benchmark_alpha_model,
        rebalance="buy_and_hold",
        long_only=True,
        cash_buffer_percentage=0.01,
        data_handler=benchmark_data_handler,
    )
    benchmark_backtest.run()

    # Get the equity curves and inspect their structure
    strategy_curve = strategy_backtest.get_equity_curve()
    benchmark_curve = benchmark_backtest.get_equity_curve()

    # Performance Output
    tearsheet = TearsheetStatistics(
        strategy_equity=strategy_backtest.get_equity_curve(),
        benchmark_equity=benchmark_backtest.get_equity_curve(),
        title="US Sector Momentum - Top 3 Sectors",
    )
    tearsheet.plot_results()

    # Instead of assuming column names, let's return the entire DataFrames
    return {
        "strategy_equity": strategy_curve,
        "benchmark_equity": benchmark_curve,
    }


if __name__ == "__main__":
    st.set_page_config(page_title="US Sector Momentum Strategy", layout="wide")

    st.title("US Sector Momentum - Top 3 Sectors")

    # Add a button to clear the cache
    if st.button("Clear Cache and Rerun Strategy"):
        st.cache_data.clear()
        st.experimental_rerun()

    # Convert date to timestamp with time and timezone
    end_dt = pd.Timestamp("2024-12-31 23:59:00", tz=pytz.UTC)

    with st.spinner(
        "Running backtest strategy... (this may take a few minutes on first run)"
    ):
        results = run_strategy(end_dt)

    # Get the equity curves
    strategy_equity = results["strategy_equity"]["Equity"]  # Get the Equity series
    benchmark_equity = results["benchmark_equity"]["Equity"]  # Get the Equity series

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

    # with col2:
    #     st.subheader("Portfolio Value")
    #     # Create value data
    #     value_data = pd.DataFrame(
    #         {
    #             "Strategy": strategy_equity / 1_000_000,
    #             "Benchmark": benchmark_equity / 1_000_000,
    #         }
    #     )
    #     fig_value = px.line(
    #         value_data,
    #         title="Portfolio Value (Multiple of Initial Investment)",
    #     )
    #     fig_value.update_layout(yaxis_title="Value (Multiple of Initial)")
    #     st.plotly_chart(fig_value)

    # Calculate returns with datetime index
    strategy_returns = strategy_equity.pct_change().dropna()
    benchmark_returns = benchmark_equity.pct_change().dropna()

    # Calculate drawdowns (fix the percentage calculation)
    strategy_drawdown = (
        strategy_equity / strategy_equity.cummax() - 1
    )  # Remove the *100 here
    benchmark_drawdown = (
        benchmark_equity / benchmark_equity.cummax() - 1
    )  # Remove the *100 here

    # Create drawdown chart
    drawdown_data = pd.DataFrame(
        {
            "Strategy": strategy_drawdown
            * 100,  # Convert to percentage here for display
            "Benchmark": benchmark_drawdown
            * 100,  # Convert to percentage here for display
        }
    )

    fig_drawdown = px.line(
        drawdown_data,
        title="Drawdown (%)",
    )
    fig_drawdown.update_layout(yaxis_title="Drawdown (%)", showlegend=True, height=300)
    st.plotly_chart(fig_drawdown)
    # Ensure we have a datetime index
    if not isinstance(strategy_returns.index, pd.DatetimeIndex):
        strategy_returns.index = pd.to_datetime(strategy_returns.index)
    if not isinstance(benchmark_returns.index, pd.DatetimeIndex):
        benchmark_returns.index = pd.to_datetime(benchmark_returns.index)

    # Calculate metrics
    total_days = len(strategy_returns)
    ann_factor = np.sqrt(252)  # Annualization factor for daily data

    # Calculate CAGR
    total_return_strategy = (strategy_equity.iloc[-1] / strategy_equity.iloc[0]) - 1
    total_return_benchmark = (benchmark_equity.iloc[-1] / benchmark_equity.iloc[0]) - 1
    years = total_days / 252
    cagr_strategy = (1 + total_return_strategy) ** (1 / years) - 1
    cagr_benchmark = (1 + total_return_benchmark) ** (1 / years) - 1

    # Calculate Sortino Ratio (using 0% as minimum acceptable return)
    downside_returns_strat = strategy_returns[strategy_returns < 0]
    downside_returns_bench = benchmark_returns[benchmark_returns < 0]
    downside_std_strat = np.sqrt(252) * np.sqrt(np.mean(downside_returns_strat**2))
    downside_std_bench = np.sqrt(252) * np.sqrt(np.mean(downside_returns_bench**2))
    sortino_ratio_strat = (strategy_returns.mean() * 252) / downside_std_strat
    sortino_ratio_bench = (benchmark_returns.mean() * 252) / downside_std_bench

    # Calculate drawdown durations
    def get_drawdown_duration(equity_series):
        """
        Calculate the maximum drawdown duration in days.

        Parameters
        ----------
        equity_series : pd.Series
            The equity curve series

        Returns
        -------
        int
            The maximum drawdown duration in days
        """
        # Calculate drawdown series
        drawdown = equity_series / equity_series.cummax() - 1

        # Find drawdown periods
        is_drawdown = drawdown < 0

        # If no drawdown, return 0
        if not is_drawdown.any():
            return 0

        # Find start of each drawdown period
        drawdown_starts = (is_drawdown != is_drawdown.shift()).cumsum()[is_drawdown]

        # Group by drawdown period and count days
        drawdown_durations = drawdown_starts.groupby(drawdown_starts).size()

        # Return the maximum duration
        return int(drawdown_durations.max())

    metrics = pd.DataFrame(
        {
            "Strategy": [
                total_return_strategy,  # Total Return
                cagr_strategy,  # CAGR
                strategy_returns.mean()
                * 252
                / (strategy_returns.std() * ann_factor),  # Sharpe
                sortino_ratio_strat,  # Sortino
                strategy_returns.std() * ann_factor,  # Annual Volatility
                strategy_drawdown.min(),  # Max Drawdown (now as decimal)
                get_drawdown_duration(strategy_equity),  # Max Drawdown Duration
            ],
            "Benchmark": [
                total_return_benchmark,
                cagr_benchmark,
                benchmark_returns.mean() * 252 / (benchmark_returns.std() * ann_factor),
                sortino_ratio_bench,
                benchmark_returns.std() * ann_factor,
                benchmark_drawdown.min(),  # Max Drawdown (now as decimal)
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

    # Create a row for the returns charts
    st.subheader("Returns Analysis")
    col3, col4 = st.columns(2)

    with col3:
        # Monthly Returns Heatmap
        st.subheader("Monthly Returns (%)")

        # Calculate monthly returns with proper datetime index
        monthly_returns = (
            strategy_returns.resample("M").apply(lambda x: (1 + x).prod() - 1) * 100
        )

        # Ensure we have enough complete years for the matrix
        n_years = len(monthly_returns) // 12
        monthly_values = monthly_returns.values[
            : n_years * 12
        ]  # Only use complete years

        # Create monthly returns matrix
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

        # Create heatmap
        fig_monthly = px.imshow(
            monthly_matrix,
            color_continuous_scale="RdYlGn",  # Red to Yellow to Green scale
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

    with col4:
        # Yearly Returns Bar Chart
        st.subheader("Yearly Returns (%)")

        # Calculate yearly returns
        yearly_returns = (
            strategy_returns.resample("Y").apply(lambda x: (1 + x).prod() - 1) * 100
        )
        yearly_returns.index = yearly_returns.index.year

        # Create bar chart
        fig_yearly = px.bar(
            x=yearly_returns.index,
            y=yearly_returns.values,
            title="Yearly Returns (%)",
            labels={"x": "Year", "y": "Return (%)"},
        )
        fig_yearly.update_layout(showlegend=False, height=400)
        # Add horizontal line at y=0
        fig_yearly.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        st.plotly_chart(fig_yearly)

    # Display performance metrics
    st.subheader("Performance Metrics")

    # Format the metrics table
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

    # Format ratios to 2 decimal places
    formatted_metrics.loc["Sharpe Ratio"] = metrics.loc["Sharpe Ratio"].map(
        "{:.2f}".format
    )
    formatted_metrics.loc["Sortino Ratio"] = metrics.loc["Sortino Ratio"].map(
        "{:.2f}".format
    )

    # Format duration as integer
    formatted_metrics.loc["Max Drawdown Duration (Days)"] = metrics.loc[
        "Max Drawdown Duration (Days)"
    ].map("{:.0f}".format)

    st.dataframe(formatted_metrics)
