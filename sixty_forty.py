####
# 60/40 US Equities/Bonds
# https://www.quantstart.com/qstrader/tutorial-60-40-portfolio/
####

from dotenv import load_dotenv

load_dotenv()
import os

import pandas as pd
import pytz
import streamlit as st
from qstrader.alpha_model.fixed_signals import FixedSignalsAlphaModel
from qstrader.asset.equity import Equity
from qstrader.asset.universe.static import StaticUniverse
from qstrader.data.backtest_data_handler import BacktestDataHandler
from qstrader.data.daily_bar_csv import CSVDailyBarDataSource
from qstrader.statistics.tearsheet import TearsheetStatistics
from qstrader.trading.backtest import BacktestTradingSession

from streamlit_utils import plot_strategy_results


@st.cache_data(ttl=3600, persist="disk")
def run_strategy(end_dt):
    # Also please note that at this stage of QSTrader development, for the 'buy & hold'
    # methodology used below it is necessary to specify a starting time of 14:30:00 UTC
    # in order for the backtest to proceed correctly.
    start_dt = pd.Timestamp("2003-09-30 14:30:00", tz=pytz.UTC)

    # Construct the symbols and assets necessary for the backtest
    strategy_symbols = ["SPY", "AGG"]
    strategy_assets = ["EQ:%s" % symbol for symbol in strategy_symbols]
    strategy_universe = StaticUniverse(strategy_assets)

    # To avoid loading all CSV files in the directory, set the
    # data source to load only those provided symbols
    csv_dir = os.environ.get("QSTRADER_CSV_DATA_DIR", ".")
    data_source = CSVDailyBarDataSource(csv_dir, Equity, csv_symbols=strategy_symbols)
    data_handler = BacktestDataHandler(strategy_universe, data_sources=[data_source])

    # Construct an Alpha Model that simply provides
    # static allocations to a universe of assets
    # In this case 60% SPY ETF, 40% AGG ETF,
    # rebalanced at the end of each month
    strategy_alpha_model = FixedSignalsAlphaModel({"EQ:SPY": 0.6, "EQ:AGG": 0.4})
    strategy_backtest = BacktestTradingSession(
        start_dt,
        end_dt,
        strategy_universe,
        strategy_alpha_model,
        rebalance="end_of_month",
        long_only=True,
        cash_buffer_percentage=0.01,
        data_handler=data_handler,
    )
    strategy_backtest.run()

    # Construct benchmark assets (buy & hold SPY)
    benchmark_assets = ["EQ:SPY"]
    benchmark_universe = StaticUniverse(benchmark_assets)

    # Construct a benchmark Alpha Model that provides
    # 100% static allocation to the SPY ETF, with no rebalance
    benchmark_alpha_model = FixedSignalsAlphaModel({"EQ:SPY": 1.0})
    benchmark_backtest = BacktestTradingSession(
        start_dt,
        end_dt,
        benchmark_universe,
        benchmark_alpha_model,
        rebalance="buy_and_hold",
        long_only=True,
        cash_buffer_percentage=0.01,
        data_handler=data_handler,
    )
    benchmark_backtest.run()

    # Performance Output
    tearsheet = TearsheetStatistics(
        strategy_equity=strategy_backtest.get_equity_curve(),
        benchmark_equity=benchmark_backtest.get_equity_curve(),
        title="60/40 US Equities/Bonds",
    )
    tearsheet.plot_results()

    return {
        "strategy_equity": strategy_backtest.get_equity_curve(),
        "benchmark_equity": benchmark_backtest.get_equity_curve(),
    }


if __name__ == "__main__":
    st.set_page_config(page_title="60/40 US Equities/Bonds", layout="wide")

    st.title("60/40 US Equities/Bonds")

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

    plot_strategy_results(strategy_equity, benchmark_equity)
