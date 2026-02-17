#!/usr/bin/env python3
"""
Benchmark Calculator for Paper Trading

Calculates performance metrics against benchmarks:
- Equal-weight portfolio benchmark
- Bitcoin-only benchmark
- Alpha (excess return)
- Sharpe ratios
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple


def calculate_sharpe(returns: np.ndarray, periods_per_year: int = 365 * 24) -> float:
    """
    Calculate Sharpe ratio from returns.

    Args:
        returns: Array of returns
        periods_per_year: Number of periods per year (default: hourly = 365*24)

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0

    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)

    if std_return == 0:
        return 0.0

    sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
    return sharpe


def calculate_returns(prices: np.ndarray) -> np.ndarray:
    """Calculate returns from price series."""
    if len(prices) < 2:
        return np.array([])
    return np.diff(prices) / prices[:-1]


def get_benchmark_data(csv_path: Path, tickers: list) -> Dict:
    """
    Calculate benchmark metrics from CSV data.

    Args:
        csv_path: Path to paper trading CSV with OHLCV data
        tickers: List of tickers in the portfolio

    Returns:
        Dictionary with benchmark metrics
    """
    if not csv_path.exists():
        return None

    try:
        # Read CSV (assuming format: timestamp, ticker, open, high, low, close, volume)
        df = pd.read_csv(csv_path)

        # Get close prices for each ticker
        btc_prices = df[df['ticker'] == 'BTC/USD']['close'].values
        all_tickers_prices = {}

        for ticker in tickers:
            ticker_data = df[df['ticker'] == ticker]['close'].values
            if len(ticker_data) > 0:
                all_tickers_prices[ticker] = ticker_data

        if len(btc_prices) == 0:
            return None

        # Calculate Bitcoin-only benchmark
        btc_returns = calculate_returns(btc_prices)
        btc_sharpe = calculate_sharpe(btc_returns)
        btc_total_return = (btc_prices[-1] / btc_prices[0] - 1) * 100

        # Calculate equal-weight benchmark
        # For each timestamp, average the returns across all tickers
        min_length = min(len(prices) for prices in all_tickers_prices.values())
        equal_weight_portfolio = np.zeros(min_length)

        for ticker, prices in all_tickers_prices.items():
            equal_weight_portfolio += prices[:min_length] / len(all_tickers_prices)

        eqw_returns = calculate_returns(equal_weight_portfolio)
        eqw_sharpe = calculate_sharpe(eqw_returns)
        eqw_total_return = (equal_weight_portfolio[-1] / equal_weight_portfolio[0] - 1) * 100

        return {
            'btc_sharpe': btc_sharpe,
            'btc_total_return': btc_total_return,
            'btc_current_price': btc_prices[-1],
            'btc_start_price': btc_prices[0],
            'eqw_sharpe': eqw_sharpe,
            'eqw_total_return': eqw_total_return,
            'num_periods': len(btc_prices)
        }

    except Exception as e:
        print(f"Error calculating benchmarks: {e}")
        return None


def calculate_alpha(strategy_return: float, benchmark_return: float) -> float:
    """Calculate alpha (excess return over benchmark)."""
    return strategy_return - benchmark_return


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
        tickers = ['AAVE/USD', 'AVAX/USD', 'BTC/USD', 'LINK/USD', 'ETH/USD', 'LTC/USD', 'UNI/USD']

        benchmarks = get_benchmark_data(csv_path, tickers)

        if benchmarks:
            print(f"Bitcoin-only Benchmark:")
            print(f"  Sharpe: {benchmarks['btc_sharpe']:.4f}")
            print(f"  Total Return: {benchmarks['btc_total_return']:+.2f}%")
            print()
            print(f"Equal-Weight Benchmark:")
            print(f"  Sharpe: {benchmarks['eqw_sharpe']:.4f}")
            print(f"  Total Return: {benchmarks['eqw_total_return']:+.2f}%")
        else:
            print("Could not calculate benchmarks")
    else:
        print("Usage: python benchmark_calculator.py <path_to_csv>")
