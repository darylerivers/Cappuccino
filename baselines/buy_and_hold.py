"""Buy-and-Hold Baseline Strategy

This baseline implements equal-weight buy-and-hold to compare against DRL models.
Answers: "Does reinforcement learning actually beat passive investing?"

Usage:
    python baselines/buy_and_hold.py --data data/price_array_train.npy --initial-capital 1000
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import json
from datetime import datetime


class BuyAndHoldBaseline:
    """Equal-weight buy-and-hold portfolio with rebalancing."""

    def __init__(self, initial_capital: float = 1000, rebalance_frequency: int = 0):
        """Initialize baseline.

        Args:
            initial_capital: Starting capital
            rebalance_frequency: Rebalance every N steps (0 = no rebalancing)
        """
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        self.portfolio_values = []
        self.returns = []

    def run(self, price_array: np.ndarray) -> Dict[str, float]:
        """Execute buy-and-hold strategy.

        Args:
            price_array: Shape (timesteps, n_assets) with prices

        Returns:
            Dictionary of performance metrics
        """
        n_steps, n_assets = price_array.shape

        # Initial allocation: equal weight
        initial_prices = price_array[0]
        cash_per_asset = self.initial_capital / n_assets
        holdings = cash_per_asset / initial_prices  # shares of each asset

        # Track portfolio value over time
        self.portfolio_values = []
        for t in range(n_steps):
            prices = price_array[t]
            portfolio_value = np.sum(holdings * prices)
            self.portfolio_values.append(portfolio_value)

            # Rebalance if needed
            if self.rebalance_frequency > 0 and t > 0 and t % self.rebalance_frequency == 0:
                cash_per_asset = portfolio_value / n_assets
                holdings = cash_per_asset / prices

        # Calculate returns
        self.returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]

        # Calculate metrics
        metrics = self._calculate_metrics()
        return metrics

    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        portfolio_values = np.array(self.portfolio_values)
        returns = self.returns

        # Total return
        total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100

        # Annualized return (assuming 1h timeframe, 24*365 periods per year)
        n_periods = len(portfolio_values)
        periods_per_year = 24 * 365
        years = n_periods / periods_per_year
        if years > 0:
            annualized_return = ((portfolio_values[-1] / portfolio_values[0]) ** (1 / years) - 1) * 100
        else:
            annualized_return = 0.0

        # Volatility (annualized)
        if len(returns) > 0:
            volatility = np.std(returns) * np.sqrt(periods_per_year) * 100
        else:
            volatility = 0.0

        # Sharpe ratio (assuming 0% risk-free rate)
        if volatility > 0:
            sharpe_ratio = annualized_return / volatility
        else:
            sharpe_ratio = 0.0

        # Maximum drawdown
        cumulative = portfolio_values / portfolio_values[0]
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown) * 100

        # Win rate
        if len(returns) > 0:
            win_rate = (np.sum(returns > 0) / len(returns)) * 100
        else:
            win_rate = 0.0

        # Sortino ratio (downside deviation)
        if len(returns) > 0:
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_deviation = np.std(downside_returns) * np.sqrt(periods_per_year) * 100
                if downside_deviation > 0:
                    sortino_ratio = annualized_return / downside_deviation
                else:
                    sortino_ratio = 0.0
            else:
                sortino_ratio = sharpe_ratio  # No downside
        else:
            sortino_ratio = 0.0

        return {
            "total_return_pct": float(total_return),
            "annualized_return_pct": float(annualized_return),
            "volatility_pct": float(volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "sortino_ratio": float(sortino_ratio),
            "max_drawdown_pct": float(max_drawdown),
            "win_rate_pct": float(win_rate),
            "initial_value": float(self.portfolio_values[0]),
            "final_value": float(self.portfolio_values[-1]),
            "n_periods": int(len(self.portfolio_values)),
        }


def load_price_data(data_path: Path) -> np.ndarray:
    """Load price array from NPY file."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = np.load(data_path)
    print(f"Loaded price data: shape {data.shape}")
    return data


def main():
    parser = argparse.ArgumentParser(description="Buy-and-Hold Baseline")
    parser.add_argument("--data", type=Path, default=Path("data/price_array_val.npy"),
                        help="Path to price array NPY file")
    parser.add_argument("--initial-capital", type=float, default=1000,
                        help="Initial capital")
    parser.add_argument("--rebalance-freq", type=int, default=0,
                        help="Rebalance every N steps (0=no rebalancing)")
    parser.add_argument("--output", type=Path, default=Path("baselines/results_buy_and_hold.json"),
                        help="Output JSON file")
    args = parser.parse_args()

    # Load data
    price_array = load_price_data(args.data)

    # Run baseline
    print(f"\nRunning Buy-and-Hold Baseline...")
    print(f"  Initial Capital: ${args.initial_capital}")
    print(f"  Rebalance Frequency: {args.rebalance_freq if args.rebalance_freq > 0 else 'Never'}")

    baseline = BuyAndHoldBaseline(
        initial_capital=args.initial_capital,
        rebalance_frequency=args.rebalance_freq
    )

    metrics = baseline.run(price_array)

    # Display results
    print("\n" + "="*50)
    print("Buy-and-Hold Performance Metrics")
    print("="*50)
    print(f"Total Return:        {metrics['total_return_pct']:>10.2f}%")
    print(f"Annualized Return:   {metrics['annualized_return_pct']:>10.2f}%")
    print(f"Volatility:          {metrics['volatility_pct']:>10.2f}%")
    print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:>10.3f}")
    print(f"Sortino Ratio:       {metrics['sortino_ratio']:>10.3f}")
    print(f"Max Drawdown:        {metrics['max_drawdown_pct']:>10.2f}%")
    print(f"Win Rate:            {metrics['win_rate_pct']:>10.2f}%")
    print(f"Initial Value:       ${metrics['initial_value']:>10.2f}")
    print(f"Final Value:         ${metrics['final_value']:>10.2f}")
    print(f"Periods:             {metrics['n_periods']:>10d}")
    print("="*50)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "strategy": "buy_and_hold",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "initial_capital": args.initial_capital,
            "rebalance_frequency": args.rebalance_freq,
            "data_file": str(args.data),
        },
        "metrics": metrics,
    }

    with args.output.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output}")

    return metrics


if __name__ == "__main__":
    main()
