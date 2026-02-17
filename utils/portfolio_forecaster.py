#!/usr/bin/env python3
"""
Portfolio Forecasting Tool

Analyzes paper trading performance and projects future portfolio valuation
using multiple methodologies:
- Historical momentum extrapolation
- Monte Carlo simulation with volatility
- Conservative/Moderate/Aggressive scenarios

Usage:
    python portfolio_forecaster.py
    python portfolio_forecaster.py --days 30 --simulations 10000
    python portfolio_forecaster.py --positions-file paper_trades/positions_state.json
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class PortfolioForecaster:
    """Forecast future portfolio valuations based on historical performance."""

    def __init__(self, positions_file: Path, session_files: List[Path]):
        self.positions_file = positions_file
        self.session_files = session_files

        # Load current state
        self.current_state = self._load_positions()

        # Load historical performance
        self.historical_df = self._load_historical_sessions()

    def _load_positions(self) -> Dict:
        """Load current portfolio positions state."""
        if not self.positions_file.exists():
            raise FileNotFoundError(f"Positions file not found: {self.positions_file}")

        with open(self.positions_file) as f:
            return json.load(f)

    def _load_historical_sessions(self) -> pd.DataFrame:
        """Load and combine historical trading sessions."""
        dfs = []
        for session_file in self.session_files:
            if session_file.exists():
                try:
                    df = pd.read_csv(session_file)
                    if len(df) > 0:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        dfs.append(df)
                except Exception as e:
                    print(f"Warning: Could not load {session_file}: {e}")

        if not dfs:
            print("No historical session data found - using current state only")
            return pd.DataFrame()

        # Combine and sort
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values('timestamp').reset_index(drop=True)
        combined = combined.drop_duplicates(subset=['timestamp'], keep='last')

        return combined

    def get_performance_metrics(self) -> Dict:
        """Calculate key performance metrics from historical data."""
        current_value = self.current_state['portfolio_value']
        initial_value = self.current_state['portfolio_protection']['initial_value']
        high_water_mark = self.current_state['portfolio_protection']['high_water_mark']

        total_return_pct = (current_value / initial_value - 1) * 100
        drawdown_from_peak_pct = (high_water_mark - current_value) / high_water_mark * 100

        metrics = {
            'current_value': current_value,
            'initial_value': initial_value,
            'total_return_pct': total_return_pct,
            'high_water_mark': high_water_mark,
            'drawdown_from_peak_pct': drawdown_from_peak_pct,
        }

        # Calculate volatility and return rate if we have historical data
        if len(self.historical_df) > 1:
            self.historical_df['returns'] = self.historical_df['total_asset'].pct_change()

            # Calculate time-based metrics
            start_time = self.historical_df['timestamp'].iloc[0]
            end_time = self.historical_df['timestamp'].iloc[-1]
            days_elapsed = (end_time - start_time).total_seconds() / 86400

            if days_elapsed > 0:
                total_return = (self.historical_df['total_asset'].iloc[-1] /
                               self.historical_df['total_asset'].iloc[0] - 1)
                daily_return = (1 + total_return) ** (1 / days_elapsed) - 1

                metrics['daily_return_pct'] = daily_return * 100
                metrics['volatility_pct'] = self.historical_df['returns'].std() * 100
                metrics['sharpe_ratio'] = (daily_return / self.historical_df['returns'].std()
                                          if self.historical_df['returns'].std() > 0 else 0)

        # Estimate metrics from current state if no historical data
        if 'daily_return_pct' not in metrics:
            # Assume the return happened over a reasonable timeframe (e.g., 7 days)
            assumed_days = 7
            total_return = current_value / initial_value - 1
            daily_return = (1 + total_return) ** (1 / assumed_days) - 1
            metrics['daily_return_pct'] = daily_return * 100
            metrics['volatility_pct'] = 1.5  # Conservative estimate
            metrics['sharpe_ratio'] = metrics['daily_return_pct'] / metrics['volatility_pct']

        return metrics

    def forecast_linear(self, days: int) -> pd.DataFrame:
        """Simple linear extrapolation of current return rate."""
        metrics = self.get_performance_metrics()
        current_value = metrics['current_value']
        daily_return = metrics['daily_return_pct'] / 100

        dates = [datetime.now() + timedelta(days=i) for i in range(days + 1)]
        values = [current_value * (1 + daily_return) ** i for i in range(days + 1)]

        return pd.DataFrame({
            'date': dates,
            'portfolio_value': values,
            'scenario': 'linear'
        })

    def forecast_monte_carlo(self, days: int, simulations: int = 10000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Monte Carlo simulation with volatility."""
        metrics = self.get_performance_metrics()
        current_value = metrics['current_value']
        daily_return = metrics['daily_return_pct'] / 100
        volatility = metrics['volatility_pct'] / 100

        # Run simulations
        np.random.seed(42)  # For reproducibility

        # Store all simulation paths
        all_paths = np.zeros((simulations, days + 1))
        all_paths[:, 0] = current_value

        for sim in range(simulations):
            value = current_value
            for day in range(1, days + 1):
                # Random daily return based on historical mean and volatility
                daily_change = np.random.normal(daily_return, volatility)
                value *= (1 + daily_change)
                all_paths[sim, day] = value

        # Calculate percentiles for each day
        dates = [datetime.now() + timedelta(days=i) for i in range(days + 1)]

        percentiles_df = pd.DataFrame({
            'date': dates,
            'p5': np.percentile(all_paths, 5, axis=0),    # 5th percentile (pessimistic)
            'p25': np.percentile(all_paths, 25, axis=0),   # 25th percentile
            'p50': np.percentile(all_paths, 50, axis=0),   # Median
            'p75': np.percentile(all_paths, 75, axis=0),   # 75th percentile
            'p95': np.percentile(all_paths, 95, axis=0),   # 95th percentile (optimistic)
            'mean': np.mean(all_paths, axis=0),
        })

        # Sample paths for visualization
        sample_paths = []
        num_samples = min(100, simulations)
        sample_indices = np.random.choice(simulations, num_samples, replace=False)

        for idx in sample_indices:
            for day in range(days + 1):
                sample_paths.append({
                    'date': dates[day],
                    'portfolio_value': all_paths[idx, day],
                    'simulation_id': idx
                })

        samples_df = pd.DataFrame(sample_paths)

        return percentiles_df, samples_df

    def forecast_scenarios(self, days: int) -> pd.DataFrame:
        """Generate conservative/moderate/aggressive scenarios."""
        metrics = self.get_performance_metrics()
        current_value = metrics['current_value']
        daily_return = metrics['daily_return_pct'] / 100

        # Define scenario multipliers
        scenarios = {
            'conservative': daily_return * 0.5,    # 50% of current rate
            'moderate': daily_return,              # Current rate
            'aggressive': daily_return * 1.5,      # 150% of current rate
        }

        dates = [datetime.now() + timedelta(days=i) for i in range(days + 1)]

        data = []
        for scenario_name, scenario_return in scenarios.items():
            for i, date in enumerate(dates):
                value = current_value * (1 + scenario_return) ** i
                data.append({
                    'date': date,
                    'portfolio_value': value,
                    'scenario': scenario_name
                })

        return pd.DataFrame(data)

    def generate_report(self, days: int = 30, simulations: int = 10000) -> str:
        """Generate comprehensive forecast report."""
        metrics = self.get_performance_metrics()

        # Run forecasts
        linear_df = self.forecast_linear(days)
        scenarios_df = self.forecast_scenarios(days)
        mc_percentiles_df, mc_samples_df = self.forecast_monte_carlo(days, simulations)

        # Build report
        report = []
        report.append("=" * 80)
        report.append("PORTFOLIO FORECAST REPORT")
        report.append("=" * 80)
        report.append("")

        # Current state
        report.append("CURRENT STATE")
        report.append("-" * 80)
        report.append(f"  Portfolio Value:      ${metrics['current_value']:,.2f}")
        report.append(f"  Initial Value:        ${metrics['initial_value']:,.2f}")
        report.append(f"  Total Return:         {metrics['total_return_pct']:+.2f}%")
        report.append(f"  High Water Mark:      ${metrics['high_water_mark']:,.2f}")
        report.append(f"  Current Drawdown:     {metrics['drawdown_from_peak_pct']:.2f}%")
        report.append("")

        # Performance metrics
        report.append("PERFORMANCE METRICS")
        report.append("-" * 80)
        report.append(f"  Daily Return (avg):   {metrics['daily_return_pct']:+.3f}%")
        report.append(f"  Volatility (std):     {metrics['volatility_pct']:.3f}%")
        report.append(f"  Sharpe Ratio:         {metrics['sharpe_ratio']:.3f}")
        report.append("")

        # Forecast summary
        report.append(f"FORECAST PROJECTIONS ({days} days)")
        report.append("-" * 80)

        # Linear projection
        linear_final = linear_df['portfolio_value'].iloc[-1]
        linear_return = (linear_final / metrics['current_value'] - 1) * 100
        report.append(f"\nLinear Extrapolation:")
        report.append(f"  Projected Value:      ${linear_final:,.2f}")
        report.append(f"  Expected Return:      {linear_return:+.2f}%")

        # Scenarios
        report.append(f"\nScenario Analysis:")
        for scenario in ['conservative', 'moderate', 'aggressive']:
            scenario_data = scenarios_df[scenarios_df['scenario'] == scenario]
            final_value = scenario_data['portfolio_value'].iloc[-1]
            scenario_return = (final_value / metrics['current_value'] - 1) * 100
            report.append(f"  {scenario.capitalize():12s}:      ${final_value:,.2f}  ({scenario_return:+.2f}%)")

        # Monte Carlo
        report.append(f"\nMonte Carlo Simulation ({simulations:,} runs):")
        mc_final = mc_percentiles_df.iloc[-1]
        report.append(f"  5th Percentile:       ${mc_final['p5']:,.2f}  (pessimistic)")
        report.append(f"  25th Percentile:      ${mc_final['p25']:,.2f}")
        report.append(f"  Median (50th):        ${mc_final['p50']:,.2f}")
        report.append(f"  75th Percentile:      ${mc_final['p75']:,.2f}")
        report.append(f"  95th Percentile:      ${mc_final['p95']:,.2f}  (optimistic)")
        report.append(f"  Mean:                 ${mc_final['mean']:,.2f}")
        report.append("")

        # Risk assessment
        report.append("RISK ASSESSMENT")
        report.append("-" * 80)
        prob_profit = (mc_samples_df.groupby('simulation_id')['portfolio_value']
                       .last()
                       .apply(lambda x: x > metrics['current_value'])
                       .mean() * 100)
        prob_loss = 100 - prob_profit

        report.append(f"  Probability of Profit:  {prob_profit:.1f}%")
        report.append(f"  Probability of Loss:    {prob_loss:.1f}%")

        # Value at Risk (VaR)
        final_values = mc_samples_df.groupby('simulation_id')['portfolio_value'].last()
        var_95 = np.percentile(final_values, 5)  # 5th percentile = 95% confidence
        max_drawdown = metrics['current_value'] - var_95
        var_pct = (max_drawdown / metrics['current_value']) * 100

        report.append(f"  Value at Risk (95%):    ${var_95:,.2f}  (max loss: ${max_drawdown:,.2f}, {var_pct:.1f}%)")
        report.append("")

        # Positions breakdown
        report.append("CURRENT POSITIONS")
        report.append("-" * 80)
        for pos in self.current_state['positions']:
            report.append(f"  {pos['ticker']:10s}  "
                         f"Qty: {pos['holdings']:8.4f}  "
                         f"Value: ${pos['position_value']:8.2f}  "
                         f"P&L: {pos['pnl_pct']:+6.2f}%")
        report.append("")

        report.append("=" * 80)
        report.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)

        return "\n".join(report)

    def save_forecast_data(self, days: int = 30, simulations: int = 10000, output_dir: Path = Path("paper_trades")):
        """Save forecast data to files for visualization."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate forecasts
        linear_df = self.forecast_linear(days)
        scenarios_df = self.forecast_scenarios(days)
        mc_percentiles_df, mc_samples_df = self.forecast_monte_carlo(days, simulations)

        # Save to CSV
        linear_df.to_csv(output_dir / "forecast_linear.csv", index=False)
        scenarios_df.to_csv(output_dir / "forecast_scenarios.csv", index=False)
        mc_percentiles_df.to_csv(output_dir / "forecast_monte_carlo_percentiles.csv", index=False)

        # Don't save all MC samples (too large), just summary
        summary = {
            'method': 'monte_carlo',
            'simulations': simulations,
            'days': days,
            'final_values': {
                'p5': float(mc_percentiles_df.iloc[-1]['p5']),
                'p25': float(mc_percentiles_df.iloc[-1]['p25']),
                'p50': float(mc_percentiles_df.iloc[-1]['p50']),
                'p75': float(mc_percentiles_df.iloc[-1]['p75']),
                'p95': float(mc_percentiles_df.iloc[-1]['p95']),
                'mean': float(mc_percentiles_df.iloc[-1]['mean']),
            }
        }

        with open(output_dir / "forecast_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"✓ Forecast data saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Portfolio Forecasting Tool")
    parser.add_argument("--positions-file", type=Path,
                       default=Path("paper_trades/positions_state.json"),
                       help="Path to positions state JSON file")
    parser.add_argument("--days", type=int, default=30,
                       help="Number of days to forecast (default: 30)")
    parser.add_argument("--simulations", type=int, default=10000,
                       help="Number of Monte Carlo simulations (default: 10000)")
    parser.add_argument("--save-data", action="store_true",
                       help="Save forecast data to CSV files")

    args = parser.parse_args()

    # Find available session files
    session_files = [
        Path("paper_trades/ensemble_session.csv"),
        Path("paper_trades/alpaca_session.csv"),
    ]

    # Also look for dated session files
    paper_trades_dir = Path("paper_trades")
    if paper_trades_dir.exists():
        for f in paper_trades_dir.glob("*session*.csv"):
            if f not in session_files:
                session_files.append(f)

    try:
        forecaster = PortfolioForecaster(args.positions_file, session_files)

        # Generate and display report
        report = forecaster.generate_report(days=args.days, simulations=args.simulations)
        print(report)

        # Save data if requested
        if args.save_data:
            forecaster.save_forecast_data(days=args.days, simulations=args.simulations)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
