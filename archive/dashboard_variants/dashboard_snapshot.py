#!/usr/bin/env python3
"""
One-time snapshot of paper trading dashboard (non-interactive)
"""

from paper_trading_dashboard import PaperTradingDashboard
from pathlib import Path
import sys

def main():
    # Find CSV files
    ensemble_csv = sorted(Path("paper_trades").glob("watchdog_session_*.csv"))[-1]
    single_csv = Path("paper_trades/single_model_trial861.csv")

    print(f"\n📊 Paper Trading Dashboard Snapshot\n")
    print(f"Ensemble: {ensemble_csv.name}")
    print(f"Single:   {single_csv.name}")
    print(f"{'=' * 80}\n")

    # Create dashboard
    dashboard = PaperTradingDashboard(str(ensemble_csv), str(single_csv))

    # Load data
    ensemble_df = dashboard.load_data(ensemble_csv)
    single_df = dashboard.load_data(single_csv)

    # Calculate metrics
    ensemble_metrics = dashboard.calculate_metrics(ensemble_df)
    single_metrics = dashboard.calculate_metrics(single_df)

    # Render using simple text mode (not live)
    output = dashboard.render_simple(ensemble_metrics, single_metrics)
    print(output)

    return 0

if __name__ == "__main__":
    sys.exit(main())
