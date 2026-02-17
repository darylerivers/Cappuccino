#!/usr/bin/env python3
"""
Real-time monitoring dashboard for paper trading
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime
import time

TRADING_DIR = Path(__file__).parent

def load_latest_performance():
    """Load the most recent performance file"""
    results_dir = TRADING_DIR / "results"

    perf_files = list(results_dir.glob("performance_*.csv"))
    if not perf_files:
        return None

    latest = max(perf_files, key=lambda p: p.stat().st_mtime)
    return pd.read_csv(latest)

def calculate_metrics(df):
    """Calculate performance metrics from dataframe"""
    if len(df) < 2:
        return {}

    values = df['portfolio_value'].values
    returns = np.diff(values) / values[:-1]

    # Sharpe ratio
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

    # Max drawdown
    cumulative = values
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    # Win rate
    win_rate = (returns > 0).mean() if len(returns) > 0 else 0

    # Total return
    total_return = (values[-1] / values[0] - 1) * 100

    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'current_value': values[-1],
        'initial_value': values[0],
        'iterations': len(df),
        'last_update': df['timestamp'].iloc[-1] if 'timestamp' in df.columns else 'N/A'
    }

def display_dashboard():
    """Display monitoring dashboard"""

    print("\033[2J\033[H")  # Clear screen
    print("="*70)
    print("PAPER TRADING MONITOR - TRIAL #965")
    print("="*70)
    print(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Load performance data
    df = load_latest_performance()

    if df is None or len(df) == 0:
        print("\n⚠️  No performance data available yet")
        print("   Waiting for trading to start...")
        return

    # Calculate metrics
    metrics = calculate_metrics(df)

    # Display model info
    try:
        with open(TRADING_DIR / "config/model_config.json", 'r') as f:
            model_config = json.load(f)

        print("\n【Model Information】")
        print(f"  Trial:           #965 Rerun")
        print(f"  Training Sharpe: {model_config['model_info']['objective_sharpe']:.6f}")
        print(f"  Stress Test:     {model_config['stress_test_results']['overall_sharpe']:.2f} Sharpe")
        print(f"  Bear Market:     {model_config['stress_test_results']['bear_market_sharpe']:.2f} Sharpe")
    except:
        pass

    # Display current performance
    print("\n【Live Performance】")
    print(f"  Portfolio Value:  ${metrics['current_value']:,.2f}")
    print(f"  Initial Value:    ${metrics['initial_value']:,.2f}")
    print(f"  Total Return:     {metrics['total_return']:+.2f}%")
    print(f"  Sharpe Ratio:     {metrics['sharpe']:.3f}")
    print(f"  Max Drawdown:     {metrics['max_drawdown']*100:.2f}%")
    print(f"  Win Rate:         {metrics['win_rate']*100:.1f}%")
    print(f"  Iterations:       {metrics['iterations']}")
    print(f"  Last Update:      {metrics['last_update']}")

    # Performance indicators
    print("\n【Status】")

    # Sharpe status
    if metrics['sharpe'] > 2.0:
        sharpe_status = "✓ Excellent"
    elif metrics['sharpe'] > 1.0:
        sharpe_status = "✓ Good"
    elif metrics['sharpe'] > 0:
        sharpe_status = "⚠️  Modest"
    else:
        sharpe_status = "✗ Poor"
    print(f"  Sharpe:      {sharpe_status}")

    # Drawdown status
    if metrics['max_drawdown'] > -0.05:
        dd_status = "✓ Low"
    elif metrics['max_drawdown'] > -0.10:
        dd_status = "⚠️  Moderate"
    elif metrics['max_drawdown'] > -0.15:
        dd_status = "⚠️  High"
    else:
        dd_status = "✗ Critical"
    print(f"  Drawdown:    {dd_status}")

    # Return status
    if metrics['total_return'] > 5:
        return_status = "✓ Profitable"
    elif metrics['total_return'] > 0:
        return_status = "✓ Positive"
    elif metrics['total_return'] > -5:
        return_status = "⚠️  Small Loss"
    else:
        return_status = "✗ Losing"
    print(f"  Return:      {return_status}")

    # Recent trend (last 10 iterations)
    if len(df) >= 10:
        recent_values = df['portfolio_value'].tail(10).values
        recent_trend = (recent_values[-1] / recent_values[0] - 1) * 100
        trend_icon = "📈" if recent_trend > 0 else "📉"
        print(f"  Recent Trend: {trend_icon} {recent_trend:+.2f}% (last 10 iterations)")

    print("\n" + "="*70)
    print("Press Ctrl+C to exit monitor")
    print("="*70)

def main():
    """Main monitoring loop"""

    print("Starting paper trading monitor...")
    print("Press Ctrl+C to exit\n")

    try:
        while True:
            display_dashboard()
            time.sleep(10)  # Update every 10 seconds
    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")
        print("Trading continues in background.")

if __name__ == "__main__":
    main()
