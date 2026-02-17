#!/usr/bin/env python3
"""
Analyze trade patterns from arena to understand why models aren't beating benchmarks.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def analyze_portfolio_trades(portfolio_data, model_id):
    """Analyze a single portfolio's trading behavior."""

    # Extract key metrics
    total_trades = portfolio_data['total_trades']
    winning_trades = portfolio_data['winning_trades']
    value_history = portfolio_data['value_history']
    holdings = portfolio_data['holdings']
    cash = portfolio_data['cash']
    initial_value = portfolio_data['initial_value']

    # Calculate metrics
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    # Analyze value trajectory
    if len(value_history) > 1:
        values = [v for _, v in value_history]
        timestamps = [datetime.fromisoformat(t.replace('Z', '+00:00')) for t, _ in value_history]

        # Calculate hourly returns
        returns = np.diff(values) / values[:-1] * 100
        avg_return_per_step = np.mean(returns)
        volatility = np.std(returns)

        # Time in arena
        time_in_arena = (timestamps[-1] - timestamps[0]).total_seconds() / 3600

        # Final return
        final_return = (values[-1] - initial_value) / initial_value * 100

        # Peak and drawdown
        peak_value = max(values)
        current_value = values[-1]
        drawdown_from_peak = (peak_value - current_value) / peak_value * 100

        # Trading frequency
        trades_per_hour = total_trades / time_in_arena if time_in_arena > 0 else 0

    else:
        avg_return_per_step = 0
        volatility = 0
        time_in_arena = 0
        final_return = 0
        peak_value = initial_value
        drawdown_from_peak = 0
        trades_per_hour = 0

    # Analyze portfolio composition
    num_positions = len([h for h in holdings.values() if h > 0])
    total_invested = initial_value - cash
    cash_pct = cash / initial_value * 100 if initial_value > 0 else 0

    return {
        'model_id': model_id,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'win_rate': win_rate,
        'final_return': final_return,
        'avg_return_per_step': avg_return_per_step,
        'volatility': volatility,
        'time_in_arena_hours': time_in_arena,
        'trades_per_hour': trades_per_hour,
        'peak_value': peak_value,
        'drawdown_from_peak': drawdown_from_peak,
        'num_positions': num_positions,
        'cash_pct': cash_pct,
        'holdings': holdings,
    }


def main():
    """Analyze all portfolios in arena."""

    print("=" * 80)
    print("ARENA TRADE PATTERN ANALYSIS")
    print("=" * 80)
    print()

    # Load arena state
    state_file = Path("arena_state/arena_state.json")
    if not state_file.exists():
        print("ERROR: arena_state.json not found!")
        return

    with state_file.open() as f:
        data = json.load(f)

    portfolios = data.get('portfolios', {})

    if not portfolios:
        print("No portfolios found in arena!")
        return

    print(f"Analyzing {len(portfolios)} portfolios...")
    print()

    # Analyze each portfolio
    analyses = []
    for model_id, portfolio_data in portfolios.items():
        analysis = analyze_portfolio_trades(portfolio_data, model_id)
        analyses.append(analysis)

    # Sort by final return
    analyses.sort(key=lambda x: x['final_return'], reverse=True)

    # Print summary table
    print("PORTFOLIO PERFORMANCE SUMMARY")
    print("-" * 80)
    print(f"{'Model':<15} {'Return%':<10} {'Trades':<8} {'Win%':<8} {'Tr/Hr':<8} {'Cash%':<8} {'Pos':<5}")
    print("-" * 80)

    for a in analyses:
        print(
            f"{a['model_id']:<15} {a['final_return']:>8.2f}% {a['total_trades']:>6} "
            f"{a['win_rate']:>6.1f}% {a['trades_per_hour']:>6.2f} {a['cash_pct']:>6.1f}% {a['num_positions']:>3}"
        )

    print()

    # Aggregate statistics
    print("AGGREGATE STATISTICS")
    print("-" * 80)

    avg_return = np.mean([a['final_return'] for a in analyses])
    avg_trades = np.mean([a['total_trades'] for a in analyses])
    avg_win_rate = np.mean([a['win_rate'] for a in analyses if a['total_trades'] > 0])
    avg_trades_per_hour = np.mean([a['trades_per_hour'] for a in analyses])
    avg_cash_pct = np.mean([a['cash_pct'] for a in analyses])

    print(f"Average Return: {avg_return:.2f}%")
    print(f"Average Trades: {avg_trades:.1f}")
    print(f"Average Win Rate: {avg_win_rate:.1f}%")
    print(f"Average Trades/Hour: {avg_trades_per_hour:.2f}")
    print(f"Average Cash %: {avg_cash_pct:.1f}%")
    print()

    # Trading frequency analysis
    print("TRADING FREQUENCY DISTRIBUTION")
    print("-" * 80)

    low_freq = [a for a in analyses if a['trades_per_hour'] < 0.5]
    med_freq = [a for a in analyses if 0.5 <= a['trades_per_hour'] < 1.0]
    high_freq = [a for a in analyses if a['trades_per_hour'] >= 1.0]

    print(f"Low frequency (<0.5 tr/hr): {len(low_freq)} models")
    if low_freq:
        print(f"  Avg return: {np.mean([a['final_return'] for a in low_freq]):.2f}%")

    print(f"Medium frequency (0.5-1.0 tr/hr): {len(med_freq)} models")
    if med_freq:
        print(f"  Avg return: {np.mean([a['final_return'] for a in med_freq]):.2f}%")

    print(f"High frequency (>1.0 tr/hr): {len(high_freq)} models")
    if high_freq:
        print(f"  Avg return: {np.mean([a['final_return'] for a in high_freq]):.2f}%")
    print()

    # Portfolio concentration analysis
    print("PORTFOLIO CONCENTRATION")
    print("-" * 80)

    concentrated = [a for a in analyses if a['num_positions'] <= 2]
    diversified = [a for a in analyses if a['num_positions'] >= 3]

    print(f"Concentrated (≤2 positions): {len(concentrated)} models")
    if concentrated:
        print(f"  Avg return: {np.mean([a['final_return'] for a in concentrated]):.2f}%")

    print(f"Diversified (≥3 positions): {len(diversified)} models")
    if diversified:
        print(f"  Avg return: {np.mean([a['final_return'] for a in diversified]):.2f}%")
    print()

    # Top holdings analysis
    print("MOST POPULAR HOLDINGS")
    print("-" * 80)

    all_tickers = defaultdict(int)
    for a in analyses:
        for ticker, qty in a['holdings'].items():
            if qty > 0:
                all_tickers[ticker] += 1

    sorted_tickers = sorted(all_tickers.items(), key=lambda x: x[1], reverse=True)
    for ticker, count in sorted_tickers:
        print(f"{ticker:<12} held by {count}/{len(analyses)} models ({count/len(analyses)*100:.1f}%)")

    print()

    # Detailed analysis of top 3 and bottom 3
    print("TOP 3 MODELS - DETAILED")
    print("-" * 80)
    for i, a in enumerate(analyses[:3], 1):
        print(f"\n{i}. {a['model_id']}")
        print(f"   Return: {a['final_return']:.2f}%")
        print(f"   Trades: {a['total_trades']} ({a['trades_per_hour']:.2f}/hr)")
        print(f"   Win rate: {a['win_rate']:.1f}%")
        print(f"   Drawdown from peak: {a['drawdown_from_peak']:.2f}%")
        print(f"   Holdings: {a['holdings']}")
        print(f"   Cash: ${a['cash_pct']:.1f}%")

    print()
    print("BOTTOM 3 MODELS - DETAILED")
    print("-" * 80)
    for i, a in enumerate(analyses[-3:], 1):
        print(f"\n{i}. {a['model_id']}")
        print(f"   Return: {a['final_return']:.2f}%")
        print(f"   Trades: {a['total_trades']} ({a['trades_per_hour']:.2f}/hr)")
        print(f"   Win rate: {a['win_rate']:.1f}%")
        print(f"   Drawdown from peak: {a['drawdown_from_peak']:.2f}%")
        print(f"   Holdings: {a['holdings']}")
        print(f"   Cash: ${a['cash_pct']:.1f}%")

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
