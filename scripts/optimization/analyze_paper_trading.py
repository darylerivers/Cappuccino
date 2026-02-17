#!/usr/bin/env python3
"""
Analyze paper trading performance
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

def load_recent_trades():
    """Load most recent trading data from watchdog sessions"""
    trade_files = sorted(Path("paper_trades").glob("watchdog_session_*.csv"))

    if not trade_files:
        return None

    # Load recent files
    all_data = []
    for f in trade_files[-10:]:  # Last 10 sessions
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                all_data.append(df)
        except:
            pass

    if not all_data:
        return None

    combined = pd.concat(all_data, ignore_index=True)
    combined['timestamp'] = pd.to_datetime(combined['timestamp'])
    combined = combined.sort_values('timestamp')
    combined = combined.drop_duplicates(subset=['timestamp'], keep='last')

    return combined

def analyze_performance(df):
    """Analyze trading performance metrics"""
    initial_value = 1000.0  # Starting portfolio value

    # Calculate returns
    df['portfolio_return'] = df['total_asset'] / initial_value - 1
    df['portfolio_pct'] = df['portfolio_return'] * 100

    # Calculate metrics
    total_return = df['total_asset'].iloc[-1] / initial_value - 1
    current_value = df['total_asset'].iloc[-1]

    # Maximum drawdown
    running_max = df['total_asset'].expanding().max()
    drawdown = (df['total_asset'] - running_max) / running_max
    max_drawdown = drawdown.min()

    # Win rate (positive vs negative hourly returns)
    hourly_returns = df['total_asset'].pct_change()
    positive_hours = (hourly_returns > 0).sum()
    negative_hours = (hourly_returns < 0).sum()
    win_rate = positive_hours / (positive_hours + negative_hours) * 100 if (positive_hours + negative_hours) > 0 else 0

    # Count trades (when holdings change)
    assets = ['BTC/USD', 'ETH/USD', 'LTC/USD', 'BCH/USD', 'LINK/USD', 'UNI/USD', 'AAVE/USD']
    holding_cols = [f'holding_{asset}' for asset in assets]

    trades = 0
    for col in holding_cols:
        if col in df.columns:
            trades += (df[col].diff().abs() > 0.001).sum()

    # Time in market
    total_holdings = df[holding_cols].sum(axis=1)
    time_in_market = (total_holdings > 0).sum() / len(df) * 100

    # Volatility
    volatility = hourly_returns.std() * np.sqrt(24 * 365)  # Annualized

    # Sharpe ratio (using hourly returns, annualized)
    mean_hourly_return = hourly_returns.mean()
    sharpe_ratio = (mean_hourly_return / hourly_returns.std()) * np.sqrt(24 * 365) if hourly_returns.std() > 0 else 0

    return {
        'current_value': current_value,
        'total_return': total_return,
        'total_return_pct': total_return * 100,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown * 100,
        'win_rate': win_rate,
        'total_trades': trades,
        'time_in_market': time_in_market,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'hours_traded': len(df),
        'days_traded': len(df) / 24
    }

def analyze_by_asset(df):
    """Analyze performance by asset"""
    assets = ['BTC/USD', 'ETH/USD', 'LTC/USD', 'BCH/USD', 'LINK/USD', 'UNI/USD', 'AAVE/USD']

    asset_stats = {}
    for asset in assets:
        holding_col = f'holding_{asset}'
        action_col = f'action_{asset}'

        if holding_col not in df.columns:
            continue

        # Times held
        times_held = (df[holding_col] > 0).sum()
        pct_held = times_held / len(df) * 100

        # Trades
        trades = (df[holding_col].diff().abs() > 0.001).sum()

        asset_stats[asset] = {
            'times_held': times_held,
            'pct_held': pct_held,
            'trades': trades
        }

    return asset_stats

def main():
    print("=" * 100)
    print("PAPER TRADING PERFORMANCE ANALYSIS")
    print("=" * 100)

    # Load trade data
    print("\nLoading trade data...")
    df = load_recent_trades()

    if df is None or len(df) == 0:
        print("❌ No trade data found")
        return

    print(f"✓ Loaded {len(df)} trading hours")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Days: {len(df) / 24:.1f}")

    # Analyze performance
    metrics = analyze_performance(df)

    print("\n" + "=" * 100)
    print("OVERALL PERFORMANCE")
    print("=" * 100)

    print(f"\nPortfolio Value:")
    print(f"  Initial:             $1,000.00")
    print(f"  Current:             ${metrics['current_value']:.2f}")
    print(f"  P&L:                 ${metrics['current_value'] - 1000:.2f} ({metrics['total_return_pct']:.2f}%)")

    status_emoji = "🟢" if metrics['total_return_pct'] > 0 else "🔴"
    print(f"  Status:              {status_emoji} {'Profitable' if metrics['total_return_pct'] > 0 else 'Losing'}")

    print(f"\nRisk Metrics:")
    print(f"  Max drawdown:        {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Volatility (annual): {metrics['volatility']:.2f}%")
    print(f"  Sharpe ratio:        {metrics['sharpe_ratio']:.4f}")

    print(f"\nTrading Activity:")
    print(f"  Total trades:        {metrics['total_trades']}")
    print(f"  Win rate:            {metrics['win_rate']:.1f}%")
    print(f"  Time in market:      {metrics['time_in_market']:.1f}%")
    print(f"  Hours traded:        {metrics['hours_traded']}")
    print(f"  Days traded:         {metrics['days_traded']:.1f}")

    # Analyze by asset
    print("\n" + "=" * 100)
    print("PERFORMANCE BY ASSET")
    print("=" * 100)

    asset_stats = analyze_by_asset(df)

    print(f"\n{'Asset':<12} {'Times Held':<12} {'% Time':<10} {'Trades':<10}")
    print("-" * 100)

    for asset, stats in sorted(asset_stats.items(), key=lambda x: x[1]['pct_held'], reverse=True):
        print(f"{asset:<12} {stats['times_held']:<12} {stats['pct_held']:>7.1f}%   {stats['trades']:<10}")

    # Recent performance
    print("\n" + "=" * 100)
    print("RECENT PERFORMANCE (Last 24 Hours)")
    print("=" * 100)

    if len(df) >= 24:
        recent_df = df.tail(24)
        recent_metrics = analyze_performance(recent_df)

        print(f"\nValue change:        ${recent_metrics['current_value'] - recent_df['total_asset'].iloc[0]:.2f}")
        print(f"Return:              {(recent_metrics['current_value'] / recent_df['total_asset'].iloc[0] - 1) * 100:.2f}%")
        print(f"Win rate:            {recent_metrics['win_rate']:.1f}%")
        print(f"Trades:              {recent_metrics['total_trades']}")
    else:
        print("\nNot enough data (< 24 hours)")

    # Load positions state
    try:
        with open('paper_trades/positions_state.json', 'r') as f:
            positions = json.load(f)

        print("\n" + "=" * 100)
        print("CURRENT POSITIONS")
        print("=" * 100)

        print(f"\nPortfolio value:     ${positions['portfolio_value']:.2f}")
        print(f"Cash:                ${positions['cash']:.2f}")
        print(f"Positions:           {len(positions['positions'])}")

        if positions['positions']:
            print("\nActive positions:")
            for pos in positions['positions']:
                print(f"  {pos}")
        else:
            print("\n  Currently in cash (no positions)")

        # Protection status
        prot = positions.get('portfolio_protection', {})
        if prot:
            print(f"\nProtection Status:")
            print(f"  High water mark:   ${prot.get('high_water_mark', 0):.2f}")
            print(f"  In cash mode:      {prot.get('in_cash_mode', False)}")
            print(f"  Profit taken:      {prot.get('profit_taken', False)}")
    except:
        pass

    print("\n" + "=" * 100)
    print("✓ Paper trading analysis complete!")
    print("=" * 100 + "\n")

if __name__ == "__main__":
    main()
