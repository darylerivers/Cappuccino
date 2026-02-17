#!/usr/bin/env python3
"""
Compare trading performance: Agent vs Buy-and-Hold
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

def load_current_session():
    """Load current paper trading session."""
    log_file = Path("logs/paper_trading_BEST.log")
    if log_file.exists():
        df = pd.read_csv(log_file)
        if len(df) > 0:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
    return None

def load_positions():
    """Load current positions."""
    pos_file = Path("paper_trades/positions_state.json")
    if pos_file.exists():
        with open(pos_file) as f:
            return json.load(f)
    return None

def calculate_buy_hold_performance(df, tickers):
    """Calculate buy-and-hold performance for equal-weighted portfolio."""
    if df is None or len(df) == 0:
        return None

    results = {}
    for ticker in tickers:
        price_col = f'price_{ticker}'
        if price_col in df.columns:
            start_price = df.iloc[0][price_col]
            end_price = df.iloc[-1][price_col]
            returns = ((end_price - start_price) / start_price) * 100
            results[ticker] = {
                'start_price': start_price,
                'end_price': end_price,
                'return_pct': returns
            }

    # Equal-weighted portfolio
    avg_return = sum(r['return_pct'] for r in results.values()) / len(results) if results else 0

    return {
        'individual': results,
        'portfolio_return': avg_return
    }

def calculate_agent_performance(df):
    """Calculate agent's actual performance."""
    if df is None or len(df) == 0:
        return None

    start_value = df.iloc[0]['total_asset']
    end_value = df.iloc[-1]['total_asset']
    returns = ((end_value - start_value) / start_value) * 100

    return {
        'start_value': start_value,
        'end_value': end_value,
        'return_pct': returns,
        'pnl': end_value - start_value
    }

def show_hourly_performance(df, tickers):
    """Show hour-by-hour performance breakdown."""
    if df is None or len(df) < 2:
        return

    print("HOURLY PERFORMANCE")
    print("-" * 120)
    print(f"{'Time':<20s} {'Portfolio':>12s} {'Hourly Δ':>10s} {'Total Δ':>10s} {'Cash':>12s} {'Holdings':>30s}")
    print("-" * 120)

    initial_value = df.iloc[0]['total_asset']

    for idx, row in df.iterrows():
        timestamp = row['timestamp']
        portfolio_value = row['total_asset']
        cash = row['cash']

        # Calculate changes
        if idx > 0:
            prev_value = df.iloc[idx-1]['total_asset']
            hourly_change = ((portfolio_value - prev_value) / prev_value) * 100
        else:
            hourly_change = 0

        total_change = ((portfolio_value - initial_value) / initial_value) * 100

        # Find active holdings
        holdings = []
        for ticker in tickers:
            holding_col = f'holding_{ticker}'
            if holding_col in row and row[holding_col] > 0.001:
                holdings.append(ticker.split('/')[0])

        holdings_str = ', '.join(holdings[:3]) if holdings else 'CASH'
        if len(holdings) > 3:
            holdings_str += f' +{len(holdings)-3}'

        color = ''
        reset = ''
        if hourly_change > 0:
            color = '\033[92m'  # Green
            reset = '\033[0m'
        elif hourly_change < 0:
            color = '\033[91m'  # Red
            reset = '\033[0m'

        print(f"{timestamp.strftime('%Y-%m-%d %H:%M'):<20s} "
              f"${portfolio_value:>11,.2f} "
              f"{color}{hourly_change:>+9.2f}%{reset} "
              f"{total_change:>+9.2f}% "
              f"${cash:>11,.2f} "
              f"{holdings_str:>30s}")

    print()

def main():
    print("=" * 120)
    print("TRADING PERFORMANCE COMPARISON: AGENT vs BUY-AND-HOLD")
    print("=" * 120)
    print()

    # Load data
    df = load_current_session()
    positions = load_positions()

    if df is None or len(df) == 0:
        print("❌ No trading data available yet.")
        print()
        print("The paper trader just started. Check back in a few hours to see trading activity.")
        print()
        print("To monitor in real-time:")
        print("  tail -f logs/paper_trading_BEST_console.log")
        print("  python show_current_trading_status.py")
        return

    # Extract tickers
    tickers = [col.replace('price_', '') for col in df.columns if col.startswith('price_')]

    print(f"📊 DATA SUMMARY")
    print(f"   Session Start:    {df.iloc[0]['timestamp']}")
    print(f"   Last Update:      {df.iloc[-1]['timestamp']}")
    print(f"   Duration:         {df.iloc[-1]['timestamp'] - df.iloc[0]['timestamp']}")
    print(f"   Data Points:      {len(df)} hours")
    print(f"   Tickers:          {', '.join(t.split('/')[0] for t in tickers)}")
    print()

    # Agent performance
    agent_perf = calculate_agent_performance(df)
    if agent_perf:
        print("🤖 AGENT PERFORMANCE (Ensemble of 20 Models)")
        print("-" * 120)
        print(f"   Initial Value:    ${agent_perf['start_value']:,.2f}")
        print(f"   Current Value:    ${agent_perf['end_value']:,.2f}")

        pnl_color = '\033[92m' if agent_perf['pnl'] >= 0 else '\033[91m'
        reset = '\033[0m'
        print(f"   P&L:              {pnl_color}${agent_perf['pnl']:+,.2f} ({agent_perf['return_pct']:+.2f}%){reset}")
        print()

    # Buy-and-hold performance
    buyhold_perf = calculate_buy_hold_performance(df, tickers)
    if buyhold_perf:
        print("📈 BUY-AND-HOLD PERFORMANCE (Equal-Weighted Portfolio)")
        print("-" * 120)

        for ticker, perf in sorted(buyhold_perf['individual'].items(),
                                   key=lambda x: x[1]['return_pct'], reverse=True):
            color = '\033[92m' if perf['return_pct'] >= 0 else '\033[91m'
            reset = '\033[0m'
            ticker_short = ticker.split('/')[0]
            print(f"   {ticker_short:<8s}  ${perf['start_price']:>10,.2f} → ${perf['end_price']:>10,.2f}  "
                  f"{color}{perf['return_pct']:>+8.2f}%{reset}")

        print(f"\n   {'Portfolio (avg):':<10s}  "
              f"{buyhold_perf['portfolio_return']:>+8.2f}%")
        print()

    # Alpha calculation
    if agent_perf and buyhold_perf:
        alpha = agent_perf['return_pct'] - buyhold_perf['portfolio_return']
        print("📊 ALPHA (Agent vs Market)")
        print("-" * 120)
        print(f"   Agent Return:     {agent_perf['return_pct']:+.2f}%")
        print(f"   Market Return:    {buyhold_perf['portfolio_return']:+.2f}%")

        alpha_color = '\033[92m' if alpha >= 0 else '\033[91m'
        reset = '\033[0m'
        print(f"   Alpha:            {alpha_color}{alpha:+.2f}%{reset}")

        if alpha > 0:
            print(f"   ✓ Agent is OUTPERFORMING the market by {abs(alpha):.2f}%")
        else:
            print(f"   ⚠ Agent is UNDERPERFORMING the market by {abs(alpha):.2f}%")
        print()

    # Hourly breakdown
    if len(df) > 1:
        show_hourly_performance(df, tickers)

    # Current positions
    if positions and positions['positions']:
        print("💼 CURRENT POSITIONS")
        print("-" * 120)
        for pos in positions['positions']:
            pnl_color = '\033[92m' if pos['pnl_pct'] >= 0 else '\033[91m'
            reset = '\033[0m'
            print(f"   {pos['ticker']:<12s}  "
                  f"Qty: {pos['holdings']:>10.4f}  "
                  f"Value: ${pos['position_value']:>10,.2f}  "
                  f"Entry: ${pos['entry_price']:>8,.2f}  "
                  f"Current: ${pos['current_price']:>8,.2f}  "
                  f"P&L: {pnl_color}{pos['pnl_pct']:>+7.2f}%{reset}")
        print()

    print("=" * 120)
    print()

    # Guidance
    if len(df) < 24:
        print("ℹ️  NOTE: Limited data available (< 24 hours)")
        print(f"   Currently have {len(df)} hour(s) of trading data.")
        print("   Performance metrics become more reliable with longer trading history.")
        print("   Suggested minimum: 24-48 hours for meaningful comparison")
        print()
    elif len(df) < 168:
        print("ℹ️  Growing dataset (< 1 week)")
        print(f"   Currently have {len(df)} hours ({len(df)/24:.1f} days) of trading data.")
        print("   Continue monitoring - weekly performance trends will become clearer.")
        print()
    else:
        print("✓ Good dataset size for performance evaluation")
        print(f"   {len(df)} hours ({len(df)/24:.1f} days) of trading data available.")
        print()

    print("MONITORING COMMANDS:")
    print("  python compare_trading_performance.py         # Run this comparison")
    print("  python show_current_trading_status.py          # Quick status check")
    print("  tail -f logs/paper_trading_BEST_console.log    # Live trading activity")
    print("  python dashboard.py                             # Full dashboard (Page 2, 4, 5)")
    print()

if __name__ == "__main__":
    main()
