#!/usr/bin/env python3
"""
Show current trading status with market comparison.
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def load_positions():
    """Load current positions."""
    pos_file = Path("paper_trades/positions_state.json")
    if pos_file.exists():
        with open(pos_file) as f:
            return json.load(f)
    return None

def load_trading_log():
    """Load paper trading log."""
    log_file = Path("logs/paper_trading_BEST.log")
    if log_file.exists():
        df = pd.read_csv(log_file)
        if len(df) > 0:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
    return None

def calculate_market_performance(df):
    """Calculate buy-and-hold market performance."""
    if df is None or len(df) == 0:
        return None

    # Use BTC as market proxy
    start_price = df.iloc[0]['price_BTC/USD']
    end_price = df.iloc[-1]['price_BTC/USD']
    market_return = ((end_price - start_price) / start_price) * 100

    return {
        'start_price': start_price,
        'end_price': end_price,
        'return_pct': market_return
    }

def extract_trades(df):
    """Extract trades from log."""
    if df is None:
        return []

    trades = []
    tickers = [col.replace('action_', '') for col in df.columns if col.startswith('action_')]

    for idx, row in df.iterrows():
        for ticker in tickers:
            action_col = f'action_{ticker}'
            price_col = f'price_{ticker}'

            if action_col in row and abs(row[action_col]) > 0.0001:
                action_type = 'BUY' if row[action_col] > 0 else 'SELL'
                trades.append({
                    'timestamp': row['timestamp'],
                    'ticker': ticker,
                    'action': action_type,
                    'quantity': abs(row[action_col]),
                    'price': row[price_col],
                    'value': abs(row[action_col]) * row[price_col],
                    'portfolio_value': row['total_asset']
                })

    return trades

def main():
    print("=" * 100)
    print("CURRENT TRADING STATUS - ENSEMBLE_BEST (20 MODELS)")
    print("=" * 100)
    print()

    # Load positions
    positions = load_positions()
    if positions:
        print("CURRENT POSITIONS")
        print("-" * 100)
        print(f"  Portfolio Value:  ${positions['portfolio_value']:,.2f}")
        print(f"  Cash:             ${positions['cash']:,.2f}")
        print(f"  Timestamp:        {positions['timestamp']}")
        print()

        if positions['positions']:
            print("  Active Positions:")
            for pos in positions['positions']:
                pnl_color = '\033[92m' if pos['pnl_pct'] >= 0 else '\033[91m'
                reset = '\033[0m'
                print(f"    {pos['ticker']:<12s} "
                      f"Qty: {pos['holdings']:>10.4f}  "
                      f"Value: ${pos['position_value']:>10,.2f}  "
                      f"P&L: {pnl_color}{pos['pnl_pct']:>+6.2f}%{reset}  "
                      f"Entry: ${pos['entry_price']:.2f}  "
                      f"Current: ${pos['current_price']:.2f}")
        else:
            print("  No active positions (100% cash)")

        print()

        # Initial vs current
        initial = positions['portfolio_protection']['initial_value']
        current = positions['portfolio_value']
        pnl = current - initial
        pnl_pct = (pnl / initial) * 100

        pnl_color = '\033[92m' if pnl >= 0 else '\033[91m'
        reset = '\033[0m'
        print(f"  Portfolio P&L:    {pnl_color}${pnl:+,.2f} ({pnl_pct:+.2f}%){reset}")
        print()

    # Load trading log
    df = load_trading_log()
    if df is not None and len(df) > 0:
        print("TRADING HISTORY")
        print("-" * 100)
        start_time = df.iloc[0]['timestamp']
        end_time = df.iloc[-1]['timestamp']
        duration = end_time - start_time

        print(f"  Session Start:    {start_time}")
        print(f"  Last Update:      {end_time}")
        print(f"  Duration:         {duration}")
        print(f"  Data Points:      {len(df)} (hourly)")
        print()

        # Extract trades
        trades = extract_trades(df)
        if trades:
            print(f"  Total Actions:    {len(trades)}")
            print()
            print("  Recent Trades:")
            print(f"    {'Time':<20s} {'Ticker':<12s} {'Action':<6s} {'Quantity':>12s} {'Price':>12s} {'Value':>12s}")
            print("    " + "-" * 90)
            for trade in trades[-10:]:  # Last 10 trades
                print(f"    {trade['timestamp'].strftime('%Y-%m-%d %H:%M'):<20s} "
                      f"{trade['ticker']:<12s} "
                      f"{trade['action']:<6s} "
                      f"{trade['quantity']:>12.4f} "
                      f"${trade['price']:>11,.2f} "
                      f"${trade['value']:>11,.2f}")
            print()

        # Market comparison
        market = calculate_market_performance(df)
        if market:
            print("MARKET COMPARISON (BTC Buy & Hold)")
            print("-" * 100)
            print(f"  Start Price:      ${market['start_price']:,.2f}")
            print(f"  Current Price:    ${market['end_price']:,.2f}")
            print(f"  Market Return:    {market['return_pct']:+.2f}%")

            if positions:
                agent_return = pnl_pct
                alpha = agent_return - market['return_pct']
                alpha_color = '\033[92m' if alpha >= 0 else '\033[91m'
                reset = '\033[0m'
                print(f"  Agent Return:     {agent_return:+.2f}%")
                print(f"  Alpha:            {alpha_color}{alpha:+.2f}%{reset}")
            print()
    else:
        print("⚠️  No trading history yet")
        print("   The paper trader just started. Trade history will accumulate over time.")
        print("   Trades occur every hour when new market data arrives.")
        print()

    print("=" * 100)
    print()
    print("WHAT TO EXPECT:")
    print("  - Trades occur hourly (1h timeframe)")
    print("  - History accumulates in: logs/paper_trading_BEST.log")
    print("  - Dashboard Page 4 will show trade history as it grows")
    print("  - Dashboard Page 5 will show performance metrics")
    print("  - Run this script anytime: python show_current_trading_status.py")
    print()

if __name__ == "__main__":
    main()
