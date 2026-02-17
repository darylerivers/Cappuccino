#!/usr/bin/env python3
"""
Simple Paper Trader Viewer

Shows current portfolio status, recent trades, and performance metrics.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Colors
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def view_paper_trader(csv_path='paper_trades/trial250_session.csv',
                      positions_path='paper_trades/positions_state.json'):

    print(f"\n{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}PAPER TRADER STATUS - TRIAL #250{Colors.END}")
    print(f"{Colors.CYAN}{'='*80}{Colors.END}\n")

    # Load trading history
    try:
        df = pd.read_csv(csv_path)
        print(f"Trading History: {len(df)} bars processed")
        print(f"Time range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        print()
    except FileNotFoundError:
        print(f"{Colors.RED}No trading history found{Colors.END}\n")
        return

    if len(df) == 0:
        print(f"{Colors.YELLOW}No trades executed yet{Colors.END}\n")
        return

    # Current portfolio
    latest = df.iloc[-1]
    initial_capital = 500.0

    current_value = latest['total_asset']
    total_return = (current_value - initial_capital) / initial_capital * 100

    color = Colors.GREEN if total_return >= 0 else Colors.RED

    print(f"{Colors.BOLD}Portfolio Value:{Colors.END}")
    print(f"  Initial:  ${initial_capital:.2f}")
    print(f"  Current:  ${current_value:.2f}")
    print(f"  Return:   {color}{total_return:+.2f}%{Colors.END}")
    print(f"  Cash:     ${latest['cash']:.2f}")
    print()

    # Holdings
    print(f"{Colors.BOLD}Current Positions:{Colors.END}")
    tickers = ['AAVE/USD', 'AVAX/USD', 'BTC/USD', 'LINK/USD', 'ETH/USD', 'LTC/USD', 'UNI/USD']

    has_positions = False
    for ticker in tickers:
        holding_col = f'holding_{ticker}'
        price_col = f'price_{ticker}'

        if holding_col in latest and latest[holding_col] > 0.0001:
            has_positions = True
            holding = latest[holding_col]
            price = latest[price_col]
            value = holding * price

            print(f"  {ticker:12} {holding:10.4f} units @ ${price:10.2f} = ${value:8.2f}")

    if not has_positions:
        print(f"  {Colors.YELLOW}No open positions{Colors.END}")
    print()

    # Performance metrics
    if len(df) > 1:
        returns = df['total_asset'].pct_change().dropna()

        # Sharpe ratio (annualized, assuming hourly data)
        if len(returns) > 0 and returns.std() > 0:
            mean_return = returns.mean()
            std_return = returns.std()
            sharpe = mean_return / std_return * np.sqrt(365 * 24)  # Annualized

            print(f"{Colors.BOLD}Performance Metrics:{Colors.END}")
            print(f"  Trades:       {len(df)}")
            print(f"  Avg Return:   {mean_return*100:.4f}% per bar")
            print(f"  Volatility:   {std_return*100:.4f}% per bar")
            print(f"  Sharpe Ratio: {sharpe:.4f} (live)")
            print(f"  Backtest:     0.1803 (expected)")

            # Max drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min() * 100

            print(f"  Max Drawdown: {max_dd:.2f}%")
            print()

    # Recent actions
    print(f"{Colors.BOLD}Recent Trading Activity (last 5 bars):{Colors.END}")
    recent = df.tail(5)

    for idx, row in recent.iterrows():
        ts = row['timestamp']
        value = row['total_asset']
        reward = row['reward']

        # Count significant actions
        action_cols = [c for c in df.columns if c.startswith('action_')]
        significant_actions = sum(1 for col in action_cols if abs(row[col]) > 0.01)

        print(f"  {ts}: Value=${value:.2f}, Reward={reward:.2f}, Actions={significant_actions}")

    print()

    # Load position state (if available)
    try:
        with open(positions_path, 'r') as f:
            state = json.load(f)

        if state.get('positions'):
            print(f"{Colors.BOLD}Position Details:{Colors.END}")
            for pos in state['positions']:
                ticker = pos['ticker']
                entry = pos['entry_price']
                current = pos['current_price']
                pnl = pos['pnl_pct']

                color = Colors.GREEN if pnl >= 0 else Colors.RED
                print(f"  {ticker:12} Entry=${entry:.2f}, Current=${current:.2f}, P&L={color}{pnl:+.2f}%{Colors.END}")
            print()
    except (FileNotFoundError, KeyError):
        pass

    print(f"{Colors.CYAN}{'='*80}{Colors.END}\n")

if __name__ == '__main__':
    view_paper_trader()
