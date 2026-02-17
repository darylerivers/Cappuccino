#!/usr/bin/env python3
"""
Multi-Trader Viewer

Shows all active paper traders with their current status.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import subprocess

# Colors
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def get_active_traders():
    """Find all running paper trader processes."""
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )

        traders = []
        for line in result.stdout.split('\n'):
            if 'paper_trader_alpaca_polling.py' in line and 'grep' not in line:
                # Extract trial number from --log-file argument
                if '--log-file' in line:
                    parts = line.split('--log-file')
                    if len(parts) > 1:
                        csv_path = parts[1].strip().split()[0]
                        traders.append(csv_path)

        return traders
    except Exception as e:
        print(f"Error finding traders: {e}")
        return []

def view_trader(csv_path):
    """Display status for a single trader."""

    # Extract trial number
    trial_num = "Unknown"
    if 'trial' in csv_path:
        import re
        match = re.search(r'trial(\d+)', csv_path)
        if match:
            trial_num = match.group(1)

    print(f"{Colors.CYAN}{'─'*80}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}TRIAL #{trial_num}{Colors.END}")
    print(f"{Colors.CYAN}{'─'*80}{Colors.END}")

    # Load trading history
    try:
        df = pd.read_csv(csv_path)

        if len(df) == 0:
            print(f"{Colors.YELLOW}⏳ Waiting for first trade...{Colors.END}\n")
            return

        print(f"📊 Trading History: {len(df)} bars processed")
        print(f"⏱  Time range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
        print()

    except FileNotFoundError:
        print(f"{Colors.RED}❌ CSV file not found: {csv_path}{Colors.END}\n")
        return
    except pd.errors.EmptyDataError:
        print(f"{Colors.YELLOW}⏳ Waiting for first trade...{Colors.END}\n")
        return

    # Current portfolio
    latest = df.iloc[-1]
    initial_capital = 500.0

    current_value = latest['total_asset']
    total_return = (current_value - initial_capital) / initial_capital * 100

    color = Colors.GREEN if total_return >= 0 else Colors.RED

    print(f"{Colors.BOLD}💰 Portfolio:{Colors.END}")
    print(f"   Initial:  ${initial_capital:.2f}")
    print(f"   Current:  ${current_value:.2f}")
    print(f"   Return:   {color}{total_return:+.2f}%{Colors.END}")
    print(f"   Cash:     ${latest['cash']:.2f}")
    print()

    # Holdings
    tickers = ['AAVE/USD', 'AVAX/USD', 'BTC/USD', 'LINK/USD', 'ETH/USD', 'LTC/USD', 'UNI/USD']

    has_positions = False
    for ticker in tickers:
        holding_col = f'holding_{ticker}'
        price_col = f'price_{ticker}'

        if holding_col in latest and latest[holding_col] > 0.0001:
            if not has_positions:
                print(f"{Colors.BOLD}📈 Positions:{Colors.END}")
                has_positions = True

            holding = latest[holding_col]
            price = latest[price_col]
            value = holding * price

            print(f"   {ticker:12} {holding:10.4f} × ${price:10.2f} = ${value:8.2f}")

    if not has_positions:
        print(f"{Colors.YELLOW}📊 No open positions (100% cash){Colors.END}")
    print()

    # Performance metrics
    if len(df) > 1:
        returns = df['total_asset'].pct_change().dropna()

        if len(returns) > 0 and returns.std() > 0:
            mean_return = returns.mean()
            std_return = returns.std()
            sharpe_overall = mean_return / std_return * np.sqrt(365 * 24)  # Annualized

            # Max drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min() * 100

            # Calculate expanding window Sharpe (average Sharpe over time)
            expanding_sharpes = []
            min_window = 3  # Minimum bars needed for Sharpe calculation

            for i in range(min_window, len(returns) + 1):
                window_returns = returns.iloc[:i]
                if window_returns.std() > 0:
                    window_sharpe = (window_returns.mean() / window_returns.std()) * np.sqrt(365 * 24)
                    expanding_sharpes.append(window_sharpe)

            sharpe_average = np.mean(expanding_sharpes) if expanding_sharpes else None

            # Calculate Sharpe from previous bar (up to N-1) vs current (up to N)
            sharpe_previous = None
            sharpe_change_last_bar = None

            if len(returns) >= min_window + 1:
                # Sharpe up to previous bar
                prev_returns = returns.iloc[:-1]
                if prev_returns.std() > 0:
                    sharpe_previous = (prev_returns.mean() / prev_returns.std()) * np.sqrt(365 * 24)
                    sharpe_change_last_bar = sharpe_overall - sharpe_previous

            # Calculate Sharpe for different periods
            window_size = min(10, len(returns) // 3)  # Use 10 bars or 1/3 of data

            sharpe_initial = None
            sharpe_recent = None

            if len(returns) >= window_size * 2:
                # Initial period Sharpe
                initial_returns = returns.iloc[:window_size]
                if initial_returns.std() > 0:
                    sharpe_initial = (initial_returns.mean() / initial_returns.std()) * np.sqrt(365 * 24)

                # Recent period Sharpe
                recent_returns = returns.iloc[-window_size:]
                if recent_returns.std() > 0:
                    sharpe_recent = (recent_returns.mean() / recent_returns.std()) * np.sqrt(365 * 24)

            print(f"{Colors.BOLD}📊 Performance:{Colors.END}")
            print(f"   Bars:         {len(df)}")
            print(f"   Sharpe (Current): {sharpe_overall:.4f} (all data)")

            if sharpe_average is not None:
                avg_color = Colors.GREEN if sharpe_average > 0 else Colors.RED
                print(f"   Sharpe (Average): {avg_color}{sharpe_average:.4f}{Colors.END} (cumulative avg)")

            # Show last bar impact
            if sharpe_previous is not None and sharpe_change_last_bar is not None:
                change_color = Colors.GREEN if sharpe_change_last_bar > 0 else Colors.RED
                change_symbol = "↑" if sharpe_change_last_bar > 0 else "↓"
                print(f"   └─ Last bar impact: {change_color}{change_symbol} {sharpe_change_last_bar:+.4f}{Colors.END} (prev: {sharpe_previous:.4f})")

            # Show Sharpe evolution if we have enough data
            if sharpe_initial is not None and sharpe_recent is not None:
                init_color = Colors.GREEN if sharpe_initial > 0 else Colors.RED
                recent_color = Colors.GREEN if sharpe_recent > 0 else Colors.RED

                print(f"   └─ First {window_size} bars: {init_color}{sharpe_initial:>7.4f}{Colors.END}")
                print(f"   └─ Last {window_size} bars:  {recent_color}{sharpe_recent:>7.4f}{Colors.END}")

                # Show trend
                if sharpe_recent > sharpe_initial:
                    trend_pct = ((sharpe_recent - sharpe_initial) / abs(sharpe_initial) * 100) if sharpe_initial != 0 else 0
                    print(f"   └─ Trend: {Colors.GREEN}↑ Improving{Colors.END} ({sharpe_recent - sharpe_initial:+.4f}, {trend_pct:+.1f}%)")
                else:
                    trend_pct = ((sharpe_recent - sharpe_initial) / abs(sharpe_initial) * 100) if sharpe_initial != 0 else 0
                    print(f"   └─ Trend: {Colors.RED}↓ Declining{Colors.END} ({sharpe_recent - sharpe_initial:+.4f}, {trend_pct:+.1f}%)")

            print(f"   Max Drawdown: {max_dd:.2f}%")
            print(f"   Volatility:   {std_return*100:.4f}% per bar")
            print()

def main():
    """Main display loop."""

    # Header
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}ACTIVE PAPER TRADERS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*80}{Colors.END}\n")

    # Find active traders
    traders = get_active_traders()

    if not traders:
        print(f"{Colors.YELLOW}⚠️  No active paper traders found{Colors.END}\n")

        # Check for recent CSV files anyway
        csv_files = sorted(Path('paper_trades').glob('trial*_session.csv'), key=lambda x: x.stat().st_mtime, reverse=True)

        if csv_files:
            print(f"{Colors.YELLOW}Found recent session files (may be inactive):{Colors.END}\n")
            for csv in csv_files[:3]:
                view_trader(str(csv))

        return

    print(f"{Colors.GREEN}✓ Found {len(traders)} active trader(s){Colors.END}\n")

    # Display each trader
    for csv_path in sorted(traders):
        view_trader(csv_path)

    print(f"{Colors.CYAN}{Colors.BOLD}{'='*80}{Colors.END}\n")
    print(f"{Colors.CYAN}💡 Tip: Run with 'watch -n 60 python view_all_traders.py' for auto-refresh{Colors.END}\n")

if __name__ == '__main__':
    main()
