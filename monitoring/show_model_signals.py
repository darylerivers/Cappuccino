#!/usr/bin/env python3
"""
Show current model signals from paper trading logs
"""
import sys
import re
import csv
from pathlib import Path
from datetime import datetime, timezone

def parse_latest_action(log_file: Path):
    """Parse the most recent action from log file."""
    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        return None

    # Read last 50 lines to find most recent action
    with open(log_file, 'r') as f:
        lines = f.readlines()

    # Search backwards for action line
    for line in reversed(lines[-50:]):
        match = re.search(r'\[([\d\-T:+]+)\].*cash=([\d.]+).*total=([\d.]+).*reward=([\-\d.]+).*actions=\[(.*?)\]', line)
        if match:
            timestamp = match.group(1)
            cash = float(match.group(2))
            total = float(match.group(3))
            reward = float(match.group(4))
            actions_str = match.group(5)
            actions = [float(x.strip()) for x in actions_str.split(',')]

            return {
                'timestamp': timestamp,
                'cash': cash,
                'total': total,
                'reward': reward,
                'actions': actions
            }

    print("No action data found in log file")
    return None


def interpret_action(action_value: float) -> tuple[str, str]:
    """Interpret action value into signal and strength."""
    if abs(action_value) < 1.0:
        strength = "HOLD"
        color = "\033[90m"  # Gray
    elif abs(action_value) < 50:
        strength = "WEAK"
        color = "\033[93m" if action_value > 0 else "\033[94m"  # Yellow/Blue
    elif abs(action_value) < 100:
        strength = "MODERATE"
        color = "\033[92m" if action_value > 0 else "\033[96m"  # Green/Cyan
    else:
        strength = "STRONG"
        color = "\033[91m" if action_value < 0 else "\033[92m"  # Red/Green (bright)

    if action_value > 1.0:
        signal = "BUY"
    elif action_value < -1.0:
        signal = "SELL"
    else:
        signal = "HOLD"

    return signal, strength, color


def get_current_holdings(csv_file: Path, tickers: list):
    """Get current holdings from CSV file."""
    if not csv_file.exists():
        return None

    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                return None

            last_row = rows[-1]
            holdings = {}
            prices = {}

            for ticker in tickers:
                hold_key = f"holding_{ticker}"
                price_key = f"price_{ticker}"
                if hold_key in last_row and price_key in last_row:
                    holdings[ticker] = float(last_row[hold_key])
                    prices[ticker] = float(last_row[price_key])

            return {'holdings': holdings, 'prices': prices}
    except Exception as e:
        print(f"Warning: Could not read CSV: {e}")
        return None


def display_signals(data: dict, tickers: list, holdings_data: dict = None):
    """Display model signals in readable format."""
    print("\n" + "=" * 80)
    print("🤖 MODEL SIGNALS - CURRENT HOUR")
    print("=" * 80)
    print()

    print(f"📅 Timestamp:    {data['timestamp']}")
    print(f"💵 Cash:         ${data['cash']:.2f}")
    print(f"💰 Total Assets: ${data['total']:.2f}")
    print(f"📊 Reward:       {data['reward']:.6f}")
    print()

    print("─" * 80)
    if holdings_data:
        print(f"{'TICKER':<12} {'HOLDING':<15} {'PRICE':<15} {'SIGNAL':<8} {'ACTION':<12}")
    else:
        print(f"{'TICKER':<12} {'SIGNAL':<8} {'STRENGTH':<12} {'RAW ACTION':<15} {'INTERPRETATION'}")
    print("─" * 80)

    actions = data['actions']

    for ticker, action in zip(tickers, actions):
        signal, strength, color = interpret_action(action)
        end_color = "\033[0m"

        if holdings_data:
            holding = holdings_data['holdings'].get(ticker, 0.0)
            price = holdings_data['prices'].get(ticker, 0.0)
            value = holding * price

            # Show holding and signal
            if holding > 0.0001:
                hold_str = f"{holding:.4f} (${value:.2f})"
            else:
                hold_str = "0"

            print(f"{ticker:<12} {hold_str:<15} ${price:>13,.2f} {color}{signal:<8}{end_color} {action:>+11.2f}")
        else:
            # Interpretation
            if signal == "BUY":
                interp = f"Model wants to BUY {ticker.split('/')[0]}"
            elif signal == "SELL":
                interp = f"Model wants to SELL {ticker.split('/')[0]}"
            else:
                interp = f"Model holding position on {ticker.split('/')[0]}"

            print(f"{ticker:<12} {color}{signal:<8}{end_color} {strength:<12} {action:>+14.2f}  {interp}")

    print("─" * 80)
    print()
    print("📝 NOTES:")
    print("  • Actions are continuous values from the neural network")
    print("  • Positive = Buy signal | Negative = Sell signal")
    print("  • Magnitude indicates strength of conviction")
    print("  • These are model suggestions - actual trades depend on portfolio state")
    print()
    print("=" * 80)


def main():
    # Default tickers
    tickers = ["AAVE/USD", "AVAX/USD", "BTC/USD", "LINK/USD", "ETH/USD", "LTC/USD", "UNI/USD"]

    # Find most recent log file
    log_candidates = [
        Path("logs/paper_trading_fixed.log"),
        Path("logs/paper_trading_corrected.log"),
        Path("logs/paper_trading_live.log"),
    ]

    log_file = None
    for candidate in log_candidates:
        if candidate.exists():
            log_file = candidate
            break

    if log_file is None:
        print("No paper trading log file found!")
        print("Expected locations:", log_candidates)
        sys.exit(1)

    print(f"Reading from: {log_file}")

    data = parse_latest_action(log_file)

    if data is None:
        sys.exit(1)

    if len(data['actions']) != len(tickers):
        print(f"Warning: Expected {len(tickers)} actions, got {len(data['actions'])}")

    # Try to get current holdings from CSV
    csv_candidates = [
        Path("paper_trades/fixed_session_20251118_153830.csv"),
        Path("paper_trades/alpaca_session.csv"),
    ]

    holdings_data = None
    for csv_file in csv_candidates:
        holdings_data = get_current_holdings(csv_file, tickers)
        if holdings_data:
            break

    display_signals(data, tickers, holdings_data)


if __name__ == "__main__":
    main()
