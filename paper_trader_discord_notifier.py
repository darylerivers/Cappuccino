#!/usr/bin/env python3
"""
Paper Trader Discord Notifier

Sends paper trading updates to Discord with performance metrics.
Supports threaded messages for each trader.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime
import time
import argparse
from integrations.discord_notifier import DiscordNotifier

class PaperTraderDiscordNotifier:
    def __init__(self, webhook_url=None):
        self.notifier = DiscordNotifier(webhook_url)

    def calculate_metrics(self, csv_path):
        """Calculate performance metrics from CSV file"""
        try:
            df = pd.read_csv(csv_path)

            if len(df) <= 1:
                return None

            returns = df['total_asset'].pct_change().dropna()

            if len(returns) == 0 or returns.std() == 0:
                return None

            # Overall Sharpe
            sharpe_overall = (returns.mean() / returns.std()) * np.sqrt(365 * 24)

            # Expanding window Sharpe (average)
            expanding_sharpes = []
            for i in range(3, len(returns) + 1):
                window_returns = returns.iloc[:i]
                if window_returns.std() > 0:
                    window_sharpe = (window_returns.mean() / window_returns.std()) * np.sqrt(365 * 24)
                    expanding_sharpes.append(window_sharpe)
            sharpe_average = np.mean(expanding_sharpes) if expanding_sharpes else None

            # First and last periods
            window_size = min(10, len(returns) // 3)

            initial_returns = returns.iloc[:window_size]
            sharpe_initial = (initial_returns.mean() / initial_returns.std()) * np.sqrt(365 * 24) if initial_returns.std() > 0 else None

            recent_returns = returns.iloc[-window_size:]
            sharpe_recent = (recent_returns.mean() / recent_returns.std()) * np.sqrt(365 * 24) if recent_returns.std() > 0 else None

            # Last bar impact
            sharpe_previous = None
            if len(returns) >= 4:
                prev_returns = returns.iloc[:-1]
                if prev_returns.std() > 0:
                    sharpe_previous = (prev_returns.mean() / prev_returns.std()) * np.sqrt(365 * 24)

            # Portfolio
            initial_value = df['total_asset'].iloc[0]
            current_value = df['total_asset'].iloc[-1]
            cash = df['cash'].iloc[-1]
            total_return = ((current_value - initial_value) / initial_value) * 100

            # Positions
            latest = df.iloc[-1]
            tickers = ['AAVE/USD', 'AVAX/USD', 'BTC/USD', 'LINK/USD', 'ETH/USD', 'LTC/USD', 'UNI/USD']
            positions = {}
            for ticker in tickers:
                holding_col = f'holding_{ticker}'
                price_col = f'price_{ticker}'
                if holding_col in latest and latest[holding_col] > 0.0001:
                    positions[ticker] = {
                        'quantity': latest[holding_col],
                        'price': latest[price_col],
                        'value': latest[holding_col] * latest[price_col]
                    }

            return {
                'bars': len(df),
                'initial_value': initial_value,
                'current_value': current_value,
                'cash': cash,
                'total_return': total_return,
                'sharpe_overall': sharpe_overall,
                'sharpe_average': sharpe_average,
                'sharpe_initial': sharpe_initial,
                'sharpe_recent': sharpe_recent,
                'sharpe_previous': sharpe_previous,
                'sharpe_change': sharpe_overall - sharpe_previous if sharpe_previous else None,
                'positions': positions,
                'window_size': window_size,
                'timestamp': latest['timestamp']
            }
        except Exception as e:
            print(f"Error calculating metrics for {csv_path}: {e}")
            return None

    def format_trader_update(self, trial_name, metrics):
        """Format trader metrics as Discord embed"""
        if not metrics:
            return None

        # Color based on performance
        if metrics['sharpe_recent'] and metrics['sharpe_recent'] > 0:
            color = 0x00ff00  # Green - positive recent Sharpe
        elif metrics['total_return'] >= 0:
            color = 0xffaa00  # Orange - break even
        else:
            color = 0xff0000  # Red - negative

        # Build fields
        fields = [
            {
                "name": "📊 Portfolio",
                "value": f"**Value:** ${metrics['current_value']:.2f}\n"
                         f"**Return:** {metrics['total_return']:+.2f}%\n"
                         f"**Cash:** ${metrics['cash']:.2f}",
                "inline": True
            },
            {
                "name": "📈 Performance",
                "value": f"**Bars:** {metrics['bars']}\n"
                         f"**Sharpe:** {metrics['sharpe_overall']:.4f}\n"
                         f"**Avg Sharpe:** {metrics['sharpe_average']:.4f}" if metrics['sharpe_average'] else "N/A",
                "inline": True
            }
        ]

        # Add Sharpe evolution
        if metrics['sharpe_initial'] is not None and metrics['sharpe_recent'] is not None:
            improvement = metrics['sharpe_recent'] - metrics['sharpe_initial']
            improvement_pct = (improvement / abs(metrics['sharpe_initial']) * 100) if metrics['sharpe_initial'] != 0 else 0
            trend_emoji = "🟢" if improvement > 0 else "🔴"

            fields.append({
                "name": "📊 Trend Analysis",
                "value": f"**First {metrics['window_size']} bars:** {metrics['sharpe_initial']:.2f}\n"
                         f"**Recent {metrics['window_size']} bars:** {metrics['sharpe_recent']:.2f}\n"
                         f"**Improvement:** {trend_emoji} {improvement:+.2f} ({improvement_pct:+.1f}%)",
                "inline": False
            })

        # Add last bar impact
        if metrics['sharpe_change'] is not None:
            bar_emoji = "🟢" if metrics['sharpe_change'] > 0 else "🔴"
            fields.append({
                "name": "⏱️ Last Bar Impact",
                "value": f"{bar_emoji} {metrics['sharpe_change']:+.4f}\n"
                         f"Previous: {metrics['sharpe_previous']:.4f}",
                "inline": True
            })

        # Add positions
        if metrics['positions']:
            position_text = ""
            for symbol, data in metrics['positions'].items():
                ticker_short = symbol.split('/')[0]
                position_text += f"**{ticker_short}:** {data['quantity']:.4f} @ ${data['price']:.2f} = ${data['value']:.2f}\n"
            fields.append({
                "name": "💼 Positions",
                "value": position_text.strip(),
                "inline": False
            })
        else:
            fields.append({
                "name": "💼 Positions",
                "value": "100% Cash (no open positions)",
                "inline": False
            })

        embed = {
            "title": f"🤖 {trial_name} - Paper Trading Update",
            "color": color,
            "fields": fields,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": f"Last update: {metrics['timestamp']}"}
        }

        return embed

    def send_trader_update(self, trial_name, csv_path, thread_id=None):
        """Send update for a single trader"""
        metrics = self.calculate_metrics(csv_path)
        if not metrics:
            print(f"No metrics available for {trial_name}")
            return False

        embed = self.format_trader_update(trial_name, metrics)
        if not embed:
            return False

        # TODO: Add thread_id support when Discord API allows it
        # For now, just send to main channel
        return self.notifier.send_message("", embed=embed)

    def send_all_traders(self):
        """Send updates for all active paper traders"""
        traders = []

        # Find all session CSV files
        csv_files = sorted(Path('paper_trades').glob('trial*_session.csv'),
                          key=lambda x: x.stat().st_mtime, reverse=True)

        for csv in csv_files:
            # Extract trial number from filename
            trial_name = csv.stem.replace('_session', '').replace('trial', 'Trial #')

            # Check if file is recent (modified in last 24 hours)
            age_hours = (time.time() - csv.stat().st_mtime) / 3600
            if age_hours < 24:
                traders.append((trial_name, str(csv)))

        if not traders:
            self.notifier.send_alert("No active paper traders found", level="warning")
            return False

        # Send header
        self.notifier.send_message(f"📊 **Paper Trader Status Update** - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Send each trader
        for trial_name, csv_path in traders:
            self.send_trader_update(trial_name, csv_path)
            time.sleep(1)  # Rate limit

        return True


def main():
    parser = argparse.ArgumentParser(description='Send paper trader updates to Discord')
    parser.add_argument('--trial', type=str, help='Specific trial to update (e.g., "trial100")')
    parser.add_argument('--all', action='store_true', help='Send updates for all traders')
    parser.add_argument('--webhook', type=str, help='Discord webhook URL (or use DISCORD_WEBHOOK_URL env var)')
    parser.add_argument('--watch', action='store_true', help='Watch mode - send updates every hour')
    parser.add_argument('--interval', type=int, default=3600, help='Watch interval in seconds (default: 3600)')

    args = parser.parse_args()

    notifier = PaperTraderDiscordNotifier(args.webhook)

    if not notifier.notifier.enabled:
        print("❌ Discord webhook not configured!")
        print("Set DISCORD_WEBHOOK_URL environment variable or use --webhook")
        sys.exit(1)

    if args.watch:
        print(f"👀 Watching paper traders (interval: {args.interval}s)...")
        while True:
            try:
                notifier.send_all_traders()
                print(f"✅ Update sent at {datetime.now().strftime('%H:%M:%S')}")
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\n👋 Stopping watch mode")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(60)  # Wait a minute before retrying

    elif args.all:
        notifier.send_all_traders()

    elif args.trial:
        csv_path = f"paper_trades/{args.trial}_session.csv"
        if not os.path.exists(csv_path):
            print(f"❌ CSV file not found: {csv_path}")
            sys.exit(1)

        trial_name = args.trial.replace('trial', 'Trial #')
        notifier.send_trader_update(trial_name, csv_path)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
