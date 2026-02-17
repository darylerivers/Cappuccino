#!/usr/bin/env python3
"""
Paper Trading Dashboard - Real-time comparison of ensemble vs single model traders
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
import sys
from typing import Dict, Tuple, Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.live import Live
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️  'rich' library not available. Install with: pip install rich")
    print("   Falling back to simple text output\n")


class PaperTradingDashboard:
    def __init__(self, ensemble_csv: str, single_csv: str):
        self.ensemble_csv = Path(ensemble_csv)
        self.single_csv = Path(single_csv)
        self.console = Console() if RICH_AVAILABLE else None

    def load_data(self, csv_path: Path) -> Optional[pd.DataFrame]:
        """Load trading data from CSV"""
        if not csv_path.exists():
            return None

        try:
            df = pd.read_csv(csv_path)
            if len(df) == 0:
                return None
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
            return None

    def calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate trading metrics from dataframe"""
        if df is None or len(df) == 0:
            return self._empty_metrics()

        # Get asset columns (holding_* and price_* columns)
        holding_cols = [col for col in df.columns if col.startswith('holding_')]
        price_cols = [col for col in df.columns if col.startswith('price_')]
        tickers = [col.replace('holding_', '') for col in holding_cols]

        # Latest row
        latest = df.iloc[-1]

        # Current positions
        positions = {}
        total_position_value = 0
        for ticker in tickers:
            qty = latest[f'holding_{ticker}']
            price = latest[f'price_{ticker}']
            if qty > 0:
                value = qty * price
                positions[ticker] = {
                    'qty': qty,
                    'price': price,
                    'value': value,
                    'pct': 0  # Will calculate after we have total
                }
                total_position_value += value

        # Calculate position percentages
        total_assets = latest['total_asset']
        for ticker in positions:
            positions[ticker]['pct'] = (positions[ticker]['value'] / total_assets) * 100

        # Count trades (when holdings change)
        trades_24h = 0
        trades_7d = 0
        trades_all = 0

        now = datetime.now(df['timestamp'].iloc[-1].tzinfo)
        cutoff_24h = now - timedelta(hours=24)
        cutoff_7d = now - timedelta(days=7)

        for ticker in tickers:
            holding_col = f'holding_{ticker}'
            changes = df[holding_col].diff().abs() > 0.001  # Threshold for float comparison

            trades_all += changes.sum()
            trades_24h += changes[df['timestamp'] >= cutoff_24h].sum()
            trades_7d += changes[df['timestamp'] >= cutoff_7d].sum()

        # Returns
        initial_value = df.iloc[0]['total_asset']
        current_value = latest['total_asset']
        total_return = ((current_value - initial_value) / initial_value) * 100

        # Calculate Sharpe ratio (annualized)
        returns = df['total_asset'].pct_change().dropna()
        if len(returns) > 1:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(365 * 24) if returns.std() > 0 else 0
        else:
            sharpe = 0

        # Max concentration
        max_concentration = max([p['pct'] for p in positions.values()]) if positions else 0

        # Recent activity
        hours_since_update = (now - latest['timestamp']).total_seconds() / 3600

        return {
            'total_asset': current_value,
            'cash': latest['cash'],
            'positions': positions,
            'trades_24h': int(trades_24h),
            'trades_7d': int(trades_7d),
            'trades_all': int(trades_all),
            'total_return': total_return,
            'sharpe': sharpe,
            'max_concentration': max_concentration,
            'num_positions': len(positions),
            'last_update': latest['timestamp'],
            'hours_since_update': hours_since_update,
            'total_rows': len(df)
        }

    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure"""
        return {
            'total_asset': 0,
            'cash': 0,
            'positions': {},
            'trades_24h': 0,
            'trades_7d': 0,
            'trades_all': 0,
            'total_return': 0,
            'sharpe': 0,
            'max_concentration': 0,
            'num_positions': 0,
            'last_update': None,
            'hours_since_update': 999,
            'total_rows': 0
        }

    def render_rich(self, ensemble_metrics: Dict, single_metrics: Dict):
        """Render dashboard using rich library"""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )

        # Header
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = Panel(
            f"[bold cyan]Paper Trading Dashboard[/bold cyan] - {now}",
            style="bold white on blue"
        )
        layout["header"].update(header)

        # Main content - split into two columns
        layout["main"].split_row(
            Layout(name="ensemble"),
            Layout(name="single")
        )

        # Ensemble trader panel
        ensemble_table = self._create_metrics_table("Ensemble (Top-20)", ensemble_metrics)
        layout["ensemble"].update(Panel(ensemble_table, title="[bold green]Ensemble Trader[/bold green]", border_style="green"))

        # Single model panel
        single_table = self._create_metrics_table("Single Model (Trial #861)", single_metrics)
        layout["single"].update(Panel(single_table, title="[bold blue]Single Model Trader[/bold blue]", border_style="blue"))

        # Footer
        footer_text = "Press Ctrl+C to exit | Updates every 60 seconds"
        layout["footer"].update(Panel(footer_text, style="dim"))

        return layout

    def _create_metrics_table(self, name: str, metrics: Dict) -> Table:
        """Create metrics table for a trader"""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="white")

        # Status
        if metrics['hours_since_update'] < 2:
            status = f"[green]✓ Active[/green]"
        elif metrics['hours_since_update'] < 24:
            status = f"[yellow]⚠ Idle {metrics['hours_since_update']:.1f}h[/yellow]"
        else:
            status = f"[red]✗ Stale {metrics['hours_since_update']:.1f}h[/red]"
        table.add_row("Status", status)

        # Performance
        total_return = metrics['total_return']
        return_color = "green" if total_return >= 0 else "red"
        return_arrow = "▲" if total_return >= 0 else "▼"
        table.add_row("Total Return", f"[{return_color}]{return_arrow} {total_return:+.2f}%[/{return_color}]")
        table.add_row("Sharpe Ratio", f"{metrics['sharpe']:.4f}")
        table.add_row("Total Asset", f"${metrics['total_asset']:.2f}")
        table.add_row("Cash", f"${metrics['cash']:.2f}")

        table.add_row("", "")  # Spacer

        # Trading activity
        table.add_row("Trades (24h)", f"{metrics['trades_24h']}")
        table.add_row("Trades (7d)", f"{metrics['trades_7d']}")
        table.add_row("Trades (All)", f"{metrics['trades_all']}")
        table.add_row("Data Points", f"{metrics['total_rows']}")

        table.add_row("", "")  # Spacer

        # Positions
        table.add_row("Active Positions", f"{metrics['num_positions']}")

        # Concentration check
        max_conc = metrics['max_concentration']
        if max_conc > 30:
            conc_str = f"[red]{max_conc:.1f}% ⚠️ OVER LIMIT[/red]"
        elif max_conc > 25:
            conc_str = f"[yellow]{max_conc:.1f}% ⚠️ Near limit[/yellow]"
        else:
            conc_str = f"[green]{max_conc:.1f}%[/green]"
        table.add_row("Max Concentration", conc_str)

        # Position details
        if metrics['positions']:
            table.add_row("", "")  # Spacer
            table.add_row("[bold]Current Positions[/bold]", "")

            # Sort by value
            sorted_positions = sorted(
                metrics['positions'].items(),
                key=lambda x: x[1]['value'],
                reverse=True
            )

            for ticker, pos in sorted_positions:
                ticker_short = ticker.replace('/USD', '')
                pct = pos['pct']
                pct_color = "red" if pct > 30 else "yellow" if pct > 25 else "green"
                table.add_row(
                    f"  {ticker_short}",
                    f"[{pct_color}]{pct:.1f}%[/{pct_color}] (${pos['value']:.2f})"
                )

        return table

    def render_simple(self, ensemble_metrics: Dict, single_metrics: Dict):
        """Render dashboard using simple text (no rich library)"""
        lines = []
        lines.append("=" * 80)
        lines.append(f"PAPER TRADING DASHBOARD - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        lines.append("")

        # Side by side comparison
        lines.append(f"{'ENSEMBLE TRADER':<40} {'SINGLE MODEL TRADER':<40}")
        lines.append(f"{'-' * 40} {'-' * 40}")

        metrics_pairs = [
            ("Status", self._status_text(ensemble_metrics), self._status_text(single_metrics)),
            ("Total Return", f"{ensemble_metrics['total_return']:+.2f}%", f"{single_metrics['total_return']:+.2f}%"),
            ("Sharpe Ratio", f"{ensemble_metrics['sharpe']:.4f}", f"{single_metrics['sharpe']:.4f}"),
            ("Total Asset", f"${ensemble_metrics['total_asset']:.2f}", f"${single_metrics['total_asset']:.2f}"),
            ("Cash", f"${ensemble_metrics['cash']:.2f}", f"${single_metrics['cash']:.2f}"),
            ("", "", ""),
            ("Trades (24h)", f"{ensemble_metrics['trades_24h']}", f"{single_metrics['trades_24h']}"),
            ("Trades (7d)", f"{ensemble_metrics['trades_7d']}", f"{single_metrics['trades_7d']}"),
            ("Trades (All)", f"{ensemble_metrics['trades_all']}", f"{single_metrics['trades_all']}"),
            ("Data Points", f"{ensemble_metrics['total_rows']}", f"{single_metrics['total_rows']}"),
            ("", "", ""),
            ("Active Positions", f"{ensemble_metrics['num_positions']}", f"{single_metrics['num_positions']}"),
            ("Max Concentration", f"{ensemble_metrics['max_concentration']:.1f}%", f"{single_metrics['max_concentration']:.1f}%"),
        ]

        for label, ens_val, single_val in metrics_pairs:
            if label == "":
                lines.append("")
            else:
                lines.append(f"{label:<20} {ens_val:<20} {single_val:<20}")

        # Position details
        lines.append("")
        lines.append(f"{'ENSEMBLE POSITIONS':<40} {'SINGLE MODEL POSITIONS':<40}")
        lines.append(f"{'-' * 40} {'-' * 40}")

        ens_pos = sorted(ensemble_metrics['positions'].items(), key=lambda x: x[1]['value'], reverse=True)
        single_pos = sorted(single_metrics['positions'].items(), key=lambda x: x[1]['value'], reverse=True)

        max_rows = max(len(ens_pos), len(single_pos))
        for i in range(max_rows):
            ens_text = ""
            single_text = ""

            if i < len(ens_pos):
                ticker, pos = ens_pos[i]
                ticker_short = ticker.replace('/USD', '')
                ens_text = f"{ticker_short:<8} {pos['pct']:5.1f}% (${pos['value']:.2f})"

            if i < len(single_pos):
                ticker, pos = single_pos[i]
                ticker_short = ticker.replace('/USD', '')
                single_text = f"{ticker_short:<8} {pos['pct']:5.1f}% (${pos['value']:.2f})"

            lines.append(f"{ens_text:<40} {single_text:<40}")

        lines.append("")
        lines.append("Press Ctrl+C to exit | Updates every 60 seconds")
        lines.append("=" * 80)

        return "\n".join(lines)

    def _status_text(self, metrics: Dict) -> str:
        """Get status text for simple rendering"""
        hours = metrics['hours_since_update']
        if hours < 2:
            return "✓ Active"
        elif hours < 24:
            return f"⚠ Idle {hours:.1f}h"
        else:
            return f"✗ Stale {hours:.1f}h"

    def run(self, refresh_interval: int = 60):
        """Run the dashboard with auto-refresh"""
        if RICH_AVAILABLE:
            self._run_rich(refresh_interval)
        else:
            self._run_simple(refresh_interval)

    def _run_rich(self, refresh_interval: int):
        """Run dashboard with rich library"""
        with Live(self._generate_display(), refresh_per_second=1, screen=True) as live:
            try:
                while True:
                    live.update(self._generate_display())
                    time.sleep(refresh_interval)
            except KeyboardInterrupt:
                pass

    def _run_simple(self, refresh_interval: int):
        """Run dashboard with simple text output"""
        try:
            while True:
                # Clear screen
                print("\033[2J\033[H", end="")

                # Generate and print display
                display = self._generate_display()
                print(display)

                # Wait for next refresh
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            print("\n\nDashboard stopped.")

    def _generate_display(self):
        """Generate the current display"""
        # Load data
        ensemble_df = self.load_data(self.ensemble_csv)
        single_df = self.load_data(self.single_csv)

        # Calculate metrics
        ensemble_metrics = self.calculate_metrics(ensemble_df)
        single_metrics = self.calculate_metrics(single_df)

        # Render
        if RICH_AVAILABLE:
            return self.render_rich(ensemble_metrics, single_metrics)
        else:
            return self.render_simple(ensemble_metrics, single_metrics)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Paper Trading Dashboard")
    parser.add_argument(
        "--ensemble-csv",
        default="paper_trades/watchdog_session_20260116_183945.csv",
        help="Path to ensemble trader CSV file"
    )
    parser.add_argument(
        "--single-csv",
        default="paper_trades/single_model_trial861.csv",
        help="Path to single model trader CSV file"
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=60,
        help="Refresh interval in seconds (default: 60)"
    )

    args = parser.parse_args()

    # Check files exist
    ensemble_path = Path(args.ensemble_csv)
    single_path = Path(args.single_csv)

    if not ensemble_path.exists():
        print(f"⚠️  Ensemble CSV not found: {ensemble_path}")
        print(f"   Looking for most recent watchdog session...")
        # Find most recent watchdog session
        watchdog_files = sorted(Path("paper_trades").glob("watchdog_session_*.csv"))
        if watchdog_files:
            ensemble_path = watchdog_files[-1]
            print(f"   Using: {ensemble_path}")
        else:
            print(f"   No watchdog session files found!")
            return 1

    if not single_path.exists():
        print(f"⚠️  Single model CSV not found: {single_path}")
        print(f"   Will show empty data for single model trader")

    # Create and run dashboard
    dashboard = PaperTradingDashboard(str(ensemble_path), str(single_path))

    print(f"\n🚀 Starting Paper Trading Dashboard")
    print(f"   Ensemble: {ensemble_path.name}")
    print(f"   Single Model: {single_path.name}")
    print(f"   Refresh: {args.refresh}s\n")

    if RICH_AVAILABLE:
        print("   Using rich terminal UI (install 'rich' for best experience)")

    time.sleep(2)  # Give user time to read

    dashboard.run(args.refresh)

    return 0


if __name__ == "__main__":
    sys.exit(main())
