#!/usr/bin/env python3
"""Enhanced Dashboard with Optimization Metrics

Shows:
- DRL vs Baseline comparison
- Test results (pass/fail)
- Profit protection events
- Risk thresholds (from constants.py)
- Paper trading performance
- System health

Usage:
    python dashboard_optimized.py
    python dashboard_optimized.py --refresh 10  # Update every 10 seconds
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

try:
    import pandas as pd
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "pandas"], check=True)
    import pandas as pd

# Import constants
try:
    from constants import RISK, TRADING
    CONSTANTS_AVAILABLE = True
except ImportError:
    CONSTANTS_AVAILABLE = False


class OptimizedDashboard:
    """Enhanced dashboard showing optimization metrics."""

    def __init__(self):
        self.refresh_interval = 10
        self.width = 100

    def clear_screen(self):
        """Clear terminal."""
        print("\033[2J\033[H", end="")

    def render(self):
        """Render full dashboard."""
        self.clear_screen()

        # Header
        print("=" * self.width)
        print("CAPPUCCINO OPTIMIZED DASHBOARD".center(self.width))
        print(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(self.width))
        print("=" * self.width)
        print()

        # Section 1: Test Results
        self._render_test_results()
        print()

        # Section 2: Baseline Comparison
        self._render_baseline_comparison()
        print()

        # Section 3: Paper Trading Performance
        self._render_paper_trading()
        print()

        # Section 4: Open Positions & Stop-Loss Levels
        self._render_positions_table()
        print()

        # Section 5: Profit Protection
        self._render_profit_protection()
        print()

        # Section 6: Risk Configuration
        self._render_risk_config()
        print()

        # Section 7: System Status
        self._render_system_status()
        print()

        print("=" * self.width)
        print("Press Ctrl+C to exit".center(self.width))
        print("=" * self.width)

    def _render_test_results(self):
        """Show test suite status."""
        print("📊 TEST SUITE STATUS")
        print("-" * self.width)

        # Check if tests exist
        test_file = Path("tests/test_critical.py")
        if not test_file.exists():
            print("  ⚠️  Test file not found: tests/test_critical.py")
            return

        # Run pytest to get results (quick check)
        try:
            result = subprocess.run(
                ["pytest", "tests/test_critical.py", "-v", "--tb=no", "--quiet"],
                capture_output=True,
                text=True,
                timeout=30
            )

            # Parse output
            output = result.stdout + result.stderr

            if "passed" in output.lower():
                # Extract test counts
                import re
                match = re.search(r'(\d+) passed', output)
                if match:
                    passed = int(match.group(1))
                    print(f"  ✅ All tests passed: {passed}/25")
                else:
                    print(f"  ✅ Tests passed")
            elif "failed" in output.lower():
                match_passed = re.search(r'(\d+) passed', output)
                match_failed = re.search(r'(\d+) failed', output)
                passed = int(match_passed.group(1)) if match_passed else 0
                failed = int(match_failed.group(1)) if match_failed else 0
                print(f"  ❌ Tests failed: {passed} passed, {failed} failed")
                print(f"     Run: pytest tests/test_critical.py -v")
            else:
                print(f"  ⚠️  Tests not run yet")
                print(f"     Run: pytest tests/test_critical.py -v")

        except FileNotFoundError:
            print("  ⚠️  pytest not installed. Run: pip install pytest")
        except subprocess.TimeoutExpired:
            print("  ⏱️  Tests timed out (still running?)")
        except Exception as e:
            print(f"  ⚠️  Could not run tests: {e}")

    def _render_baseline_comparison(self):
        """Show DRL vs baseline performance."""
        print("📈 PERFORMANCE COMPARISON")
        print("-" * self.width)

        # Check for baseline results
        baseline_file = Path("baselines/results_buy_and_hold.json")

        if not baseline_file.exists():
            print("  ⚠️  Baseline not run yet")
            print("     Run: python baselines/buy_and_hold.py --data data/price_array_val.npy")
            return

        try:
            with baseline_file.open() as f:
                baseline = json.load(f)

            metrics = baseline.get('metrics', {})

            print(f"  Buy-and-Hold Baseline:")
            print(f"    Total Return:    {metrics.get('total_return_pct', 0):>10.2f}%")
            print(f"    Sharpe Ratio:    {metrics.get('sharpe_ratio', 0):>10.3f}")
            print(f"    Max Drawdown:    {metrics.get('max_drawdown_pct', 0):>10.2f}%")
            print(f"    Win Rate:        {metrics.get('win_rate_pct', 0):>10.2f}%")

            # Check for DRL results (from paper trading)
            paper_csv = Path("paper_trades/alpaca_session.csv")
            if paper_csv.exists():
                df = pd.read_csv(paper_csv)
                if len(df) > 1:
                    drl_return = (df['total_asset'].iloc[-1] / df['total_asset'].iloc[0] - 1) * 100
                    print()
                    print(f"  DRL Paper Trading:")
                    print(f"    Total Return:    {drl_return:>10.2f}%")

                    # Calculate Sharpe (simple)
                    returns = df['total_asset'].pct_change().dropna()
                    if len(returns) > 1:
                        sharpe = returns.mean() / returns.std() * (24 * 365) ** 0.5
                        print(f"    Sharpe Ratio:    {sharpe:>10.3f}")

                    # Comparison
                    print()
                    baseline_sharpe = metrics.get('sharpe_ratio', 0)
                    if sharpe > baseline_sharpe:
                        diff = ((sharpe - baseline_sharpe) / baseline_sharpe * 100) if baseline_sharpe > 0 else 0
                        print(f"  🎉 DRL outperforms baseline by {diff:.1f}%")
                    else:
                        diff = ((baseline_sharpe - sharpe) / baseline_sharpe * 100) if baseline_sharpe > 0 else 0
                        print(f"  ⚠️  Baseline outperforms DRL by {diff:.1f}%")

        except Exception as e:
            print(f"  ⚠️  Error loading baseline: {e}")

    def _render_paper_trading(self):
        """Show paper trading status."""
        print("💰 PAPER TRADING PERFORMANCE")
        print("-" * self.width)

        csv_file = Path("paper_trades/alpaca_session.csv")
        if not csv_file.exists():
            print("  ⚠️  No paper trading data yet")
            return

        try:
            df = pd.read_csv(csv_file)

            if len(df) == 0:
                print("  ⚠️  No trades recorded yet")
                return

            latest = df.iloc[-1]
            initial = df.iloc[0]

            # Performance metrics
            total_return = (latest['total_asset'] / initial['total_asset'] - 1) * 100

            print(f"  Initial Capital:   ${initial['total_asset']:>10.2f}")
            print(f"  Current Value:     ${latest['total_asset']:>10.2f}")
            print(f"  Total Return:      {total_return:>10.2f}%")
            print(f"  Cash Balance:      ${latest['cash']:>10.2f}")
            print(f"  Trades Recorded:   {len(df):>10d}")

            # Latest timestamp
            print(f"  Last Update:       {latest['timestamp']}")

        except Exception as e:
            print(f"  ⚠️  Error reading paper trades: {e}")

    def _render_positions_table(self):
        """Show detailed positions with stop-loss levels."""
        print("📊 OPEN POSITIONS & STOP-LOSS LEVELS")
        print("-" * self.width)

        positions_file = Path("paper_trades/positions_state.json")
        if not positions_file.exists():
            print("  ℹ️  No position state file yet (will be created on first trade)")
            return

        try:
            with positions_file.open() as f:
                state = json.load(f)

            positions = state.get("positions", [])

            if len(positions) == 0:
                print("  ℹ️  No open positions (100% cash)")
                return

            # Table header
            print(f"  {'Asset':<8} {'Qty':>8} {'Entry':>10} {'Current':>10} {'Value':>10} {'P&L':>8} {'Stop-Loss':>10} {'Distance':>10}")
            print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*10}")

            # Sort by position value (largest first)
            positions.sort(key=lambda p: p['position_value'], reverse=True)

            for pos in positions:
                ticker = pos['ticker'].split('/')[0]  # BTC/USD -> BTC
                qty = pos['holdings']
                entry = pos['entry_price']
                current = pos['current_price']
                value = pos['position_value']
                pnl = pos['pnl_pct']
                stop = pos['stop_loss_price']
                distance = pos['distance_to_stop_pct']

                # Color P&L
                pnl_str = f"{pnl:>7.2f}%"
                if pnl > 0:
                    pnl_str = f"+{pnl:.2f}%"

                # Color distance to stop-loss
                if distance < 2:
                    distance_str = f"⚠️ {distance:.1f}%"
                elif distance < 5:
                    distance_str = f"{distance:.1f}%"
                else:
                    distance_str = f"{distance:.1f}%"

                print(f"  {ticker:<8} {qty:>8.2f} ${entry:>9.2f} ${current:>9.2f} ${value:>9.2f} {pnl_str:>8} ${stop:>9.2f} {distance_str:>10}")

            # Show portfolio protection status
            print()
            portfolio_prot = state.get("portfolio_protection", {})
            initial_value = portfolio_prot.get("initial_value")
            hwm = portfolio_prot.get("high_water_mark", 0)

            if initial_value:
                current_value = state.get("portfolio_value", 0)
                gain = ((current_value / initial_value - 1) * 100) if initial_value > 0 else 0
                drawdown = ((hwm - current_value) / hwm * 100) if hwm > 0 else 0

                print(f"  Portfolio:")
                print(f"    Gain from start:     {gain:>8.2f}%")
                print(f"    High water mark:     ${hwm:>10.2f}")
                print(f"    Drawdown from peak:  {drawdown:>8.2f}%")

                if portfolio_prot.get("in_cash_mode"):
                    print(f"    Status:              🛑 CASH MODE (cooldown active)")
                elif portfolio_prot.get("profit_taken"):
                    print(f"    Status:              💰 Partial profits taken")

        except Exception as e:
            print(f"  ⚠️  Error reading positions: {e}")

    def _render_profit_protection(self):
        """Show profit protection events."""
        print("🛡️  PROFIT PROTECTION EVENTS")
        print("-" * self.width)

        log_file = Path("paper_trades/profit_protection.log")

        if not log_file.exists():
            print("  ℹ️  No profit protection events yet (log file not created)")
            print("     Events will appear when protection triggers")
            return

        try:
            with log_file.open() as f:
                lines = f.readlines()

            if len(lines) == 0:
                print("  ℹ️  No events logged yet")
                return

            # Show last 5 events
            recent = lines[-5:]
            print(f"  Recent events (last 5 of {len(lines)}):")
            for line in recent:
                # Parse timestamp and message
                if "]" in line:
                    parts = line.split("]", 1)
                    if len(parts) == 2:
                        timestamp = parts[0].replace("[", "")
                        message = parts[1].strip()

                        # Add emoji based on event type
                        if "STOP" in message.upper():
                            emoji = "🛑"
                        elif "PROFIT" in message.upper():
                            emoji = "💰"
                        elif "CASH" in message.upper():
                            emoji = "💵"
                        else:
                            emoji = "📊"

                        print(f"    {emoji} {timestamp}: {message[:70]}")

        except Exception as e:
            print(f"  ⚠️  Error reading log: {e}")

    def _render_risk_config(self):
        """Show current risk configuration."""
        print("⚙️  RISK CONFIGURATION")
        print("-" * self.width)

        if not CONSTANTS_AVAILABLE:
            print("  ⚠️  constants.py not available")
            return

        print("  Per-Position Limits:")
        print(f"    Max Position:      {RISK.MAX_POSITION_PCT * 100:>6.1f}%")
        print(f"    Stop-Loss:         {RISK.STOP_LOSS_PCT * 100:>6.1f}%")
        print(f"    Trailing Stop:     {RISK.TRAILING_STOP_PCT * 100:>6.1f}% {'(disabled)' if RISK.TRAILING_STOP_PCT == 0 else ''}")

        print()
        print("  Portfolio Profit Protection:")
        print(f"    Trailing Stop:     {RISK.PORTFOLIO_TRAILING_STOP_PCT * 100:>6.2f}%")
        print(f"    Profit Take:       {RISK.PROFIT_TAKE_THRESHOLD_PCT * 100:>6.1f}% (sell {RISK.PROFIT_TAKE_AMOUNT_PCT * 100:.0f}%)")
        print(f"    Move-to-Cash:      {RISK.MOVE_TO_CASH_THRESHOLD_PCT * 100:>6.1f}% {'(disabled)' if RISK.MOVE_TO_CASH_THRESHOLD_PCT == 0 else ''}")

        print()
        print("  Trading:")
        print(f"    Transaction Cost:  {TRADING.BUY_COST_PCT * 100:>6.2f}%")
        print(f"    Initial Capital:   ${TRADING.INITIAL_CAPITAL:>8.0f}")

    def _render_system_status(self):
        """Show system process status."""
        print("🖥️  SYSTEM STATUS")
        print("-" * self.width)

        # Check processes
        processes = {
            "Paper Trader": "paper_trader_alpaca_polling",
            "Watchdog": "system_watchdog",
            "Auto-Deployer": "auto_model_deployer",
            "Performance Monitor": "performance_monitor"
        }

        for name, pattern in processes.items():
            try:
                result = subprocess.run(
                    ["pgrep", "-f", pattern],
                    capture_output=True,
                    text=True
                )

                if result.returncode == 0 and result.stdout.strip():
                    pid = result.stdout.strip().split()[0]
                    print(f"  ✅ {name:<20} Running (PID: {pid})")
                else:
                    print(f"  ❌ {name:<20} Not running")
            except Exception:
                print(f"  ⚠️  {name:<20} Status unknown")

    def run(self, once: bool = False):
        """Run dashboard loop."""
        try:
            while True:
                self.render()

                if once:
                    break

                import time
                time.sleep(self.refresh_interval)

        except KeyboardInterrupt:
            print("\n\nDashboard stopped.")


def main():
    parser = argparse.ArgumentParser(description="Optimized Cappuccino Dashboard")
    parser.add_argument("--refresh", type=int, default=10, help="Refresh interval (seconds)")
    parser.add_argument("--once", action="store_true", help="Run once (no refresh)")
    args = parser.parse_args()

    dashboard = OptimizedDashboard()
    dashboard.refresh_interval = args.refresh
    dashboard.run(once=args.once)


if __name__ == "__main__":
    main()
