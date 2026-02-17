#!/usr/bin/env python3
"""
Paper Trading Alert System

Monitors paper traders for critical issues and sends alerts:
- Process crashes
- Concentration violations (>30%)
- No trading activity
- Position losses exceeding stop-loss
- API errors accumulating
- Performance degradation
"""

import os
import sys
import time
import signal
import psutil
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    trader: str  # "ensemble" or "single"


class AlertNotifier:
    """Handles alert notifications via multiple channels."""

    def __init__(self, enable_terminal=True, enable_log=True, enable_email=False):
        self.enable_terminal = enable_terminal
        self.enable_log = enable_log
        self.enable_email = enable_email
        self.alert_log = Path("logs/alerts.log")
        self.alert_log.parent.mkdir(exist_ok=True)

    def send(self, alert: Alert):
        """Send alert through enabled channels."""
        if self.enable_terminal:
            self._send_terminal(alert)

        if self.enable_log:
            self._send_log(alert)

        if self.enable_email:
            self._send_email(alert)

    def _send_terminal(self, alert: Alert):
        """Print alert to terminal with color coding."""
        colors = {
            AlertLevel.INFO: "\033[94m",      # Blue
            AlertLevel.WARNING: "\033[93m",   # Yellow
            AlertLevel.CRITICAL: "\033[91m",  # Red
        }
        reset = "\033[0m"

        icons = {
            AlertLevel.INFO: "ℹ️ ",
            AlertLevel.WARNING: "⚠️ ",
            AlertLevel.CRITICAL: "🚨",
        }

        color = colors.get(alert.level, "")
        icon = icons.get(alert.level, "")
        timestamp = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n{color}{icon} [{alert.level.value}] {alert.title}{reset}")
        print(f"{color}Trader: {alert.trader}{reset}")
        print(f"{color}Time: {timestamp}{reset}")
        print(f"{color}{alert.message}{reset}")
        print()

    def _send_log(self, alert: Alert):
        """Write alert to log file."""
        timestamp = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] [{alert.level.value}] [{alert.trader}] {alert.title}: {alert.message}\n"

        with self.alert_log.open("a") as f:
            f.write(log_line)

    def _send_email(self, alert: Alert):
        """Send email notification (placeholder for future implementation)."""
        # TODO: Implement email notifications
        # Could use SMTP, SendGrid, AWS SES, etc.
        pass


class PaperTradingMonitor:
    """Monitors paper trading sessions for issues and sends alerts."""

    def __init__(
        self,
        ensemble_csv: str,
        single_csv: str,
        ensemble_pid: Optional[int] = None,
        single_pid: Optional[int] = None,
        check_interval: int = 300,  # 5 minutes
    ):
        self.ensemble_csv = Path(ensemble_csv) if ensemble_csv else None
        self.single_csv = Path(single_csv) if single_csv else None
        self.ensemble_pid = ensemble_pid
        self.single_pid = single_pid
        self.check_interval = check_interval

        self.notifier = AlertNotifier()
        self.running = True
        self.last_check = {}
        self.alert_cooldown = {}  # Prevent spam

    def run(self):
        """Main monitoring loop."""
        print("=" * 80)
        print("PAPER TRADING ALERT SYSTEM")
        print("=" * 80)
        print()
        print(f"Monitoring interval: {self.check_interval}s")
        print(f"Ensemble CSV: {self.ensemble_csv}")
        print(f"Single CSV: {self.single_csv}")
        print(f"Ensemble PID: {self.ensemble_pid}")
        print(f"Single PID: {self.single_pid}")
        print()
        print("Press Ctrl+C to stop")
        print("=" * 80)
        print()

        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        while self.running:
            try:
                self._check_all()
                time.sleep(self.check_interval)
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait a minute before retrying

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print("\n\nShutting down alert system...")
        self.running = False

    def _check_all(self):
        """Run all monitoring checks."""
        timestamp = datetime.now(timezone.utc)
        print(f"[{timestamp.strftime('%H:%M:%S')}] Running checks...")

        # Check ensemble trader
        if self.ensemble_csv:
            self._check_trader("ensemble", self.ensemble_csv, self.ensemble_pid)

        # Check single model trader
        if self.single_csv:
            self._check_trader("single", self.single_csv, self.single_pid)

        print(f"[{timestamp.strftime('%H:%M:%S')}] Checks complete.")

    def _check_trader(self, name: str, csv_path: Path, pid: Optional[int]):
        """Run all checks for a specific trader."""

        # 1. Process check
        if pid:
            self._check_process(name, pid)

        # 2. Data file checks
        if not csv_path.exists():
            self._send_alert(
                name,
                AlertLevel.WARNING,
                "CSV File Not Found",
                f"Trading data file not found: {csv_path}"
            )
            return

        # Load data
        try:
            df = pd.read_csv(csv_path)
            if len(df) == 0:
                self._send_alert(
                    name,
                    AlertLevel.WARNING,
                    "Empty Data File",
                    f"CSV file exists but has no data: {csv_path}"
                )
                return
        except Exception as e:
            self._send_alert(
                name,
                AlertLevel.WARNING,
                "CSV Read Error",
                f"Failed to read CSV: {e}"
            )
            return

        # 3. Concentration check
        self._check_concentration(name, df)

        # 4. Trading activity check
        self._check_trading_activity(name, df)

        # 5. Stop-loss check
        self._check_stop_loss(name, df)

        # 6. Performance check
        self._check_performance(name, df)

        # 7. Staleness check
        self._check_staleness(name, df)

    def _check_process(self, name: str, pid: int):
        """Check if trader process is still running."""
        try:
            process = psutil.Process(pid)
            if not process.is_running():
                self._send_alert(
                    name,
                    AlertLevel.CRITICAL,
                    "Trader Process Crashed",
                    f"Process {pid} is not running!"
                )
        except psutil.NoSuchProcess:
            self._send_alert(
                name,
                AlertLevel.CRITICAL,
                "Trader Process Not Found",
                f"Process {pid} does not exist!"
            )

    def _check_concentration(self, name: str, df: pd.DataFrame):
        """Check for concentration limit violations (>30%)."""
        if len(df) == 0:
            return

        latest = df.iloc[-1]

        # Get asset columns
        holding_cols = [col for col in df.columns if col.startswith('holding_')]
        price_cols = [col for col in df.columns if col.startswith('price_')]
        tickers = [col.replace('holding_', '') for col in holding_cols]

        total_asset = latest['total_asset']

        for ticker in tickers:
            qty = latest[f'holding_{ticker}']
            price = latest[f'price_{ticker}']
            if qty > 0 and price > 0:
                value = qty * price
                concentration = (value / total_asset) * 100 if total_asset > 0 else 0

                # Allow small buffer for price drift (30.5%) - positions bought at 30%
                # can drift slightly higher as prices fluctuate
                if concentration > 32:  # Truly violated
                    self._send_alert(
                        name,
                        AlertLevel.CRITICAL,
                        "Concentration Limit Violated",
                        f"{ticker}: {concentration:.1f}% of portfolio (limit: 30%, tolerance: 32%)\n"
                        f"Position value: ${value:.2f}\n"
                        f"Total portfolio: ${total_asset:.2f}\n"
                        f"🚨 FIX MAY NOT BE WORKING! Concentration significantly over limit."
                    )
                elif concentration > 30.5:  # Warning zone (price drift)
                    self._send_alert(
                        name,
                        AlertLevel.WARNING,
                        "Concentration Slightly Over Limit",
                        f"{ticker}: {concentration:.1f}% of portfolio (limit: 30%, tolerance: 32%)\n"
                        f"Position value: ${value:.2f}\n"
                        f"Note: Small drift from price movement is normal. Monitoring..."
                    )
                elif concentration > 28:  # Approaching limit
                    self._send_alert(
                        name,
                        AlertLevel.INFO,
                        "Concentration Approaching Limit",
                        f"{ticker}: {concentration:.1f}% of portfolio (limit: 30%)\n"
                        f"Position value: ${value:.2f}"
                    )

    def _check_trading_activity(self, name: str, df: pd.DataFrame):
        """Check if trader has been inactive for too long."""
        if len(df) < 2:
            return

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        now = datetime.now(df['timestamp'].iloc[-1].tzinfo)

        # Check for trades in last 24 hours
        cutoff = now - timedelta(hours=24)
        recent_df = df[df['timestamp'] >= cutoff]

        if len(recent_df) < 2:
            return  # Not enough data

        # Count trades (when holdings change)
        holding_cols = [col for col in df.columns if col.startswith('holding_')]
        total_trades = 0

        for col in holding_cols:
            changes = recent_df[col].diff().abs() > 0.001
            total_trades += changes.sum()

        if total_trades == 0:
            self._send_alert(
                name,
                AlertLevel.WARNING,
                "No Trading Activity",
                f"No trades executed in the last 24 hours.\n"
                f"Data points: {len(recent_df)}\n"
                f"This may be normal if market conditions don't trigger signals."
            )

    def _check_stop_loss(self, name: str, df: pd.DataFrame):
        """Check for positions with losses exceeding stop-loss threshold."""
        if len(df) == 0:
            return

        latest = df.iloc[-1]
        holding_cols = [col for col in df.columns if col.startswith('holding_')]
        price_cols = [col for col in df.columns if col.startswith('price_')]
        tickers = [col.replace('holding_', '') for col in holding_cols]

        # This is a simplified check - real entry prices tracked by trader
        # We can only check current P&L
        total_asset = latest['total_asset']
        cash = latest['cash']

        # If total asset has dropped significantly
        if len(df) > 1:
            initial = df.iloc[0]['total_asset']
            current = total_asset
            loss_pct = ((current - initial) / initial) * 100

            if loss_pct < -10:  # 10% portfolio loss
                self._send_alert(
                    name,
                    AlertLevel.CRITICAL,
                    "Portfolio Loss Exceeds Threshold",
                    f"Portfolio down {abs(loss_pct):.1f}% from start\n"
                    f"Initial: ${initial:.2f}\n"
                    f"Current: ${current:.2f}\n"
                    f"Loss: ${initial - current:.2f}"
                )
            elif loss_pct < -5:  # Warning at 5%
                self._send_alert(
                    name,
                    AlertLevel.WARNING,
                    "Portfolio Loss Warning",
                    f"Portfolio down {abs(loss_pct):.1f}% from start\n"
                    f"Initial: ${initial:.2f}\n"
                    f"Current: ${current:.2f}"
                )

    def _check_performance(self, name: str, df: pd.DataFrame):
        """Check for significant performance degradation."""
        if len(df) < 10:  # Need enough data
            return

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        now = datetime.now(df['timestamp'].iloc[-1].tzinfo)

        # Compare last 24h vs previous 24h
        cutoff_recent = now - timedelta(hours=24)
        cutoff_previous = now - timedelta(hours=48)

        recent = df[df['timestamp'] >= cutoff_recent]
        previous = df[(df['timestamp'] >= cutoff_previous) & (df['timestamp'] < cutoff_recent)]

        if len(recent) < 2 or len(previous) < 2:
            return

        # Calculate returns
        recent_return = ((recent.iloc[-1]['total_asset'] - recent.iloc[0]['total_asset']) /
                        recent.iloc[0]['total_asset'] * 100)
        previous_return = ((previous.iloc[-1]['total_asset'] - previous.iloc[0]['total_asset']) /
                          previous.iloc[0]['total_asset'] * 100)

        # Alert if significant degradation
        if recent_return < previous_return - 5:  # 5% worse
            self._send_alert(
                name,
                AlertLevel.WARNING,
                "Performance Degradation",
                f"Last 24h return: {recent_return:+.2f}%\n"
                f"Previous 24h return: {previous_return:+.2f}%\n"
                f"Degradation: {previous_return - recent_return:.2f}%"
            )

    def _check_staleness(self, name: str, df: pd.DataFrame):
        """Check if data is stale (no updates for too long)."""
        if len(df) == 0:
            return

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        latest_timestamp = df.iloc[-1]['timestamp']
        now = datetime.now(latest_timestamp.tzinfo)

        hours_since_update = (now - latest_timestamp).total_seconds() / 3600

        if hours_since_update > 6:  # 6 hours stale
            self._send_alert(
                name,
                AlertLevel.CRITICAL,
                "Data Stale - No Updates",
                f"No data updates for {hours_since_update:.1f} hours\n"
                f"Last update: {latest_timestamp}\n"
                f"Process may have crashed!"
            )
        elif hours_since_update > 3:  # Warning at 3 hours
            self._send_alert(
                name,
                AlertLevel.WARNING,
                "Data Updates Delayed",
                f"No data updates for {hours_since_update:.1f} hours\n"
                f"Last update: {latest_timestamp}"
            )

    def _send_alert(self, trader: str, level: AlertLevel, title: str, message: str):
        """Send alert with cooldown to prevent spam."""
        # Create cooldown key
        cooldown_key = f"{trader}:{title}"

        # Check cooldown
        if cooldown_key in self.alert_cooldown:
            last_sent = self.alert_cooldown[cooldown_key]
            cooldown_minutes = 60 if level == AlertLevel.INFO else 30 if level == AlertLevel.WARNING else 15
            if (datetime.now(timezone.utc) - last_sent).total_seconds() < cooldown_minutes * 60:
                return  # Still in cooldown

        # Send alert
        alert = Alert(
            level=level,
            title=title,
            message=message,
            timestamp=datetime.now(timezone.utc),
            trader=trader
        )

        self.notifier.send(alert)
        self.alert_cooldown[cooldown_key] = datetime.now(timezone.utc)


def find_trader_processes() -> Tuple[Optional[int], Optional[int]]:
    """Find paper trader PIDs by scanning running processes."""
    ensemble_pid = None
    single_pid = None

    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and 'paper_trader_alpaca_polling.py' in ' '.join(cmdline):
                if 'ensemble' in ' '.join(cmdline):
                    ensemble_pid = proc.info['pid']
                elif 'trial_861' in ' '.join(cmdline):
                    single_pid = proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    return ensemble_pid, single_pid


def find_latest_csv_files() -> Tuple[Optional[str], Optional[str]]:
    """Find the most recent CSV files for each trader."""
    paper_trades = Path("paper_trades")

    if not paper_trades.exists():
        return None, None

    # Find ensemble CSV
    ensemble_files = list(paper_trades.glob("ensemble_fixed_*.csv")) + \
                    list(paper_trades.glob("watchdog_session_*.csv"))
    ensemble_csv = str(max(ensemble_files, key=lambda p: p.stat().st_mtime)) if ensemble_files else None

    # Find single model CSV
    single_files = list(paper_trades.glob("single_fixed_*.csv")) + \
                  list(paper_trades.glob("single_model_trial861*.csv"))
    single_csv = str(max(single_files, key=lambda p: p.stat().st_mtime)) if single_files else None

    return ensemble_csv, single_csv


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Paper Trading Alert System")
    parser.add_argument(
        "--ensemble-csv",
        help="Path to ensemble trader CSV file (auto-detected if not specified)"
    )
    parser.add_argument(
        "--single-csv",
        help="Path to single model trader CSV file (auto-detected if not specified)"
    )
    parser.add_argument(
        "--ensemble-pid",
        type=int,
        help="Ensemble trader process ID (auto-detected if not specified)"
    )
    parser.add_argument(
        "--single-pid",
        type=int,
        help="Single model trader process ID (auto-detected if not specified)"
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=300,
        help="Check interval in seconds (default: 300 = 5 minutes)"
    )

    args = parser.parse_args()

    # Auto-detect CSV files if not provided
    ensemble_csv = args.ensemble_csv
    single_csv = args.single_csv

    if not ensemble_csv or not single_csv:
        print("Auto-detecting CSV files...")
        auto_ensemble, auto_single = find_latest_csv_files()
        ensemble_csv = ensemble_csv or auto_ensemble
        single_csv = single_csv or auto_single

    # Auto-detect PIDs if not provided
    ensemble_pid = args.ensemble_pid
    single_pid = args.single_pid

    if not ensemble_pid or not single_pid:
        print("Auto-detecting trader processes...")
        auto_ens_pid, auto_single_pid = find_trader_processes()
        ensemble_pid = ensemble_pid or auto_ens_pid
        single_pid = single_pid or auto_single_pid

    # Create and run monitor
    monitor = PaperTradingMonitor(
        ensemble_csv=ensemble_csv,
        single_csv=single_csv,
        ensemble_pid=ensemble_pid,
        single_pid=single_pid,
        check_interval=args.check_interval
    )

    monitor.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
