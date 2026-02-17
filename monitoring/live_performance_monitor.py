#!/usr/bin/env python3
"""
Live Performance Degradation Monitor

Monitors paper trading performance and alerts on model degradation.

Features:
- Tracks live Sharpe ratio vs backtest expectations
- Discord alerts at degradation thresholds
- Automatic trading halt on severe degradation
- Hourly performance summaries
- Rolling window analysis

Thresholds:
- WARNING: Live Sharpe < (Backtest - 0.5)
- CRITICAL: Live Sharpe < -1.0
- EMERGENCY: Live Sharpe < -2.0 for 24 hours → auto-stop

Usage:
    python monitoring/live_performance_monitor.py --paper-trader-csv paper_trades/trial250_session.csv --check-interval 3600
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import DISCORD

# Discord integration
try:
    from integrations.discord_notifier import DiscordNotifier
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False


class LivePerformanceMonitor:
    """Monitor live trading performance and detect model degradation."""

    def __init__(
        self,
        paper_trader_csv: str,
        backtest_sharpe: float = None,
        check_interval: int = 3600,  # 1 hour
        warning_threshold: float = 0.5,  # Alert if live < backtest - 0.5
        critical_threshold: float = -1.0,  # Alert if live < -1.0
        emergency_threshold: float = -2.0,  # Stop trading if < -2.0
        emergency_duration_hours: int = 24,  # Duration before auto-stop
        min_bars_for_calc: int = 10,  # Minimum bars before calculating Sharpe
        check_improvement_trend: bool = True,  # Don't auto-stop if improving
        improvement_threshold: float = 50.0,  # % improvement to consider "improving"
    ):
        self.paper_trader_csv = Path(paper_trader_csv)
        self.backtest_sharpe = backtest_sharpe
        self.check_interval = check_interval
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.emergency_threshold = emergency_threshold
        self.emergency_duration_hours = emergency_duration_hours
        self.min_bars_for_calc = min_bars_for_calc
        self.check_improvement_trend = check_improvement_trend
        self.improvement_threshold = improvement_threshold

        # State
        self.running = True
        self.state_file = Path("monitoring/live_monitor_state.json")
        self.state_file.parent.mkdir(exist_ok=True, parents=True)
        self.state = self._load_state()

        # Discord
        self.discord = None
        if DISCORD_AVAILABLE and DISCORD.ENABLED:
            try:
                self.discord = DiscordNotifier()
                if not self.discord.enabled:
                    self.discord = None
            except Exception as e:
                print(f"⚠️  Discord init failed: {e}")

        # Setup logging
        self._setup_logging()

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Auto-detect backtest Sharpe if not provided
        if self.backtest_sharpe is None:
            self.backtest_sharpe = self._auto_detect_backtest_sharpe()

    def _setup_logging(self):
        """Setup logging."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        log_format = '[%(asctime)s] [%(levelname)s] %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('logs/live_performance_monitor.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def _load_state(self) -> Dict:
        """Load monitor state."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            "last_check_time": None,
            "last_bar_count": 0,
            "last_sharpe": None,
            "warning_sent": False,
            "critical_sent": False,
            "emergency_start_time": None,
            "alert_history": [],
        }

    def _save_state(self):
        """Save monitor state."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def _auto_detect_backtest_sharpe(self) -> float:
        """Try to detect backtest Sharpe from view_paper_trader.py output."""
        try:
            result = subprocess.run(
                ["python", "view_paper_trader.py"],
                capture_output=True,
                text=True,
                timeout=10
            )

            # Parse output for "Backtest: X.XXXX"
            for line in result.stdout.split('\n'):
                if 'Backtest:' in line:
                    parts = line.split('Backtest:')[1].strip().split()
                    if parts:
                        sharpe = float(parts[0])
                        self.logger.info(f"Auto-detected backtest Sharpe: {sharpe:.4f}")
                        return sharpe
        except Exception as e:
            self.logger.warning(f"Could not auto-detect backtest Sharpe: {e}")

        # Default fallback
        return 0.18  # Trial #250's backtest Sharpe

    def calculate_live_sharpe(self) -> Tuple[Optional[float], int, float]:
        """Calculate current live Sharpe ratio from paper trading CSV.

        Returns:
            (sharpe, num_bars, total_return_pct)
        """
        if not self.paper_trader_csv.exists():
            return None, 0, 0.0

        try:
            df = pd.read_csv(self.paper_trader_csv)

            if len(df) < self.min_bars_for_calc:
                return None, len(df), 0.0

            # Calculate returns
            returns = df['total_asset'].pct_change().dropna()

            if len(returns) == 0:
                return None, len(df), 0.0

            # Calculate Sharpe ratio (annualized)
            # Assuming hourly data: 365 days * 24 hours
            periods_per_year = 365 * 24
            sharpe = returns.mean() / returns.std() * np.sqrt(periods_per_year) if returns.std() > 0 else 0.0

            # Total return
            initial_capital = 500.0
            current_capital = df['total_asset'].iloc[-1]
            total_return_pct = (current_capital - initial_capital) / initial_capital * 100

            return sharpe, len(df), total_return_pct

        except Exception as e:
            self.logger.error(f"Error calculating Sharpe: {e}")
            return None, 0, 0.0

    def calculate_improvement_trend(self) -> Optional[float]:
        """Calculate improvement trend (recent Sharpe vs initial Sharpe).

        Returns:
            Improvement percentage (positive = improving)
        """
        if not self.paper_trader_csv.exists():
            return None

        try:
            df = pd.read_csv(self.paper_trader_csv)

            if len(df) < self.min_bars_for_calc * 2:
                return None

            returns = df['total_asset'].pct_change().dropna()

            if len(returns) < self.min_bars_for_calc * 2:
                return None

            # Calculate window size (at least 1/3 of data or 10 bars)
            window_size = min(10, len(returns) // 3)

            # Initial period Sharpe
            initial_returns = returns.iloc[:window_size]
            if initial_returns.std() == 0:
                return None
            sharpe_initial = (initial_returns.mean() / initial_returns.std()) * np.sqrt(365 * 24)

            # Recent period Sharpe
            recent_returns = returns.iloc[-window_size:]
            if recent_returns.std() == 0:
                return None
            sharpe_recent = (recent_returns.mean() / recent_returns.std()) * np.sqrt(365 * 24)

            # Calculate improvement percentage
            if sharpe_initial == 0:
                return None

            improvement_pct = ((sharpe_recent - sharpe_initial) / abs(sharpe_initial)) * 100

            return improvement_pct

        except Exception as e:
            self.logger.error(f"Error calculating improvement trend: {e}")
            return None

    def check_degradation(self, live_sharpe: float, num_bars: int, total_return: float) -> Dict:
        """Check for model degradation and determine alert level.

        Returns:
            {
                'level': 'ok' | 'warning' | 'critical' | 'emergency',
                'message': str,
                'should_stop': bool
            }
        """
        # Calculate gap
        sharpe_gap = live_sharpe - self.backtest_sharpe

        # Check emergency (severe degradation for extended period)
        if live_sharpe < self.emergency_threshold:
            if self.state['emergency_start_time'] is None:
                self.state['emergency_start_time'] = datetime.now(timezone.utc).isoformat()
                self._save_state()
            else:
                # Check duration
                emergency_start = datetime.fromisoformat(self.state['emergency_start_time'])
                duration = (datetime.now(timezone.utc) - emergency_start).total_seconds() / 3600

                if duration >= self.emergency_duration_hours:
                    # Check if model is improving before auto-stopping
                    if self.check_improvement_trend:
                        improvement = self.calculate_improvement_trend()
                        if improvement is not None and improvement > self.improvement_threshold:
                            self.logger.info(f"⚠️  Model below emergency threshold BUT improving by {improvement:.1f}% - NOT auto-stopping")
                            return {
                                'level': 'warning',
                                'message': f"⚠️  Model below threshold BUT improving {improvement:+.1f}% - continuing to monitor",
                                'should_stop': False,
                                'sharpe_gap': sharpe_gap,
                            }

                    # Model not improving sufficiently - auto-stop
                    return {
                        'level': 'emergency',
                        'message': f"🚨 EMERGENCY: Live Sharpe {live_sharpe:.2f} < {self.emergency_threshold} for {duration:.1f}h! Auto-stopping trading.",
                        'should_stop': True,
                        'sharpe_gap': sharpe_gap,
                    }
        else:
            # Reset emergency timer if recovered
            if self.state['emergency_start_time'] is not None:
                self.state['emergency_start_time'] = None
                self._save_state()

        # Check critical
        if live_sharpe < self.critical_threshold:
            return {
                'level': 'critical',
                'message': f"🔴 CRITICAL: Live Sharpe {live_sharpe:.2f} < {self.critical_threshold}! Model severely degraded.",
                'should_stop': False,
                'sharpe_gap': sharpe_gap,
            }

        # Check warning
        if sharpe_gap < -self.warning_threshold:
            return {
                'level': 'warning',
                'message': f"⚠️  WARNING: Live Sharpe {live_sharpe:.2f} vs Backtest {self.backtest_sharpe:.2f} (gap: {sharpe_gap:.2f})",
                'should_stop': False,
                'sharpe_gap': sharpe_gap,
            }

        # All good
        return {
            'level': 'ok',
            'message': f"✅ Performance OK: Live Sharpe {live_sharpe:.2f} vs Backtest {self.backtest_sharpe:.2f}",
            'should_stop': False,
            'sharpe_gap': sharpe_gap,
        }

    def send_alert(self, alert_info: Dict, live_sharpe: float, num_bars: int, total_return: float):
        """Send Discord alert for degradation."""
        if not self.discord:
            return

        level = alert_info['level']

        # Avoid duplicate alerts
        if level == 'warning' and self.state.get('warning_sent'):
            return
        if level == 'critical' and self.state.get('critical_sent'):
            return

        # Determine color
        color_map = {
            'ok': 0x00ff00,      # Green
            'warning': 0xffa500,  # Orange
            'critical': 0xff0000, # Red
            'emergency': 0x8b0000 # Dark red
        }
        color = color_map.get(level, 0x808080)

        # Build embed
        fields = [
            {"name": "📊 Live Sharpe", "value": f"{live_sharpe:.4f}", "inline": True},
            {"name": "📈 Backtest Sharpe", "value": f"{self.backtest_sharpe:.4f}", "inline": True},
            {"name": "📉 Gap", "value": f"{alert_info['sharpe_gap']:+.4f}", "inline": True},
            {"name": "⏱️  Runtime", "value": f"{num_bars} bars", "inline": True},
            {"name": "💰 Return", "value": f"{total_return:+.2f}%", "inline": True},
        ]

        if level == 'emergency':
            fields.append({
                "name": "🚨 Action",
                "value": "STOPPING PAPER TRADER",
                "inline": False
            })

        try:
            self.discord.send_message(
                content=f"**Model Performance Alert: {level.upper()}**",
                embed={
                    "title": "🔍 Live Trading Performance Monitor",
                    "description": alert_info['message'],
                    "color": color,
                    "fields": fields,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "footer": {"text": "Live Performance Monitor"}
                }
            )

            # Mark alert as sent
            if level == 'warning':
                self.state['warning_sent'] = True
            elif level == 'critical':
                self.state['critical_sent'] = True

            self.state['alert_history'].append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'level': level,
                'live_sharpe': live_sharpe,
                'message': alert_info['message']
            })
            self._save_state()

        except Exception as e:
            self.logger.error(f"Failed to send Discord alert: {e}")

    def stop_paper_trader(self):
        """Stop the paper trader process."""
        self.logger.warning("Attempting to stop paper trader...")

        try:
            # Find paper trader process
            result = subprocess.run(
                ["pgrep", "-f", "paper_trader_alpaca_polling.py"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid:
                        self.logger.warning(f"Stopping paper trader (PID: {pid})")
                        subprocess.run(["kill", pid])
                        self.logger.warning("Paper trader stopped")
                        return True
            else:
                self.logger.warning("Paper trader process not found")
                return False

        except Exception as e:
            self.logger.error(f"Failed to stop paper trader: {e}")
            return False

    def send_hourly_summary(self, live_sharpe: float, num_bars: int, total_return: float, alert_info: Dict):
        """Send hourly performance summary to Discord."""
        if not self.discord:
            return

        # Only send summary if no alert was sent (avoid spam)
        if alert_info['level'] != 'ok':
            return

        # Send summary every 6 hours to avoid spam
        last_check = self.state.get('last_check_time')
        if last_check:
            last_check_dt = datetime.fromisoformat(last_check)
            hours_since = (datetime.now(timezone.utc) - last_check_dt).total_seconds() / 3600
            if hours_since < 6:
                return

        try:
            self.discord.send_message(
                content="",
                embed={
                    "title": "📊 Live Trading Summary",
                    "description": f"{num_bars} bars processed",
                    "color": 0x00ff00,
                    "fields": [
                        {"name": "Live Sharpe", "value": f"{live_sharpe:.4f}", "inline": True},
                        {"name": "Backtest", "value": f"{self.backtest_sharpe:.4f}", "inline": True},
                        {"name": "Gap", "value": f"{alert_info['sharpe_gap']:+.4f}", "inline": True},
                        {"name": "Return", "value": f"{total_return:+.2f}%", "inline": True},
                        {"name": "Status", "value": "✅ Normal", "inline": True},
                    ],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "footer": {"text": "6-Hour Summary"}
                }
            )
        except Exception as e:
            self.logger.error(f"Failed to send summary: {e}")

    def run(self):
        """Main monitoring loop."""
        self.logger.info("=" * 80)
        self.logger.info("LIVE PERFORMANCE MONITOR STARTED")
        self.logger.info("=" * 80)
        self.logger.info(f"CSV File: {self.paper_trader_csv}")
        self.logger.info(f"Backtest Sharpe: {self.backtest_sharpe:.4f}")
        self.logger.info(f"Check interval: {self.check_interval}s ({self.check_interval/3600:.1f}h)")
        self.logger.info(f"Thresholds:")
        self.logger.info(f"  Warning: < Backtest - {self.warning_threshold}")
        self.logger.info(f"  Critical: < {self.critical_threshold}")
        self.logger.info(f"  Emergency: < {self.emergency_threshold} for {self.emergency_duration_hours}h")
        self.logger.info("=" * 80)

        # Send startup notification
        if self.discord:
            self.discord.send_message(
                content="🚀 **Live Performance Monitor Started**",
                embed={
                    "title": "Monitoring Paper Trading Performance",
                    "color": 0x00ff00,
                    "fields": [
                        {"name": "Backtest Sharpe", "value": f"{self.backtest_sharpe:.4f}", "inline": True},
                        {"name": "Check Interval", "value": f"{self.check_interval/3600:.1f}h", "inline": True},
                        {"name": "Auto-Stop", "value": f"Sharpe < {self.emergency_threshold} for {self.emergency_duration_hours}h", "inline": False},
                    ],
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )

        while self.running:
            try:
                # Calculate live performance
                live_sharpe, num_bars, total_return = self.calculate_live_sharpe()

                if live_sharpe is not None:
                    # Check for degradation
                    alert_info = self.check_degradation(live_sharpe, num_bars, total_return)

                    # Log status
                    self.logger.info(f"[{datetime.now(timezone.utc).isoformat()}] "
                                   f"Bars: {num_bars}, Live Sharpe: {live_sharpe:.4f}, "
                                   f"Gap: {alert_info['sharpe_gap']:+.4f}, Level: {alert_info['level']}")

                    # Send alert if needed
                    if alert_info['level'] in ['warning', 'critical', 'emergency']:
                        self.send_alert(alert_info, live_sharpe, num_bars, total_return)
                    else:
                        # Send hourly summary
                        self.send_hourly_summary(live_sharpe, num_bars, total_return, alert_info)

                    # Stop trading if emergency
                    if alert_info['should_stop']:
                        self.logger.error("EMERGENCY THRESHOLD REACHED - STOPPING PAPER TRADER")
                        self.stop_paper_trader()
                        self.running = False
                        break

                    # Update state
                    self.state['last_check_time'] = datetime.now(timezone.utc).isoformat()
                    self.state['last_bar_count'] = num_bars
                    self.state['last_sharpe'] = live_sharpe
                    self._save_state()

                else:
                    self.logger.info(f"Not enough data yet ({num_bars} bars, need {self.min_bars_for_calc})")

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}", exc_info=True)

            # Sleep until next check
            time.sleep(self.check_interval)

        self.logger.info("Live performance monitor stopped")


def parse_args():
    parser = argparse.ArgumentParser(description="Live trading performance monitor")
    parser.add_argument('--paper-trader-csv', type=str,
                       default='paper_trades/trial250_session.csv',
                       help='Path to paper trader CSV file')
    parser.add_argument('--backtest-sharpe', type=float, default=None,
                       help='Expected backtest Sharpe (auto-detected if not provided)')
    parser.add_argument('--check-interval', type=int, default=3600,
                       help='Check interval in seconds (default: 3600 = 1 hour)')
    parser.add_argument('--warning-threshold', type=float, default=0.5,
                       help='Alert if live < backtest - threshold')
    parser.add_argument('--critical-threshold', type=float, default=-1.0,
                       help='Critical alert threshold')
    parser.add_argument('--emergency-threshold', type=float, default=-2.0,
                       help='Emergency auto-stop threshold')
    parser.add_argument('--emergency-duration', type=int, default=24,
                       help='Hours at emergency threshold before auto-stop')
    parser.add_argument('--min-bars', type=int, default=10,
                       help='Minimum bars before calculating Sharpe')
    parser.add_argument('--check-improvement', action='store_true', default=True,
                       help='Check improvement trend before auto-stopping (default: True)')
    parser.add_argument('--no-check-improvement', action='store_false', dest='check_improvement',
                       help='Disable improvement trend check')
    parser.add_argument('--improvement-threshold', type=float, default=50.0,
                       help='Improvement %% to consider model improving (default: 50%%)')

    return parser.parse_args()


def main():
    args = parse_args()

    monitor = LivePerformanceMonitor(
        paper_trader_csv=args.paper_trader_csv,
        backtest_sharpe=args.backtest_sharpe,
        check_interval=args.check_interval,
        warning_threshold=args.warning_threshold,
        critical_threshold=args.critical_threshold,
        emergency_threshold=args.emergency_threshold,
        emergency_duration_hours=args.emergency_duration,
        min_bars_for_calc=args.min_bars,
        check_improvement_trend=args.check_improvement,
        improvement_threshold=args.improvement_threshold,
    )

    monitor.run()


if __name__ == '__main__':
    main()
