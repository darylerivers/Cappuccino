#!/usr/bin/env python3
"""
Performance Monitor & Alerts
Tracks system performance and generates alerts for key events.

Monitors:
- New best trials found
- Paper trading activity (trades executed, P&L)
- Training progress
- GPU utilization trends
- Database growth

Features:
- Desktop notifications
- Performance reports
- Trend analysis
- Anomaly detection

Usage:
    python performance_monitor.py --check-interval 300
"""

import argparse
import csv
import json
import logging
import os
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class PerformanceMonitor:
    def __init__(
        self,
        study_name: str = "cappuccino_3workers_20251102_2325",
        db_path: str = "databases/optuna_cappuccino.db",
        check_interval: int = 300,
        enable_notifications: bool = True,
    ):
        self.study_name = study_name
        self.db_path = db_path
        self.check_interval = check_interval
        self.enable_notifications = enable_notifications

        # State
        self.running = True
        self.state_file = Path("deployments/monitor_state.json")
        self.state_file.parent.mkdir(exist_ok=True)
        self.state = self._load_state()

        # Setup logging
        self._setup_logging()

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_logging(self):
        """Setup logging configuration."""
        log_format = '[%(asctime)s] [%(levelname)s] %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('logs/performance_monitor.log'),
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
            "last_trial_count": 0,
            "best_trial_id": None,
            "best_trial_value": None,
            "performance_history": [],
        }

    def _save_state(self):
        """Save monitor state."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def notify(self, title: str, message: str, urgency: str = "normal"):
        """Send desktop notification."""
        if not self.enable_notifications:
            return

        try:
            subprocess.run([
                "notify-send",
                "-u", urgency,
                "-a", "Cappuccino",
                title,
                message
            ], check=False, timeout=5)
        except Exception as e:
            self.logger.warning(f"Failed to send notification: {e}")

    def get_trial_stats(self) -> Dict:
        """Get trial statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get total trials
        query = """
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN state = 'COMPLETE' THEN 1 ELSE 0 END) as complete,
                   SUM(CASE WHEN state = 'RUNNING' THEN 1 ELSE 0 END) as running,
                   SUM(CASE WHEN state = 'FAIL' THEN 1 ELSE 0 END) as failed
            FROM trials t
            JOIN studies s ON t.study_id = s.study_id
            WHERE s.study_name = ?
        """
        cursor.execute(query, (self.study_name,))
        total, complete, running, failed = cursor.fetchone()

        # Get best trial
        query = """
            SELECT t.trial_id, t.number, tv.value, t.datetime_complete
            FROM trials t
            JOIN studies s ON t.study_id = s.study_id
            JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE s.study_name = ? AND t.state = 'COMPLETE'
            ORDER BY tv.value ASC
            LIMIT 1
        """
        cursor.execute(query, (self.study_name,))
        result = cursor.fetchone()

        best_trial = None
        if result:
            best_trial = {
                "trial_id": result[0],
                "number": result[1],
                "value": result[2],
                "datetime": result[3],
            }

        # Get recent completions (last hour)
        query = """
            SELECT COUNT(*)
            FROM trials t
            JOIN studies s ON t.study_id = s.study_id
            WHERE s.study_name = ?
            AND t.state = 'COMPLETE'
            AND t.datetime_complete > datetime('now', '-1 hour')
        """
        cursor.execute(query, (self.study_name,))
        recent_completions = cursor.fetchone()[0]

        conn.close()

        return {
            "total": total or 0,
            "complete": complete or 0,
            "running": running or 0,
            "failed": failed or 0,
            "best_trial": best_trial,
            "recent_completions": recent_completions,
        }

    def check_new_best_trial(self, stats: Dict):
        """Check for new best trial."""
        best_trial = stats.get("best_trial")

        if not best_trial:
            return

        current_best_id = best_trial["trial_id"]
        current_best_value = best_trial["value"]

        last_best_id = self.state.get("best_trial_id")
        last_best_value = self.state.get("best_trial_value")

        if last_best_id != current_best_id:
            # New best trial found!
            improvement = ""
            if last_best_value is not None:
                diff = last_best_value - current_best_value
                pct = abs(diff / last_best_value) * 100
                improvement = f" ({pct:.2f}% improvement)"

            message = f"Trial #{best_trial['number']} (ID: {current_best_id})\n" \
                     f"Value: {current_best_value:.6f}{improvement}"

            self.logger.info(f"🎯 NEW BEST TRIAL: {message}")
            self.notify("🎯 New Best Trial!", message, urgency="normal")

            # Update state
            self.state["best_trial_id"] = current_best_id
            self.state["best_trial_value"] = current_best_value
            self._save_state()

    def get_gpu_stats(self) -> Optional[Dict]:
        """Get GPU statistics."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode != 0:
                return None

            temp, util, mem_used, mem_total, power = result.stdout.strip().split(',')

            return {
                "temperature": int(temp.strip()),
                "utilization": int(util.strip()),
                "memory_used": int(mem_used.strip()),
                "memory_total": int(mem_total.strip()),
                "power_draw": float(power.strip()),
            }

        except Exception as e:
            self.logger.warning(f"Failed to get GPU stats: {e}")
            return None

    def check_paper_trading(self) -> Dict:
        """Check paper trading activity."""
        # Find latest paper trading CSV
        csv_files = list(Path("paper_trades").glob("*.csv"))

        if not csv_files:
            return {"status": "no_data"}

        latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)

        try:
            with open(latest_csv, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows:
                return {"status": "no_data"}

            # Get latest row
            latest = rows[-1]

            # Count trades - use 'total_asset' column (not 'portfolio_value')
            trade_count = sum(1 for row in rows if float(row.get('total_asset', 0)) != 1000.0)

            return {
                "status": "active",
                "csv_file": str(latest_csv),
                "total_rows": len(rows),
                "trade_count": trade_count,
                "latest_portfolio_value": float(latest.get('total_asset', 0)),  # Fixed: use 'total_asset'
                "latest_timestamp": latest.get('timestamp', ''),
            }

        except Exception as e:
            self.logger.warning(f"Failed to read paper trading CSV: {e}")
            return {"status": "error", "error": str(e)}

    def generate_performance_report(self, stats: Dict, gpu_stats: Optional[Dict], paper_trading: Dict) -> str:
        """Generate performance report."""
        lines = []
        lines.append("=" * 80)
        lines.append("PERFORMANCE REPORT")
        lines.append("=" * 80)
        lines.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Training stats
        lines.append("TRAINING PROGRESS")
        lines.append("-" * 80)
        lines.append(f"Total Trials:     {stats['total']}")
        lines.append(f"  Complete:       {stats['complete']}")
        lines.append(f"  Running:        {stats['running']}")
        lines.append(f"  Failed:         {stats['failed']}")
        lines.append(f"Recent (1h):      {stats['recent_completions']}")

        if stats['best_trial']:
            bt = stats['best_trial']
            lines.append(f"")
            lines.append(f"Best Trial:       #{bt['number']} (ID: {bt['trial_id']})")
            lines.append(f"  Value:          {bt['value']:.6f}")
            lines.append(f"  Completed:      {bt['datetime']}")

        lines.append("")

        # GPU stats
        if gpu_stats:
            lines.append("GPU STATUS")
            lines.append("-" * 80)
            lines.append(f"Temperature:      {gpu_stats['temperature']}°C")
            lines.append(f"Utilization:      {gpu_stats['utilization']}%")
            mem_pct = (gpu_stats['memory_used'] / gpu_stats['memory_total']) * 100
            lines.append(f"Memory:           {gpu_stats['memory_used']}/{gpu_stats['memory_total']} MB ({mem_pct:.1f}%)")
            lines.append(f"Power Draw:       {gpu_stats['power_draw']}W")
            lines.append("")

        # Paper trading
        lines.append("PAPER TRADING")
        lines.append("-" * 80)
        if paper_trading["status"] == "active":
            lines.append(f"Status:           Active")
            lines.append(f"CSV File:         {paper_trading['csv_file']}")
            lines.append(f"Data Points:      {paper_trading['total_rows']}")
            lines.append(f"Trades Executed:  {paper_trading['trade_count']}")
            lines.append(f"Portfolio Value:  ${paper_trading['latest_portfolio_value']:.2f}")
            lines.append(f"Last Update:      {paper_trading['latest_timestamp']}")
        else:
            lines.append(f"Status:           {paper_trading['status']}")

        lines.append("=" * 80)

        return "\n".join(lines)

    def run_checks(self):
        """Run all performance checks."""
        self.logger.info("Running performance checks...")

        # Get trial stats
        stats = self.get_trial_stats()

        # Check for new best trial
        self.check_new_best_trial(stats)

        # Get GPU stats
        gpu_stats = self.get_gpu_stats()

        # Check paper trading
        paper_trading = self.check_paper_trading()

        # Generate report
        report = self.generate_performance_report(stats, gpu_stats, paper_trading)
        self.logger.info(f"\n{report}")

        # Save performance snapshot
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "stats": stats,
            "gpu_stats": gpu_stats,
            "paper_trading": paper_trading,
        }

        self.state["performance_history"].append(snapshot)

        # Keep only last 100 snapshots
        if len(self.state["performance_history"]) > 100:
            self.state["performance_history"] = self.state["performance_history"][-100:]

        self.state["last_check_time"] = datetime.now().isoformat()
        self._save_state()

        # Check for alerts
        self._check_alerts(stats, gpu_stats, paper_trading)

    def _check_alerts(self, stats: Dict, gpu_stats: Optional[Dict], paper_trading: Dict):
        """Check for alert conditions."""
        # Alert on slow training progress
        if stats["recent_completions"] == 0 and stats["running"] > 0:
            self.logger.warning("⚠️  No trials completed in the last hour")

        # Alert on GPU issues
        if gpu_stats:
            if gpu_stats["temperature"] > 80:
                self.logger.warning(f"⚠️  High GPU temperature: {gpu_stats['temperature']}°C")
                self.notify("⚠️ GPU Warning", f"Temperature: {gpu_stats['temperature']}°C", urgency="normal")

            if gpu_stats["utilization"] < 50 and stats["running"] > 0:
                self.logger.warning(f"⚠️  Low GPU utilization: {gpu_stats['utilization']}%")

        # Alert on paper trading activity
        if paper_trading["status"] == "active" and paper_trading["trade_count"] > 0:
            last_count = self.state.get("last_trade_count", 0)
            if paper_trading["trade_count"] > last_count:
                new_trades = paper_trading["trade_count"] - last_count
                self.logger.info(f"📈 {new_trades} new trade(s) executed!")
                self.notify("📈 Trade Executed!",
                           f"{new_trades} new trade(s)\nPortfolio: ${paper_trading['latest_portfolio_value']:.2f}",
                           urgency="normal")
                self.state["last_trade_count"] = paper_trading["trade_count"]
                self._save_state()

    def run(self):
        """Main loop."""
        self.logger.info("=" * 80)
        self.logger.info("PERFORMANCE MONITOR STARTED")
        self.logger.info("=" * 80)
        self.logger.info(f"Study: {self.study_name}")
        self.logger.info(f"Check interval: {self.check_interval}s")
        self.logger.info(f"Notifications: {self.enable_notifications}")
        self.logger.info("=" * 80)

        # Initial check
        self.run_checks()

        # Main loop
        while self.running:
            try:
                self.logger.info(f"Sleeping for {self.check_interval}s...")
                time.sleep(self.check_interval)

                if self.running:
                    self.run_checks()

            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(60)

        self.logger.info("Performance monitor stopped")


def main():
    parser = argparse.ArgumentParser(description="Performance monitoring and alerts")
    parser.add_argument("--study", default="cappuccino_3workers_20251102_2325", help="Study name")
    parser.add_argument("--db", default="databases/optuna_cappuccino.db", help="Database path")
    parser.add_argument("--check-interval", type=int, default=300, help="Check interval in seconds")
    parser.add_argument("--no-notifications", action="store_true", help="Disable desktop notifications")

    args = parser.parse_args()

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    monitor = PerformanceMonitor(
        study_name=args.study,
        db_path=args.db,
        check_interval=args.check_interval,
        enable_notifications=not args.no_notifications,
    )

    monitor.run()


if __name__ == "__main__":
    main()