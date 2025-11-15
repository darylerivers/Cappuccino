#!/usr/bin/env python3
"""
System Watchdog - Health Monitoring & Auto-Restart
Monitors all critical processes and restarts them if they crash.

Monitors:
- Training workers (3x)
- Paper trading
- Autonomous AI advisor
- GPU health
- Database integrity

Features:
- Auto-restart crashed processes
- Health checks
- Alert logging
- Resource monitoring
- Email/desktop notifications (optional)

Usage:
    python system_watchdog.py --check-interval 60
"""

import argparse
import json
import logging
import os
import psutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class SystemWatchdog:
    def __init__(
        self,
        check_interval: int = 60,
        auto_restart: bool = True,
        max_restarts: int = 3,
        restart_cooldown: int = 300,
    ):
        self.check_interval = check_interval
        self.auto_restart = auto_restart
        self.max_restarts = max_restarts
        self.restart_cooldown = restart_cooldown

        # State
        self.running = True
        self.state_file = Path("deployments/watchdog_state.json")
        self.state_file.parent.mkdir(exist_ok=True)
        self.state = self._load_state()

        # Setup logging
        self._setup_logging()

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Process definitions
        self.processes = {
            "training_workers": {
                "pattern": "1_optimize_unified.py",
                "expected_count": 3,
                "restart_cmd": self._restart_training_workers,
                "critical": True,
            },
            "paper_trading": {
                "pattern": "paper_trader_alpaca_polling.py",
                "expected_count": 1,
                "restart_cmd": self._restart_paper_trading,
                "critical": True,
            },
            "ai_advisor": {
                "pattern": "ollama_autonomous_advisor.py",
                "expected_count": 1,
                "restart_cmd": self._restart_ai_advisor,
                "critical": False,
            },
        }

    def _setup_logging(self):
        """Setup logging configuration."""
        log_format = '[%(asctime)s] [%(levelname)s] %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('logs/watchdog.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def _load_state(self) -> Dict:
        """Load watchdog state."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            "restart_counts": {},
            "last_restart_times": {},
            "alerts": [],
        }

    def _save_state(self):
        """Save watchdog state."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def _log_alert(self, severity: str, process: str, message: str):
        """Log an alert."""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "severity": severity,
            "process": process,
            "message": message,
        }

        self.state["alerts"].append(alert)

        # Keep only last 100 alerts
        if len(self.state["alerts"]) > 100:
            self.state["alerts"] = self.state["alerts"][-100:]

        self._save_state()

        log_func = {
            "INFO": self.logger.info,
            "WARNING": self.logger.warning,
            "ERROR": self.logger.error,
            "CRITICAL": self.logger.critical,
        }.get(severity, self.logger.info)

        log_func(f"[{process}] {message}")

    def find_processes(self, pattern: str) -> List[psutil.Process]:
        """Find processes matching pattern."""
        matches = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if pattern in cmdline and 'grep' not in cmdline:
                    matches.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return matches

    def check_process_health(self, name: str, config: Dict) -> Tuple[bool, str]:
        """Check health of a process."""
        pattern = config["pattern"]
        expected_count = config["expected_count"]

        procs = self.find_processes(pattern)
        actual_count = len(procs)

        if actual_count == 0:
            return False, f"No processes running (expected {expected_count})"

        if actual_count < expected_count:
            return False, f"Only {actual_count}/{expected_count} processes running"

        # Check if processes are responsive (not zombies)
        for proc in procs:
            try:
                if proc.status() == psutil.STATUS_ZOMBIE:
                    return False, f"Zombie process detected: PID {proc.pid}"
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        return True, f"OK ({actual_count}/{expected_count} running)"

    def _can_restart(self, process_name: str) -> Tuple[bool, str]:
        """Check if process can be restarted."""
        # Check restart count
        restart_count = self.state["restart_counts"].get(process_name, 0)
        if restart_count >= self.max_restarts:
            return False, f"Max restarts ({self.max_restarts}) exceeded"

        # Check cooldown
        last_restart = self.state["last_restart_times"].get(process_name)
        if last_restart:
            last_restart_dt = datetime.fromisoformat(last_restart)
            elapsed = (datetime.now() - last_restart_dt).total_seconds()
            if elapsed < self.restart_cooldown:
                return False, f"Cooldown period ({int(self.restart_cooldown - elapsed)}s remaining)"

        return True, "OK"

    def _record_restart(self, process_name: str):
        """Record a restart."""
        self.state["restart_counts"][process_name] = \
            self.state["restart_counts"].get(process_name, 0) + 1
        self.state["last_restart_times"][process_name] = datetime.now().isoformat()
        self._save_state()

    def _reset_restart_count(self, process_name: str):
        """Reset restart count after successful operation."""
        if process_name in self.state["restart_counts"]:
            self.state["restart_counts"][process_name] = 0
            self._save_state()

    def _restart_training_workers(self) -> bool:
        """Restart training workers."""
        self.logger.info("Restarting training workers...")

        # Kill existing workers
        subprocess.run(["pkill", "-f", "1_optimize_unified.py"], check=False)
        time.sleep(5)

        # Start new workers
        cmd = """
        STUDY_NAME="cappuccino_3workers_20251102_2325"
        N_PARALLEL=3
        mkdir -p logs/parallel_training

        for i in $(seq 1 $N_PARALLEL); do
            echo "[$(date)] Launching worker $i..."
            python -u 1_optimize_unified.py \
                --n-trials 500 \
                --gpu 0 \
                --study-name $STUDY_NAME \
                2>&1 | sed "s/^/[W$i] /" > logs/parallel_training/worker_$i.log &
            sleep 5
        done
        """

        try:
            subprocess.Popen(["bash", "-c", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(15)

            # Verify workers started
            procs = self.find_processes("1_optimize_unified.py")
            if len(procs) >= 3:
                self.logger.info(f"Successfully restarted {len(procs)} training workers")
                return True
            else:
                self.logger.error(f"Only {len(procs)} workers started")
                return False

        except Exception as e:
            self.logger.error(f"Failed to restart training workers: {e}")
            return False

    def _restart_paper_trading(self) -> bool:
        """Restart paper trading."""
        self.logger.info("Restarting paper trading...")

        # Kill existing
        subprocess.run(["pkill", "-f", "paper_trader_alpaca_polling.py"], check=False)
        time.sleep(2)

        # Check for auto-deployer state
        deployer_state_file = Path("deployments/deployment_state.json")
        model_dir = "train_results/cwd_tests/trial_141_1h"  # Default

        if deployer_state_file.exists():
            try:
                with open(deployer_state_file, 'r') as f:
                    state = json.load(f)
                    if state.get("deployment_history"):
                        latest = state["deployment_history"][-1]
                        model_dir = latest.get("model_dir", model_dir)
                        self.logger.info(f"Using auto-deployed model: {model_dir}")
            except Exception as e:
                self.logger.warning(f"Could not load deployer state: {e}")

        # Start paper trading
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/paper_trading_watchdog_{timestamp}.log"
        csv_file = f"paper_trades/watchdog_session_{timestamp}.csv"

        cmd = [
            "nohup", "python", "-u", "paper_trader_alpaca_polling.py",
            "--model-dir", model_dir,
            "--tickers", "BTC/USD", "ETH/USD", "LTC/USD", "BCH/USD", "LINK/USD", "UNI/USD", "AAVE/USD",
            "--timeframe", "1h",
            "--history-hours", "120",
            "--poll-interval", "60",
            "--gpu", "-1",
            "--log-file", csv_file,
        ]

        try:
            with open(log_file, 'w') as f:
                proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

            time.sleep(5)

            # Verify started
            if proc.poll() is None:
                self.logger.info(f"Successfully restarted paper trading (PID {proc.pid})")
                return True
            else:
                self.logger.error("Paper trading process terminated immediately")
                return False

        except Exception as e:
            self.logger.error(f"Failed to restart paper trading: {e}")
            return False

    def _restart_ai_advisor(self) -> bool:
        """Restart AI advisor."""
        self.logger.info("Restarting AI advisor...")

        # Kill existing
        subprocess.run(["pkill", "-f", "ollama_autonomous_advisor.py"], check=False)
        time.sleep(2)

        # Start AI advisor
        try:
            subprocess.run(["./start_autonomous_advisor.sh"], check=True, capture_output=True)
            time.sleep(5)

            # Verify started
            procs = self.find_processes("ollama_autonomous_advisor.py")
            if len(procs) >= 1:
                self.logger.info("Successfully restarted AI advisor")
                return True
            else:
                self.logger.error("AI advisor did not start")
                return False

        except Exception as e:
            self.logger.error(f"Failed to restart AI advisor: {e}")
            return False

    def check_gpu_health(self) -> Tuple[bool, str]:
        """Check GPU health."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode != 0:
                return False, "nvidia-smi failed"

            temp, util = result.stdout.strip().split(',')
            temp = int(temp.strip())
            util = int(util.strip())

            # Check temperature
            if temp > 85:
                return False, f"GPU temperature critical: {temp}°C"

            if temp > 80:
                self._log_alert("WARNING", "GPU", f"High temperature: {temp}°C")

            return True, f"OK (Temp: {temp}°C, Util: {util}%)"

        except Exception as e:
            return False, f"GPU check failed: {e}"

    def check_disk_space(self) -> Tuple[bool, str]:
        """Check disk space."""
        try:
            usage = psutil.disk_usage('/')
            percent = usage.percent

            if percent > 95:
                return False, f"Disk space critical: {percent}% used"

            if percent > 90:
                self._log_alert("WARNING", "DISK", f"Low disk space: {percent}% used")

            return True, f"OK ({percent}% used)"

        except Exception as e:
            return False, f"Disk check failed: {e}"

    def check_database_integrity(self) -> Tuple[bool, str]:
        """Check database integrity."""
        db_path = "databases/optuna_cappuccino.db"

        if not Path(db_path).exists():
            return False, "Database file not found"

        try:
            result = subprocess.run(
                ["sqlite3", db_path, "PRAGMA integrity_check;"],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode != 0:
                return False, "Database integrity check failed"

            if "ok" in result.stdout.lower():
                return True, "OK"
            else:
                return False, f"Database issues: {result.stdout}"

        except Exception as e:
            return False, f"Database check failed: {e}"

    def run_health_checks(self):
        """Run all health checks."""
        self.logger.info("=" * 80)
        self.logger.info("Running health checks...")

        all_healthy = True

        # Check processes
        for name, config in self.processes.items():
            healthy, status = self.check_process_health(name, config)

            if healthy:
                self.logger.info(f"✓ {name}: {status}")
                self._reset_restart_count(name)
            else:
                severity = "CRITICAL" if config["critical"] else "WARNING"
                self._log_alert(severity, name, f"Health check failed: {status}")

                if self.auto_restart:
                    can_restart, reason = self._can_restart(name)

                    if can_restart:
                        self.logger.warning(f"Attempting to restart {name}...")
                        success = config["restart_cmd"]()

                        if success:
                            self._record_restart(name)
                            self._log_alert("INFO", name, "Successfully restarted")
                        else:
                            self._log_alert("ERROR", name, "Restart failed")
                            all_healthy = False
                    else:
                        self._log_alert("ERROR", name, f"Cannot restart: {reason}")
                        all_healthy = False
                else:
                    self.logger.warning(f"Auto-restart disabled for {name}")
                    all_healthy = False

        # Check GPU
        healthy, status = self.check_gpu_health()
        if healthy:
            self.logger.info(f"✓ GPU: {status}")
        else:
            self._log_alert("CRITICAL", "GPU", status)
            all_healthy = False

        # Check disk
        healthy, status = self.check_disk_space()
        if healthy:
            self.logger.info(f"✓ Disk: {status}")
        else:
            self._log_alert("CRITICAL", "DISK", status)
            all_healthy = False

        # Check database
        healthy, status = self.check_database_integrity()
        if healthy:
            self.logger.info(f"✓ Database: {status}")
        else:
            self._log_alert("ERROR", "DATABASE", status)
            all_healthy = False

        if all_healthy:
            self.logger.info("All systems healthy ✓")
        else:
            self.logger.warning("Some systems unhealthy!")

        self.logger.info("=" * 80)

    def run(self):
        """Main loop."""
        self.logger.info("=" * 80)
        self.logger.info("SYSTEM WATCHDOG STARTED")
        self.logger.info("=" * 80)
        self.logger.info(f"Check interval: {self.check_interval}s")
        self.logger.info(f"Auto-restart: {self.auto_restart}")
        self.logger.info(f"Max restarts: {self.max_restarts}")
        self.logger.info(f"Restart cooldown: {self.restart_cooldown}s")
        self.logger.info("=" * 80)

        # Initial check
        self.run_health_checks()

        # Main loop
        while self.running:
            try:
                self.logger.info(f"Sleeping for {self.check_interval}s...")
                time.sleep(self.check_interval)

                if self.running:
                    self.run_health_checks()

            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(60)

        self.logger.info("System watchdog stopped")


def main():
    parser = argparse.ArgumentParser(description="System watchdog for health monitoring")
    parser.add_argument("--check-interval", type=int, default=60, help="Check interval in seconds")
    parser.add_argument("--no-auto-restart", action="store_true", help="Disable auto-restart")
    parser.add_argument("--max-restarts", type=int, default=3, help="Max restarts per process")
    parser.add_argument("--restart-cooldown", type=int, default=300, help="Cooldown between restarts (seconds)")

    args = parser.parse_args()

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    watchdog = SystemWatchdog(
        check_interval=args.check_interval,
        auto_restart=not args.no_auto_restart,
        max_restarts=args.max_restarts,
        restart_cooldown=args.restart_cooldown,
    )

    watchdog.run()


if __name__ == "__main__":
    main()
