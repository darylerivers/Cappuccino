#!/usr/bin/env python3
"""
Auto-Replace Best Model for 5m Training
Monitors cappuccino_5m_fresh study and auto-replaces paper trader when better trial completes.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import optuna


class AutoReplace5m:
    def __init__(self, study_name="cappuccino_5m_speed", check_interval=300):
        self.study_name = study_name
        self.check_interval = check_interval
        self.db_path = "databases/optuna_cappuccino.db"
        self.state_file = Path("deployments/auto_replace_5m_state.json")
        self.state_file.parent.mkdir(exist_ok=True)
        self.running = True

        # Load state
        self.state = self._load_state()

        # Signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        print(f"Received signal {signum}, shutting down...")
        self.running = False

    def _load_state(self):
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {
            "last_deployed_trial": None,
            "last_deployed_value": None,
            "current_pid": None,
        }

    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def get_best_trial(self):
        """Get current best completed trial."""
        try:
            study = optuna.load_study(
                study_name=self.study_name,
                storage=f"sqlite:///{self.db_path}"
            )

            complete = [t for t in study.get_trials(deepcopy=False)
                       if t.state == optuna.trial.TrialState.COMPLETE]

            if not complete:
                return None, None

            best = max(complete, key=lambda t: t.value if t.value else float('-inf'))
            return best.number, best.value

        except Exception as e:
            print(f"Error getting best trial: {e}")
            return None, None

    def stop_current_trader(self):
        """Stop currently running paper trader."""
        pid = self.state.get("current_pid")

        if pid:
            try:
                os.kill(pid, 0)  # Check if exists
                print(f"Stopping paper trader PID {pid}")
                os.kill(pid, signal.SIGTERM)
                time.sleep(2)

                # Force kill if still running
                try:
                    os.kill(pid, 0)
                    os.kill(pid, signal.SIGKILL)
                    print(f"Force killed PID {pid}")
                except ProcessLookupError:
                    pass

            except ProcessLookupError:
                print(f"PID {pid} not running")

        # Also kill by name
        subprocess.run(["pkill", "-f", "paper_trader.*trial.*_5m"], check=False)

    def deploy_trial(self, trial_num, value):
        """Deploy a trial to paper trading."""
        model_dir = f"train_results/cwd_tests/trial_{trial_num}_5m"

        if not Path(model_dir).exists():
            print(f"Model directory not found: {model_dir}")
            return False

        # Stop current trader
        self.stop_current_trader()
        time.sleep(2)

        # Start new trader
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/paper_trader_trial{trial_num}_{timestamp}.log"
        csv_file = f"paper_trades/trial{trial_num}_session.csv"

        # Use correct Python environment
        python_path = "/home/mrc/.pyenv/versions/cappuccino-rocm/bin/python"

        cmd = [
            "nohup", python_path, "-u", "scripts/deployment/paper_trader_alpaca_polling.py",
            "--model-dir", model_dir,
            # Use all 7 tickers (model trained with them) but 1h timeframe (AVAX has 1h data)
            "--tickers", "AAVE/USD", "AVAX/USD", "BTC/USD", "LINK/USD", "ETH/USD", "LTC/USD", "UNI/USD",
            "--timeframe", "1h",  # 1h has better data availability than 5m
            "--history-hours", "120",  # 5 days of history for 1h bars
            "--poll-interval", "3600",  # Poll every hour for 1h timeframe
            "--gpu", "-1",
            "--log-file", csv_file,
        ]

        with open(log_file, 'w') as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

        print(f"✅ Deployed Trial #{trial_num} (Sharpe: {value:.6f}) - PID {proc.pid}")

        # Update state
        self.state["last_deployed_trial"] = trial_num
        self.state["last_deployed_value"] = value
        self.state["current_pid"] = proc.pid
        self._save_state()

        return True

    def check_and_replace(self):
        """Check for better trial and replace if found."""
        best_num, best_value = self.get_best_trial()

        if best_num is None:
            print("No completed trials yet")
            return

        last_deployed = self.state.get("last_deployed_trial")
        last_value = self.state.get("last_deployed_value")

        print(f"Current best: Trial #{best_num} (Sharpe: {best_value:.6f})")

        if last_deployed is None:
            # First deployment
            print(f"Initial deployment...")
            self.deploy_trial(best_num, best_value)
        elif best_num != last_deployed and (last_value is None or best_value > last_value):
            # Better trial found
            improvement = best_value - (last_value or 0)
            print(f"🎯 Better trial found! Trial #{best_num} (improvement: +{improvement:.6f})")
            self.deploy_trial(best_num, best_value)
        else:
            print(f"Trial #{last_deployed} still best (current deployment)")

    def run(self):
        """Main loop."""
        print("=" * 80)
        print("AUTO-REPLACE 5M PAPER TRADER")
        print("=" * 80)
        print(f"Study: {self.study_name}")
        print(f"Check interval: {self.check_interval}s")
        print("=" * 80)

        while self.running:
            try:
                self.check_and_replace()
                print(f"\nNext check in {self.check_interval}s...\n")
                time.sleep(self.check_interval)
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(60)

        print("Auto-replace stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-interval", type=int, default=300, help="Check interval in seconds")
    args = parser.parse_args()

    replacer = AutoReplace5m(check_interval=args.check_interval)
    replacer.run()
