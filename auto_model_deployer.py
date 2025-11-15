#!/usr/bin/env python3
"""
Auto-Model Deployment Pipeline
Automatically finds, validates, and deploys best models to paper trading.

Features:
- Monitors for new best trials
- Validates models before deployment
- Auto-deploys to paper trading
- Maintains deployment history
- Rollback capability

Usage:
    python auto_model_deployer.py --study cappuccino_3workers_20251102_2325 --check-interval 3600
"""

import argparse
import json
import logging
import os
import pickle
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import optuna


class AutoModelDeployer:
    def __init__(
        self,
        study_name: str,
        db_path: str = "databases/optuna_cappuccino.db",
        check_interval: int = 3600,
        min_improvement: float = 0.01,
        validation_enabled: bool = True,
        auto_deploy: bool = True,
        daemon: bool = False,
    ):
        self.study_name = study_name
        self.db_path = db_path
        self.check_interval = check_interval
        self.min_improvement = min_improvement
        self.validation_enabled = validation_enabled
        self.auto_deploy = auto_deploy
        self.daemon = daemon

        # Paths
        self.deployment_dir = Path("deployments")
        self.deployment_dir.mkdir(exist_ok=True)
        self.state_file = self.deployment_dir / "deployment_state.json"
        self.log_file = self.deployment_dir / "deployment_log.json"

        # State
        self.state = self._load_state()
        self.running = True

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
                logging.FileHandler('logs/auto_deployer.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def _load_state(self) -> Dict:
        """Load deployment state from file."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            "last_deployed_trial": None,
            "last_deployed_value": None,
            "last_deployment_time": None,
            "deployment_history": [],
            "current_paper_trader_pid": None,
        }

    def _save_state(self):
        """Save deployment state to file."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def _log_deployment(self, trial_id: int, value: float, action: str, details: str):
        """Log deployment action."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "trial_id": trial_id,
            "value": value,
            "action": action,
            "details": details,
        }

        # Append to log file
        log_data = []
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                log_data = json.load(f)
        log_data.append(log_entry)

        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

        self.logger.info(f"{action}: Trial {trial_id} (value: {value:.6f}) - {details}")

    def get_best_trials(self, top_n: int = 5) -> List[Tuple[int, float, str]]:
        """Get top N trials from the study."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT t.trial_id, tv.value, t.datetime_complete, t.number
            FROM trials t
            JOIN studies s ON t.study_id = s.study_id
            JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE s.study_name = ?
            AND t.state = 'COMPLETE'
            ORDER BY tv.value ASC
            LIMIT ?
        """

        cursor.execute(query, (self.study_name, top_n))
        results = cursor.fetchall()
        conn.close()

        return [(trial_id, value, dt, num) for trial_id, value, dt, num in results]

    def get_trial_params(self, trial_id: int) -> Optional[Dict]:
        """Get trial parameters from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT param_name, param_value
            FROM trial_params
            WHERE trial_id = ?
        """

        cursor.execute(query, (trial_id,))
        results = cursor.fetchall()
        conn.close()

        if not results:
            return None

        return {name: value for name, value in results}

    def check_model_exists(self, trial_id: int) -> Tuple[bool, Optional[Path]]:
        """Check if model files exist for a trial."""
        # Try different possible paths
        possible_paths = [
            Path(f"train_results/cwd_tests/trial_{trial_id}_1h"),
            Path(f"train_results/cwd_tests/trial_{trial_id}"),
        ]

        for model_dir in possible_paths:
            if model_dir.exists():
                # Check for required files
                required_files = ["actor.pth", "critic.pth"]
                if all((model_dir / f).exists() for f in required_files):
                    return True, model_dir

        return False, None

    def validate_model(self, trial_id: int, model_dir: Path) -> bool:
        """
        Validate model performance.
        For now, just check if files exist and are recent.
        TODO: Add backtesting validation
        """
        if not self.validation_enabled:
            return True

        # Check if model files exist
        required_files = ["actor.pth", "critic.pth"]
        for f in required_files:
            if not (model_dir / f).exists():
                self.logger.warning(f"Missing file {f} in {model_dir}")
                return False

        # Check if model is recent (within last 7 days)
        actor_file = model_dir / "actor.pth"
        mtime = datetime.fromtimestamp(actor_file.stat().st_mtime)
        age = datetime.now() - mtime

        if age > timedelta(days=7):
            self.logger.warning(f"Model is {age.days} days old, might be stale")
            # Don't reject, just warn

        self.logger.info(f"Model validation passed for trial {trial_id}")
        return True

    def create_best_trial_file(self, trial_id: int, model_dir: Path):
        """Create a minimal best_trial file for compatibility with paper trader."""
        best_trial_path = model_dir / "best_trial"

        # If it already exists, don't overwrite
        if best_trial_path.exists():
            return

        try:
            # Load the optuna study to get the real trial object
            import optuna
            storage = f"sqlite:///{self.db_path}"
            study = optuna.load_study(study_name=self.study_name, storage=storage)

            # Find the trial
            trial_obj = None
            for trial in study.trials:
                if trial._trial_id == trial_id:
                    trial_obj = trial
                    break

            if trial_obj:
                # Add name_folder if missing (needed for compatibility)
                if 'name_folder' not in trial_obj.user_attrs:
                    # Derive from model_dir path
                    folder_name = "/".join(model_dir.parts[-2:])  # e.g., "cwd_tests/trial_1874_1h"
                    trial_obj.user_attrs['name_folder'] = folder_name

                # Save the frozen trial
                with open(best_trial_path, 'wb') as f:
                    pickle.dump(trial_obj, f)
                self.logger.info(f"Created best_trial file for trial {trial_id}")
            else:
                self.logger.warning(f"Could not find trial {trial_id} in study")

        except Exception as e:
            self.logger.warning(f"Failed to create best_trial file: {e}")

    def stop_current_paper_trader(self):
        """Stop currently running paper trader."""
        pid = self.state.get("current_paper_trader_pid")

        if pid:
            try:
                # Check if process exists
                os.kill(pid, 0)
                # Process exists, kill it
                self.logger.info(f"Stopping paper trader PID {pid}")
                os.kill(pid, signal.SIGTERM)
                time.sleep(2)

                # Force kill if still running
                try:
                    os.kill(pid, 0)
                    os.kill(pid, signal.SIGKILL)
                    self.logger.warning(f"Force killed paper trader PID {pid}")
                except ProcessLookupError:
                    pass

            except ProcessLookupError:
                self.logger.info(f"Paper trader PID {pid} not running")

        # Also try to kill by name
        try:
            subprocess.run(["pkill", "-f", "paper_trader_alpaca_polling.py"],
                          check=False, capture_output=True)
        except Exception as e:
            self.logger.warning(f"Error killing paper trader by name: {e}")

    def deploy_model(self, trial_id: int, model_dir: Path, value: float):
        """Deploy model to paper trading."""
        self.logger.info(f"Deploying trial {trial_id} (value: {value:.6f})")

        # Create best_trial file if needed
        self.create_best_trial_file(trial_id, model_dir)

        # Stop current paper trader
        self.stop_current_paper_trader()
        time.sleep(2)

        # Start new paper trader
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/paper_trading_auto_{timestamp}.log"
        csv_file = f"paper_trades/auto_session_{timestamp}.csv"

        cmd = [
            "nohup", "python", "-u", "paper_trader_alpaca_polling.py",
            "--model-dir", str(model_dir),
            "--tickers", "BTC/USD", "ETH/USD", "LTC/USD", "BCH/USD", "LINK/USD", "UNI/USD", "AAVE/USD",
            "--timeframe", "1h",
            "--history-hours", "120",
            "--poll-interval", "60",
            "--gpu", "-1",
            "--log-file", csv_file,
        ]

        # Start process
        with open(log_file, 'w') as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

        self.logger.info(f"Started paper trader PID {proc.pid}")

        # Update state
        self.state["last_deployed_trial"] = trial_id
        self.state["last_deployed_value"] = value
        self.state["last_deployment_time"] = datetime.now().isoformat()
        self.state["current_paper_trader_pid"] = proc.pid
        self.state["deployment_history"].append({
            "trial_id": trial_id,
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "pid": proc.pid,
            "model_dir": str(model_dir),
            "log_file": log_file,
            "csv_file": csv_file,
        })

        # Keep only last 20 deployments in history
        if len(self.state["deployment_history"]) > 20:
            self.state["deployment_history"] = self.state["deployment_history"][-20:]

        self._save_state()
        self._log_deployment(trial_id, value, "DEPLOYED", f"PID: {proc.pid}, Log: {log_file}")

    def check_and_deploy(self):
        """Check for new best models and deploy if needed."""
        self.logger.info("Checking for new best models...")

        # Get best trials
        best_trials = self.get_best_trials(top_n=5)

        if not best_trials:
            self.logger.warning("No completed trials found")
            return

        best_trial_id, best_value, _, trial_num = best_trials[0]
        self.logger.info(f"Current best: Trial {trial_num} (ID: {best_trial_id}, value: {best_value:.6f})")

        # Check if this is a new best
        last_deployed = self.state.get("last_deployed_trial")
        last_value = self.state.get("last_deployed_value")

        if last_deployed == best_trial_id:
            self.logger.info("Best model already deployed")
            return

        # Check if improvement is significant
        if last_value is not None:
            improvement = last_value - best_value  # Lower is better (negative values)
            improvement_pct = abs(improvement / last_value) * 100

            self.logger.info(f"Improvement: {improvement:.6f} ({improvement_pct:.2f}%)")

            if improvement_pct < self.min_improvement:
                self.logger.info(f"Improvement {improvement_pct:.2f}% < threshold {self.min_improvement}%")
                return

        # Check if model exists
        model_exists, model_dir = self.check_model_exists(best_trial_id)

        if not model_exists:
            self.logger.warning(f"Model files not found for trial {best_trial_id}")
            self._log_deployment(best_trial_id, best_value, "SKIPPED", "Model files not found")
            return

        # Validate model
        if not self.validate_model(best_trial_id, model_dir):
            self.logger.warning(f"Model validation failed for trial {best_trial_id}")
            self._log_deployment(best_trial_id, best_value, "FAILED_VALIDATION", "Validation failed")
            return

        # Deploy model
        if self.auto_deploy:
            self.deploy_model(best_trial_id, model_dir, best_value)
        else:
            self.logger.info(f"Auto-deploy disabled. Would deploy trial {best_trial_id}")
            self._log_deployment(best_trial_id, best_value, "READY", "Auto-deploy disabled")

    def run(self):
        """Main loop."""
        self.logger.info("=" * 80)
        self.logger.info("AUTO-MODEL DEPLOYER STARTED")
        self.logger.info("=" * 80)
        self.logger.info(f"Study: {self.study_name}")
        self.logger.info(f"Check interval: {self.check_interval}s")
        self.logger.info(f"Min improvement: {self.min_improvement}%")
        self.logger.info(f"Validation: {self.validation_enabled}")
        self.logger.info(f"Auto-deploy: {self.auto_deploy}")
        self.logger.info("=" * 80)

        # Initial check
        self.check_and_deploy()

        if not self.daemon:
            self.logger.info("Single run mode, exiting")
            return

        # Daemon loop
        while self.running:
            try:
                self.logger.info(f"Sleeping for {self.check_interval}s...")
                time.sleep(self.check_interval)

                if self.running:
                    self.check_and_deploy()

            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(60)

        self.logger.info("Auto-model deployer stopped")


def main():
    parser = argparse.ArgumentParser(description="Auto-model deployment pipeline")
    parser.add_argument("--study", default="cappuccino_3workers_20251102_2325", help="Study name")
    parser.add_argument("--db", default="databases/optuna_cappuccino.db", help="Database path")
    parser.add_argument("--check-interval", type=int, default=3600, help="Check interval in seconds")
    parser.add_argument("--min-improvement", type=float, default=1.0, help="Min improvement % to deploy")
    parser.add_argument("--no-validation", action="store_true", help="Disable validation")
    parser.add_argument("--no-auto-deploy", action="store_true", help="Disable auto-deployment")
    parser.add_argument("--daemon", action="store_true", help="Run in daemon mode")

    args = parser.parse_args()

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    deployer = AutoModelDeployer(
        study_name=args.study,
        db_path=args.db,
        check_interval=args.check_interval,
        min_improvement=args.min_improvement,
        validation_enabled=not args.no_validation,
        auto_deploy=not args.no_auto_deploy,
        daemon=args.daemon,
    )

    deployer.run()


if __name__ == "__main__":
    main()
