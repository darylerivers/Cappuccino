#!/usr/bin/env python3
"""
Paper Trading Deployer V2 - Simple Model Deployment

Deploys models to paper trading without complex dependencies.
"""

import logging
import subprocess
import time
import psutil
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional
from utils.path_detector import PathDetector

logger = logging.getLogger(__name__)


class PaperTradingDeployer:
    """Deploys models to paper trading."""

    def __init__(self, optuna_db: str = None):
        """Initialize deployer."""
        # Auto-detect database if not provided
        if optuna_db is None:
            detector = PathDetector()
            self.optuna_db = detector.find_optuna_db()
        else:
            self.optuna_db = optuna_db

    def _create_best_trial_file(self, trial_num: int, model_dir: Path) -> bool:
        """
        Create best_trial pickle file needed by paper trader.

        Args:
            trial_num: Trial number
            model_dir: Model directory path

        Returns:
            True if successful
        """
        try:
            import optuna

            # Get list of all studies from database
            conn = sqlite3.connect(self.optuna_db)
            cursor = conn.cursor()
            cursor.execute("SELECT study_name FROM studies")
            study_names = [row[0] for row in cursor.fetchall()]
            conn.close()

            # Find the trial in any study
            trial_obj = None
            for study_name in study_names:
                try:
                    study = optuna.load_study(
                        study_name=study_name,
                        storage=f"sqlite:///{self.optuna_db}"
                    )

                    # Check if this study has the trial
                    for trial in study.trials:
                        if trial.number == trial_num and trial.state == optuna.trial.TrialState.COMPLETE:
                            trial_obj = trial
                            logger.info(f"Found trial {trial_num} in study '{study_name}'")
                            break

                    if trial_obj:
                        break
                except Exception as e:
                    logger.debug(f"Could not load study {study_name}: {e}")
                    continue

            if not trial_obj:
                logger.error(f"Could not find completed trial {trial_num} in any study")
                return False

            # Save trial object as pickle
            best_trial_path = model_dir / "best_trial"
            with open(best_trial_path, 'wb') as f:
                pickle.dump(trial_obj, f)

            logger.info(f"Created best_trial file for trial {trial_num}")
            return True

        except Exception as e:
            logger.error(f"Failed to create best_trial file: {e}", exc_info=True)
            return False

    def deploy(self, trial_num: int) -> Dict[str, Any]:
        """
        Deploy a trial's model to paper trading.

        Args:
            trial_num: Trial number to deploy

        Returns:
            Dict with success status and process info
        """
        try:
            logger.info(f"Deploying trial {trial_num} to paper trading")

            # Find model directory
            model_dir = Path(f"train_results/cwd_tests/trial_{trial_num}_1h")
            if not model_dir.exists():
                return {'success': False, 'error': f'Model directory not found: {model_dir}'}

            # Check for model files
            actor_path = model_dir / "actor.pth"
            if not actor_path.exists():
                return {'success': False, 'error': f'Actor model not found: {actor_path}'}

            # Create best_trial file if it doesn't exist
            best_trial_path = model_dir / "best_trial"
            if not best_trial_path.exists():
                logger.info(f"Creating best_trial file for trial {trial_num}")
                if not self._create_best_trial_file(trial_num, model_dir):
                    return {'success': False, 'error': 'Failed to create best_trial file'}

            # Auto-detect log directory and create log file path
            detector = PathDetector()
            log_dir = detector.find_log_dir()
            log_file = Path(f"{log_dir}/paper_trading_trial{trial_num}.log")
            log_file.parent.mkdir(exist_ok=True)

            # Build command (use -u for unbuffered output)
            cmd = [
                'python', '-u', 'paper_trader_alpaca_polling.py',
                '--model-dir', str(model_dir),
                '--timeframe', '1h'
            ]

            logger.info(f"Starting paper trader: {' '.join(cmd)}")
            logger.info(f"Log file: {log_file}")

            # Start process
            with open(log_file, 'w') as log_f:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    cwd=Path.cwd()
                )

            pid = process.pid
            logger.info(f"Paper trader started with PID {pid}")

            # Wait a bit and check if it's still running
            time.sleep(5)

            if not self._is_process_running(pid):
                # Process died, check logs
                with open(log_file) as f:
                    error_log = f.read()

                logger.error(f"Process died immediately. Log:\n{error_log[-500:]}")

                return {
                    'success': False,
                    'error': 'Process died after start',
                    'log_snippet': error_log[-500:]
                }

            logger.info(f"Trial {trial_num} deployed successfully")

            return {
                'success': True,
                'trial_number': trial_num,
                'process_id': pid,
                'log_file': str(log_file)
            }

        except Exception as e:
            logger.error(f"Deployment error: {str(e)}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def _is_process_running(self, pid: int) -> bool:
        """Check if a process is running."""
        try:
            process = psutil.Process(pid)
            return process.is_running()
        except psutil.NoSuchProcess:
            return False

    def monitor(self, pid: int) -> str:
        """
        Monitor a deployed process.

        Args:
            pid: Process ID

        Returns:
            Status string
        """
        if self._is_process_running(pid):
            try:
                process = psutil.Process(pid)
                status = process.status()
                cpu_percent = process.cpu_percent(interval=0.1)
                memory_mb = process.memory_info().rss / 1024 / 1024

                return f"running (CPU: {cpu_percent:.1f}%, RAM: {memory_mb:.1f}MB)"
            except:
                return "running"
        else:
            return "stopped"

    def stop(self, pid: int) -> bool:
        """
        Stop a deployed process.

        Args:
            pid: Process ID

        Returns:
            True if stopped successfully
        """
        try:
            if self._is_process_running(pid):
                process = psutil.Process(pid)
                process.terminate()
                process.wait(timeout=10)
                logger.info(f"Stopped process {pid}")
                return True
            else:
                logger.info(f"Process {pid} already stopped")
                return True

        except Exception as e:
            logger.error(f"Failed to stop process {pid}: {e}")
            return False


def main():
    """Test deployer."""
    import argparse

    parser = argparse.ArgumentParser(description='Paper Trading Deployer V2')
    parser.add_argument('trial_num', type=int, help='Trial number to deploy')
    parser.add_argument('--stop-pid', type=int, help='Stop a process')
    parser.add_argument('--monitor-pid', type=int, help='Monitor a process')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    deployer = PaperTradingDeployer()

    if args.stop_pid:
        result = deployer.stop(args.stop_pid)
        print(f"Stop result: {result}")
    elif args.monitor_pid:
        status = deployer.monitor(args.monitor_pid)
        print(f"Status: {status}")
    else:
        result = deployer.deploy(args.trial_num)
        print(f"\nDeployment: {'SUCCESS' if result['success'] else 'FAILED'}")
        if result['success']:
            print(f"PID: {result['process_id']}")
            print(f"Log: {result['log_file']}")
        else:
            print(f"Error: {result['error']}")


if __name__ == '__main__':
    main()
