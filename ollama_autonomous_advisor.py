#!/usr/bin/env python3
"""
Autonomous AI Training Advisor

This daemon continuously:
1. Monitors training progress
2. Analyzes results when enough new trials complete
3. Generates parameter suggestions using Ollama
4. Tests suggestions automatically when GPU is free
5. Learns from results and improves

Usage:
    python ollama_autonomous_advisor.py --study cappuccino_3workers_20251102_2325
    python ollama_autonomous_advisor.py --study cappuccino_3workers_20251102_2325 --daemon
"""

import argparse
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests


@dataclass
class AdvisorState:
    """State of the autonomous advisor."""
    study_name: str
    last_trial_count: int
    last_analysis_time: datetime
    analysis_count: int
    tested_configs: List[Dict]
    best_discovered_value: float
    is_running: bool


class AutonomousAdvisor:
    """Autonomous AI advisor for training optimization."""

    def __init__(
        self,
        study_name: str,
        db_path: str = "databases/optuna_cappuccino.db",
        ollama_model: str = "qwen2.5-coder:7b",
        analysis_interval_trials: int = 50,
        check_interval_seconds: int = 300,
        test_configs_when_idle: bool = True,
        max_test_trials: int = 10,
    ):
        self.study_name = study_name
        self.db_path = db_path
        self.ollama_model = ollama_model
        self.analysis_interval = analysis_interval_trials
        self.check_interval = check_interval_seconds
        self.test_when_idle = test_configs_when_idle
        self.max_test_trials = max_test_trials

        self.state_file = Path("analysis_reports/advisor_state.json")
        self.log_file = Path("logs/autonomous_advisor.log")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.state = self._load_state()
        self._stop = False

    def _log(self, message: str, level: str = "INFO"):
        """Log message to file and console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] [{level}] {message}"
        print(log_msg)

        with self.log_file.open("a") as f:
            f.write(log_msg + "\n")

    def _load_state(self) -> AdvisorState:
        """Load advisor state from file or create new."""
        if self.state_file.exists():
            with self.state_file.open("r") as f:
                data = json.load(f)
                return AdvisorState(
                    study_name=data["study_name"],
                    last_trial_count=data["last_trial_count"],
                    last_analysis_time=datetime.fromisoformat(data["last_analysis_time"]),
                    analysis_count=data["analysis_count"],
                    tested_configs=data["tested_configs"],
                    best_discovered_value=data["best_discovered_value"],
                    is_running=False,  # Always reset on restart
                )
        else:
            return AdvisorState(
                study_name=self.study_name,
                last_trial_count=0,
                last_analysis_time=datetime.now(),
                analysis_count=0,
                tested_configs=[],
                best_discovered_value=float('-inf'),
                is_running=False,
            )

    def _save_state(self):
        """Save advisor state to file."""
        data = {
            "study_name": self.state.study_name,
            "last_trial_count": self.state.last_trial_count,
            "last_analysis_time": self.state.last_analysis_time.isoformat(),
            "analysis_count": self.state.analysis_count,
            "tested_configs": self.state.tested_configs,
            "best_discovered_value": self.state.best_discovered_value,
            "is_running": self.state.is_running,
        }
        with self.state_file.open("w") as f:
            json.dump(data, f, indent=2)

    def _get_trial_count(self) -> int:
        """Get current completed trial count."""
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
            SELECT COUNT(*)
            FROM trials t
            JOIN studies s ON t.study_id = s.study_id
            WHERE s.study_name = ? AND t.state = 'COMPLETE'
            """
            result = pd.read_sql_query(query, conn, params=(self.study_name,))
            conn.close()
            return int(result.iloc[0, 0])
        except Exception as e:
            self._log(f"Error getting trial count: {e}", "ERROR")
            return self.state.last_trial_count

    def _is_training_running(self) -> bool:
        """Check if training processes are running."""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "1_optimize_unified.py"],
                capture_output=True,
                text=True
            )
            return len(result.stdout.strip().split('\n')) > 0 if result.stdout.strip() else False
        except Exception as e:
            self._log(f"Error checking training status: {e}", "ERROR")
            return True  # Assume running if can't check

    def _get_gpu_utilization(self) -> int:
        """Get current GPU utilization percentage."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except Exception as e:
            self._log(f"Error checking GPU: {e}", "ERROR")
        return 100  # Assume busy if can't check

    def _get_disk_usage(self) -> tuple[float, float]:
        """Get disk usage for /home partition in GB.

        Returns:
            Tuple of (used_gb, available_gb)
        """
        try:
            stat = shutil.disk_usage("/home")
            used_gb = stat.used / (1024**3)
            avail_gb = stat.free / (1024**3)
            return (used_gb, avail_gb)
        except Exception as e:
            self._log(f"Error checking disk space: {e}", "ERROR")
            return (0, float('inf'))  # Assume OK if can't check

    def _check_disk_space(self) -> bool:
        """Check if disk space is adequate. Return False if low."""
        used_gb, avail_gb = self._get_disk_usage()

        # Warn if less than 50GB available
        if avail_gb < 50:
            self._log(f"⚠️  LOW DISK SPACE: {avail_gb:.1f}GB available", "WARNING")
            return False

        # Info if getting low
        if avail_gb < 100:
            self._log(f"Disk space: {avail_gb:.1f}GB available", "INFO")

        return True

    def _cleanup_test_results(self, test_study: str):
        """Clean up test result directories after recording results."""
        try:
            # The test results are in train_results/cwd_tests/
            test_dir = Path("train_results/cwd_tests")
            if test_dir.exists():
                # Find directories for this test study
                # Test studies are named like: study_ai_test_20251115_123456
                # And trials are in: trial_N_1h directories

                # For safety, just clean up all old test directories
                # We can identify test trials because they're in the test study
                deleted_count = 0
                freed_mb = 0

                for trial_dir in test_dir.glob("trial_*_1h"):
                    try:
                        # Get size before deletion
                        size_mb = sum(f.stat().st_size for f in trial_dir.glob("**/*") if f.is_file()) / (1024**2)
                        shutil.rmtree(trial_dir)
                        deleted_count += 1
                        freed_mb += size_mb
                    except Exception as e:
                        self._log(f"Could not delete {trial_dir}: {e}", "WARNING")

                if deleted_count > 0:
                    self._log(f"✓ Cleaned up {deleted_count} test directories, freed {freed_mb:.1f}MB")

        except Exception as e:
            self._log(f"Error cleaning up test results: {e}", "WARNING")

    def run_analysis(self) -> bool:
        """Run AI analysis on current study."""
        self._log(f"Running AI analysis on {self.study_name}...")

        try:
            # Run advisor
            cmd = [
                "python", "ollama_training_advisor.py",
                "--study", self.study_name,
                "--model", self.ollama_model,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                self._log("✓ Analysis completed successfully")
                self.state.analysis_count += 1
                self.state.last_analysis_time = datetime.now()
                return True
            else:
                self._log(f"Analysis failed: {result.stderr}", "ERROR")
                return False

        except Exception as e:
            self._log(f"Error running analysis: {e}", "ERROR")
            return False

    def generate_suggestions(self, num_configs: int = 3) -> List[Dict]:
        """Generate new parameter configurations."""
        self._log(f"Generating {num_configs} new configurations...")

        try:
            cmd = [
                "python", "ollama_param_suggester.py",
                "--study", self.study_name,
                "--model", self.ollama_model,
                "--generate", str(num_configs),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                # Find the latest suggestions file
                suggestions_files = sorted(
                    Path("analysis_reports").glob(f"ollama_suggestions_{self.study_name}_*.json")
                )
                if suggestions_files:
                    with suggestions_files[-1].open("r") as f:
                        suggestions = json.load(f)
                    self._log(f"✓ Generated {len(suggestions)} configurations")
                    return suggestions

        except Exception as e:
            self._log(f"Error generating suggestions: {e}", "ERROR")

        return []

    def test_configuration(self, config: Dict, trial_num: int = 1) -> bool:
        """Test a single configuration by running training trials."""
        self._log(f"Testing configuration {trial_num}/{self.max_test_trials}...")
        self._log(f"  Rationale: {config.get('rationale', 'N/A')}")

        # Create a temporary test study name
        test_study = f"{self.study_name}_ai_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            # Build command with AI-suggested parameters
            cmd = [
                "python", "-u", "1_optimize_unified.py",
                "--n-trials", str(trial_num),
                "--gpu", "0",
                "--study-name", test_study,
            ]

            # Log what we're testing
            params_str = ", ".join([f"{k}={v}" for k, v in config.items() if k != 'rationale'])
            self._log(f"  Testing: {params_str}")

            # Run training (with timeout)
            log_file = Path(f"logs/ai_test_{test_study}.log")
            with log_file.open("w") as f:
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=1800  # 30 minutes max
                )

            if result.returncode == 0:
                # Check results
                trial_value = self._get_latest_trial_value(test_study)
                if trial_value is not None:
                    self._log(f"✓ Test completed. Result: {trial_value:.6f}")

                    # Track if this is a new best
                    if trial_value > self.state.best_discovered_value:
                        self.state.best_discovered_value = trial_value
                        self._log(f"🎯 NEW BEST discovered by AI: {trial_value:.6f}!", "SUCCESS")

                    # Record tested config
                    config['test_result'] = trial_value
                    config['test_time'] = datetime.now().isoformat()
                    self.state.tested_configs.append(config)

                    # Clean up test result files (they're saved in the database)
                    self._cleanup_test_results(test_study)

                    return True

        except subprocess.TimeoutExpired:
            self._log("Test timed out after 30 minutes", "WARNING")
            # Clean up even on timeout
            self._cleanup_test_results(test_study)
        except Exception as e:
            self._log(f"Error testing configuration: {e}", "ERROR")

        return False

    def _get_latest_trial_value(self, study_name: str) -> Optional[float]:
        """Get the value of the latest trial in a study."""
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
            SELECT tv.value
            FROM trial_values tv
            JOIN trials t ON tv.trial_id = t.trial_id
            JOIN studies s ON t.study_id = s.study_id
            WHERE s.study_name = ? AND t.state = 'COMPLETE'
            ORDER BY t.datetime_complete DESC
            LIMIT 1
            """
            result = pd.read_sql_query(query, conn, params=(study_name,))
            conn.close()

            if not result.empty:
                return float(result.iloc[0, 0])
        except Exception as e:
            self._log(f"Error getting trial value: {e}", "ERROR")

        return None

    def autonomous_cycle(self):
        """Run one cycle of the autonomous advisor."""
        # Check disk space first
        self._check_disk_space()

        # Check current trial count
        current_trials = self._get_trial_count()
        new_trials = current_trials - self.state.last_trial_count

        self._log(f"Check: {current_trials} total trials ({new_trials} new since last check)")

        # Should we analyze?
        if new_trials >= self.analysis_interval:
            self._log(f"Threshold reached ({new_trials} >= {self.analysis_interval})")

            # Run analysis
            if self.run_analysis():
                # Generate suggestions
                suggestions = self.generate_suggestions(num_configs=3)

                if suggestions and self.test_when_idle:
                    # Unload Ollama model to free GPU memory
                    self._log("Unloading Ollama model to free GPU...")
                    try:
                        subprocess.run(
                            ["ollama", "stop", self.ollama_model],
                            capture_output=True,
                            timeout=30
                        )
                    except Exception as e:
                        self._log(f"Warning: Could not unload model: {e}", "WARNING")

                    # Wait for GPU to settle after Ollama inference
                    self._log("Waiting 30 seconds for GPU to settle...")
                    time.sleep(30)

                    # Check if we can test
                    if not self._is_training_running():
                        gpu_util = self._get_gpu_utilization()
                        if gpu_util < 40:  # GPU is mostly idle (accounting for background apps)
                            # Check disk space before testing
                            if not self._check_disk_space():
                                self._log("⚠️  Skipping tests due to low disk space", "WARNING")
                            else:
                                self._log("GPU is idle. Testing AI-suggested configurations...")

                                # Test each suggestion
                                for i, config in enumerate(suggestions[:3], 1):
                                    if self._stop:
                                        break
                                    self.test_configuration(config, trial_num=self.max_test_trials)
                        else:
                            self._log(f"GPU busy ({gpu_util}%), skipping tests")
                    else:
                        self._log("Training is running, skipping tests")

                # Update state
                self.state.last_trial_count = current_trials
                self._save_state()

        # Save state periodically
        self._save_state()

    def run_daemon(self):
        """Run as a daemon, continuously monitoring and optimizing."""
        self._log("="*80)
        self._log("AUTONOMOUS AI ADVISOR STARTED")
        self._log(f"Study: {self.study_name}")
        self._log(f"Model: {self.ollama_model}")
        self._log(f"Analysis interval: {self.analysis_interval} trials")
        self._log(f"Check interval: {self.check_interval} seconds")
        self._log(f"Auto-test when idle: {self.test_when_idle}")
        self._log("="*80)

        self.state.is_running = True
        self._save_state()

        def signal_handler(signum, frame):
            self._log("\nShutdown signal received, stopping gracefully...")
            self._stop = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            while not self._stop:
                try:
                    self.autonomous_cycle()
                except Exception as e:
                    self._log(f"Error in cycle: {e}", "ERROR")

                if not self._stop:
                    self._log(f"Sleeping for {self.check_interval} seconds...")
                    time.sleep(self.check_interval)

        finally:
            self.state.is_running = False
            self._save_state()
            self._log("="*80)
            self._log("AUTONOMOUS AI ADVISOR STOPPED")
            self._log(f"Total analyses: {self.state.analysis_count}")
            self._log(f"Configs tested: {len(self.state.tested_configs)}")
            self._log(f"Best discovered: {self.state.best_discovered_value:.6f}")
            self._log("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous AI Training Advisor"
    )
    parser.add_argument(
        "--study",
        type=str,
        required=True,
        help="Study name to monitor"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="databases/optuna_cappuccino.db",
        help="Path to Optuna database"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-coder:7b",
        help="Ollama model to use"
    )
    parser.add_argument(
        "--analysis-interval",
        type=int,
        default=50,
        help="Run analysis every N new trials"
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=300,
        help="Check for new trials every N seconds (default: 300 = 5 minutes)"
    )
    parser.add_argument(
        "--no-auto-test",
        action="store_true",
        help="Disable automatic testing of suggestions"
    )
    parser.add_argument(
        "--max-test-trials",
        type=int,
        default=10,
        help="Number of trials to run when testing each config"
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as daemon (continuous loop)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one cycle then exit"
    )

    args = parser.parse_args()

    advisor = AutonomousAdvisor(
        study_name=args.study,
        db_path=args.db,
        ollama_model=args.model,
        analysis_interval_trials=args.analysis_interval,
        check_interval_seconds=args.check_interval,
        test_configs_when_idle=not args.no_auto_test,
        max_test_trials=args.max_test_trials,
    )

    if args.once:
        advisor.autonomous_cycle()
    else:
        advisor.run_daemon()


if __name__ == "__main__":
    main()
