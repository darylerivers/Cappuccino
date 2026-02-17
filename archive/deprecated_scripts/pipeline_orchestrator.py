#!/usr/bin/env python3
"""
Pipeline Orchestrator
Main entry point for automated trading pipeline.

Orchestrates: Training → Backtesting → CGE Stress → Paper Trading → Live Trading
"""

import argparse
import json
import logging
import os
import signal
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Load environment
from dotenv import load_dotenv
load_dotenv('.env.training')

# Import pipeline components
from pipeline.state_manager import PipelineStateManager
from pipeline.gates import BacktestGate, CGEStressGate, PaperTradingGate
from pipeline.backtest_runner import BacktestRunner
from pipeline.cge_runner import CGEStressRunner
from pipeline.notifications import PipelineNotifier


class PipelineOrchestrator:
    """Main orchestrator for automated trading pipeline."""

    def __init__(self, config_path: str = "config/pipeline_config.json",
                 dry_run: bool = False, daemon: bool = False):
        self.config = self._load_config(config_path)
        self.dry_run = dry_run or self.config["pipeline"].get("dry_run", False)
        self.daemon = daemon or self.config["pipeline"].get("daemon_mode", False)

        # Components
        self.state_manager = PipelineStateManager()
        self.backtest_runner = BacktestRunner(config=self.config)
        self.cge_runner = CGEStressRunner(
            num_scenarios=self.config["gates"]["cge_stress"].get("num_scenarios", 200)
        )
        self.notifier = PipelineNotifier(self.config["notifications"])

        # Gates
        db_path = self.config["database"].get("optuna_db_path", "databases/optuna_cappuccino.db")
        self.backtest_gate = BacktestGate(self.config["gates"]["backtest"], db_path=db_path)
        self.cge_gate = CGEStressGate(self.config["gates"]["cge_stress"])
        self.paper_gate = PaperTradingGate(self.config["gates"]["paper_trading"])

        # Database
        self.db_path = db_path
        self.study_name = self.config["database"].get("study_name") or os.getenv('ACTIVE_STUDY_NAME')

        # State
        self.running = True

        # Setup logging
        self._setup_logging()

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.logger.info("Pipeline Orchestrator initialized")
        if self.dry_run:
            self.logger.warning("DRY RUN MODE - No actual deployments will occur")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load config from {config_path}: {e}")
            sys.exit(1)

    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        log_format = '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('logs/pipeline_orchestrator.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def get_new_best_trials(self) -> List[Dict]:
        """Get new best trials from Optuna that aren't in pipeline yet."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get top 100 trials (more than default 20) since many may be cleaned up
            query = """
                SELECT t.trial_id, t.number, tv.value, t.datetime_complete
                FROM trials t
                JOIN studies s ON t.study_id = s.study_id
                JOIN trial_values tv ON t.trial_id = tv.trial_id
                WHERE s.study_name = ?
                AND t.state = 'COMPLETE'
                ORDER BY tv.value DESC
                LIMIT 100
            """

            cursor.execute(query, (self.study_name,))
            results = cursor.fetchall()
            conn.close()

            # Filter out trials already in pipeline
            new_trials = []
            for trial_id, trial_num, value, dt in results:
                if not self.state_manager.is_trial_in_pipeline(trial_num):
                    new_trials.append({
                        "trial_id": trial_id,
                        "trial_number": trial_num,
                        "value": value,
                        "datetime_complete": dt
                    })

            return new_trials

        except Exception as e:
            self.logger.error(f"Failed to get best trials: {e}", exc_info=True)
            return []

    def process_backtest_stage(self):
        """Process trials pending backtest."""
        trials = self.state_manager.get_trials_pending_backtest()

        for trial in trials:
            trial_num = trial["trial_number"]

            self.logger.info(f"Processing backtest for trial {trial_num}")

            # Run backtest
            retries = self.config["retry"]["backtest_max_retries"]
            metrics = self.backtest_runner.run(trial_num, retries=retries)

            if metrics is None:
                self.logger.error(f"Backtest failed for trial {trial_num}")
                self.state_manager.update_stage(
                    trial_num, "backtest", "failed",
                    error="Backtest execution failed"
                )
                self.notifier.gate_failed(trial_num, "backtest", "Execution failed")
                continue

            # Validate against gate
            passed, error = self.backtest_gate.validate(trial_num, metrics)

            if passed:
                self.logger.info(f"Trial {trial_num} PASSED backtest gate")
                self.state_manager.update_stage(
                    trial_num, "backtest", "passed", metrics=metrics
                )
                self.notifier.gate_passed(trial_num, "backtest", metrics)
            else:
                self.logger.warning(f"Trial {trial_num} FAILED backtest gate: {error}")
                self.state_manager.update_stage(
                    trial_num, "backtest", "failed", error=error
                )
                self.notifier.gate_failed(trial_num, "backtest", error)

    def process_cge_stage(self):
        """Process trials pending CGE stress test."""
        trials = self.state_manager.get_trials_pending_cge()

        for trial in trials:
            trial_num = trial["trial_number"]

            self.logger.info(f"Processing CGE stress test for trial {trial_num}")

            # Run CGE stress test
            retries = self.config["retry"]["cge_max_retries"]
            metrics = self.cge_runner.run(trial_num, retries=retries)

            if metrics is None:
                self.logger.error(f"CGE stress test failed for trial {trial_num}")
                self.state_manager.update_stage(
                    trial_num, "cge_stress", "failed",
                    error="CGE stress test execution failed"
                )
                self.notifier.gate_failed(trial_num, "cge_stress", "Execution failed")
                continue

            # Validate against gate
            passed, error = self.cge_gate.validate(trial_num, metrics)

            if passed:
                self.logger.info(f"Trial {trial_num} PASSED CGE stress gate")
                self.state_manager.update_stage(
                    trial_num, "cge_stress", "passed", metrics=metrics
                )
                self.notifier.gate_passed(trial_num, "cge_stress", metrics)
            else:
                self.logger.warning(f"Trial {trial_num} FAILED CGE stress gate: {error}")
                self.state_manager.update_stage(
                    trial_num, "cge_stress", "failed", error=error
                )
                self.notifier.gate_failed(trial_num, "cge_stress", error)

    def process_paper_trading_stage(self):
        """Process trials ready for paper trading deployment."""
        trials = self.state_manager.get_trials_pending_paper()

        if not trials:
            return

        # Only deploy one trial at a time to paper trading
        trial = trials[0]
        trial_num = trial["trial_number"]

        self.logger.info(f"Deploying trial {trial_num} to paper trading")

        if self.dry_run:
            self.logger.info(f"DRY RUN: Would deploy trial {trial_num} to paper trading")
            self.state_manager.update_stage(
                trial_num, "paper_trading", "deployed",
                metrics={"dry_run": True}
            )
            return

        try:
            # Deploy to paper trading via auto_model_deployer
            from auto_model_deployer import AutoModelDeployer

            deployer = AutoModelDeployer(
                study_name=self.study_name,
                db_path=self.db_path,
                arena_mode=True,  # Use arena mode
                auto_deploy=True
            )

            # Add to arena
            model_dir = self.backtest_runner._find_model_dir(trial_num)
            if model_dir:
                success = deployer.add_to_arena(trial_num, model_dir, trial["value"])

                if success:
                    self.logger.info(f"Trial {trial_num} added to Model Arena")
                    self.state_manager.update_stage(
                        trial_num, "paper_trading", "deployed",
                        metrics={"deployed_at": datetime.now().isoformat()}
                    )
                    self.notifier.deployed_to_paper(trial_num)
                else:
                    self.logger.warning(f"Failed to add trial {trial_num} to arena")
                    self.state_manager.update_stage(
                        trial_num, "paper_trading", "failed",
                        error="Failed to add to arena"
                    )
            else:
                self.logger.error(f"Model directory not found for trial {trial_num}")
                self.state_manager.update_stage(
                    trial_num, "paper_trading", "failed",
                    error="Model directory not found"
                )

        except Exception as e:
            self.logger.error(f"Failed to deploy trial {trial_num}: {e}", exc_info=True)
            self.state_manager.update_stage(
                trial_num, "paper_trading", "failed",
                error=str(e)
            )
            self.notifier.error_occurred(trial_num, str(e))

    def check_new_trials(self):
        """Check for new best trials and add to pipeline."""
        # Check if manual trials mode is enabled
        use_manual = self.config["pipeline"].get("use_manual_trials", False)
        manual_trials = self.config["pipeline"].get("manual_trials", [])

        if use_manual and manual_trials:
            self.logger.info(f"Using manual trial list: {manual_trials}")
            trials_to_process = []

            for trial_num in manual_trials:
                if not self.state_manager.is_trial_in_pipeline(trial_num):
                    # Get value from database
                    value = self._get_trial_value(trial_num)
                    if value is not None:
                        trials_to_process.append({
                            "trial_number": trial_num,
                            "value": value
                        })
        else:
            trials_to_process = self.get_new_best_trials()

        if trials_to_process:
            self.logger.info(f"Found {len(trials_to_process)} trials to process")

            for trial in trials_to_process:
                trial_num = trial["trial_number"]
                value = trial["value"]

                # Check if model directory exists before adding
                model_dir = self.backtest_runner._find_model_dir(trial_num)
                if not model_dir:
                    self.logger.warning(f"Skipping trial {trial_num} - model directory not found")
                    continue

                self.logger.info(f"Adding trial {trial_num} (value: {value:.6f}) to pipeline")
                self.state_manager.add_trial(trial_num, value)

    def _get_trial_value(self, trial_number: int) -> Optional[float]:
        """Get trial value from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT tv.value FROM trials t
                JOIN trial_values tv ON t.trial_id = tv.trial_id
                WHERE t.number = ?
            """, (trial_number,))
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else None
        except Exception as e:
            self.logger.error(f"Error getting trial value: {e}")
            return None

    def check_emergency_stop(self) -> bool:
        """Check if emergency stop file exists."""
        emergency_stop_file = Path("deployments/pipeline_emergency_stop")
        return emergency_stop_file.exists()

    def run_once(self):
        """Run one iteration of the pipeline."""
        self.logger.info("=" * 80)
        self.logger.info("Pipeline check starting")
        self.logger.info("=" * 80)

        # Check emergency stop
        if self.check_emergency_stop():
            self.logger.warning("EMERGENCY STOP FILE DETECTED - Pipeline halted")
            return

        # Check for new trials
        self.check_new_trials()

        # Process each stage
        self.process_backtest_stage()
        self.process_cge_stage()
        self.process_paper_trading_stage()

        # Cleanup old trials
        self.state_manager.cleanup_old_trials(keep_last_n=100)

        self.logger.info("Pipeline check completed")

    def run(self):
        """Main loop."""
        self.logger.info("=" * 80)
        self.logger.info("PIPELINE ORCHESTRATOR STARTED")
        self.logger.info("=" * 80)
        self.logger.info(f"Study: {self.study_name}")
        self.logger.info(f"Database: {self.db_path}")
        self.logger.info(f"Dry run: {self.dry_run}")
        self.logger.info(f"Daemon mode: {self.daemon}")
        self.logger.info("=" * 80)

        # Initial run
        self.run_once()

        if not self.daemon:
            self.logger.info("Single run mode, exiting")
            return

        # Daemon loop
        check_interval = self.config["pipeline"]["check_interval_seconds"]

        while self.running:
            try:
                self.logger.info(f"Sleeping for {check_interval}s...")
                time.sleep(check_interval)

                if self.running:
                    self.run_once()

            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(60)

        self.logger.info("Pipeline orchestrator stopped")

    def print_status(self):
        """Print pipeline status."""
        all_trials = self.state_manager.get_all_trials()

        print("\n" + "=" * 80)
        print("PIPELINE STATUS")
        print("=" * 80)
        print(f"Total trials in pipeline: {len(all_trials)}")
        print()

        # Count by stage
        stage_counts = {}
        for trial in all_trials.values():
            stage = trial.get("current_stage", "unknown")
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

        print("Trials by stage:")
        for stage, count in sorted(stage_counts.items()):
            print(f"  {stage}: {count}")
        print()

        # Recent trials
        recent = sorted(
            all_trials.values(),
            key=lambda x: x.get("discovered_at", ""),
            reverse=True
        )[:10]

        print("Recent trials:")
        for trial in recent:
            num = trial["trial_number"]
            stage = trial["current_stage"]
            value = trial["value"]
            print(f"  Trial {num}: stage={stage}, value={value:.6f}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Pipeline Orchestrator")
    parser.add_argument("--config", default="config/pipeline_config.json",
                        help="Configuration file path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Dry run mode (no actual deployments)")
    parser.add_argument("--daemon", action="store_true",
                        help="Run in daemon mode")
    parser.add_argument("--once", action="store_true",
                        help="Run once and exit (overrides daemon)")
    parser.add_argument("--status", action="store_true",
                        help="Print status and exit")

    args = parser.parse_args()

    # Create orchestrator
    orchestrator = PipelineOrchestrator(
        config_path=args.config,
        dry_run=args.dry_run,
        daemon=args.daemon and not args.once
    )

    # Run based on mode
    if args.status:
        orchestrator.print_status()
    elif args.once:
        orchestrator.run_once()
    else:
        orchestrator.run()


if __name__ == "__main__":
    main()
