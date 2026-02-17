#!/usr/bin/env python3
"""
Pipeline V2 - Simple, Robust Trial Processing

Complete rewrite of the pipeline orchestrator with:
- Single SQLite database for state
- Clear error handling
- Retry logic
- Health monitoring
- Easy debugging
"""

import sqlite3
import time
import json
import logging
import signal
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

# Configuration
MAX_DEPLOYED_TRIALS = 10  # Maximum number of simultaneously deployed trials

# Get Optuna database from environment or use default
DEFAULT_OPTUNA_DB = os.getenv('OPTUNA_DB', 'databases/optuna_cappuccino.db')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PipelineV2:
    """Simple, robust pipeline for processing trials."""

    def __init__(self, db_path: str = "pipeline_v2.db",
                 optuna_db: str = None,
                 config_path: str = "config/pipeline_v2_config.json"):
        """
        Initialize pipeline.

        Args:
            db_path: Path to pipeline state database
            optuna_db: Path to Optuna trials database
            config_path: Path to configuration file
        """
        self.db_path = db_path
        self.optuna_db = optuna_db if optuna_db else DEFAULT_OPTUNA_DB
        self.config = self._load_config(config_path)
        self.running = True

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Initialize database
        self._init_database()

        logger.info("Pipeline V2 initialized")
        logger.info(f"State DB: {db_path}")
        logger.info(f"Optuna DB: {self.optuna_db}")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration or use defaults."""
        default_config = {
            "stages": {
                "backtest": {"enabled": True, "max_attempts": 3, "timeout": 60},
                "cge_stress": {"enabled": False, "max_attempts": 2, "timeout": 300},
                "deploy": {"enabled": True, "max_attempts": 1, "auto_restart": True}
            },
            "daemon": {
                "check_interval": 300,  # 5 minutes
                "health_check_interval": 60
            },
            "retry": {
                "delays": [5, 30, 300]  # Exponential backoff: 5s, 30s, 5min
            }
        }

        try:
            with open(config_path) as f:
                config = json.load(f)
                logger.info(f"Loaded config from {config_path}")
                return config
        except FileNotFoundError:
            logger.warning(f"Config not found, using defaults")
            return default_config

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False

    @contextmanager
    def _db_connection(self, db_path: Optional[str] = None):
        """Context manager for database connections."""
        conn = sqlite3.connect(db_path or self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_database(self):
        """Initialize pipeline state database."""
        with self._db_connection() as conn:
            cursor = conn.cursor()

            # Trials table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trials (
                    trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trial_number INTEGER UNIQUE NOT NULL,
                    value REAL,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Stages table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trial_id INTEGER,
                    stage_name TEXT,
                    status TEXT DEFAULT 'pending',
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    attempts INTEGER DEFAULT 0,
                    max_attempts INTEGER DEFAULT 3,
                    error_message TEXT,
                    results TEXT,
                    FOREIGN KEY (trial_id) REFERENCES trials(trial_id),
                    UNIQUE(trial_id, stage_name)
                )
            ''')

            # Deployments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS deployments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trial_id INTEGER,
                    process_id INTEGER,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    stopped_at TIMESTAMP,
                    status TEXT DEFAULT 'running',
                    log_file TEXT,
                    FOREIGN KEY (trial_id) REFERENCES trials(trial_id)
                )
            ''')

            # Health table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    component TEXT,
                    status TEXT,
                    message TEXT
                )
            ''')

            conn.commit()
            logger.info("Database initialized")

    def discover_trials(self) -> List[int]:
        """
        Discover new completed trials from Optuna database.

        Returns:
            List of trial numbers that need processing
        """
        logger.info("Discovering trials...")

        try:
            with self._db_connection(self.optuna_db) as optuna_conn:
                cursor = optuna_conn.cursor()
                cursor.execute('''
                    SELECT t.number, tv.value
                    FROM trials t
                    LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
                    WHERE t.state = 'COMPLETE'
                    ORDER BY t.number
                ''')

                optuna_trials = {row[0]: row[1] for row in cursor.fetchall()}
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                logger.info("Fresh study - no trials table yet. Waiting for first trial...")
                optuna_trials = {}
            else:
                raise

        logger.info(f"Found {len(optuna_trials)} completed trials in Optuna")

        # Check which ones we haven't processed yet
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT trial_number FROM trials')
            processed = {row[0] for row in cursor.fetchall()}

        new_trials = [num for num in optuna_trials.keys() if num not in processed]

        # Add new trials to our database
        if new_trials:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                for trial_num in new_trials:
                    cursor.execute('''
                        INSERT INTO trials (trial_number, value, status)
                        VALUES (?, ?, 'pending')
                    ''', (trial_num, optuna_trials[trial_num]))

                    trial_id = cursor.lastrowid

                    # Initialize stages
                    for stage_name, stage_config in self.config['stages'].items():
                        if stage_config['enabled']:
                            cursor.execute('''
                                INSERT INTO stages (trial_id, stage_name, status, max_attempts)
                                VALUES (?, ?, 'pending', ?)
                            ''', (trial_id, stage_name, stage_config['max_attempts']))

                conn.commit()

            logger.info(f"Added {len(new_trials)} new trials: {new_trials}")
        else:
            logger.info("No new trials to process")

        return new_trials

    def get_pending_trials(self) -> List[Dict]:
        """Get trials that need processing."""
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT trial_id, trial_number, value, status
                FROM trials
                WHERE status IN ('pending', 'processing')
                ORDER BY trial_number
            ''')

            return [dict(row) for row in cursor.fetchall()]

    def process_trial(self, trial_id: int) -> bool:
        """
        Process a single trial through all enabled stages.

        Args:
            trial_id: Internal trial ID

        Returns:
            True if all stages passed, False otherwise
        """
        # Get trial info
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT trial_number FROM trials WHERE trial_id = ?', (trial_id,))
            row = cursor.fetchone()
            if not row:
                logger.error(f"Trial {trial_id} not found")
                return False
            trial_num = row[0]

        logger.info(f"Processing trial {trial_num} (ID: {trial_id})")

        # Update status to processing
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE trials
                SET status = 'processing', updated_at = CURRENT_TIMESTAMP
                WHERE trial_id = ?
            ''', (trial_id,))
            conn.commit()

        # Get stages to execute
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, stage_name, status, attempts, max_attempts
                FROM stages
                WHERE trial_id = ?
                ORDER BY id
            ''', (trial_id,))

            stages = [dict(row) for row in cursor.fetchall()]

        # Execute each stage
        all_passed = True
        for stage in stages:
            stage_result = self._execute_stage(trial_id, stage)
            if not stage_result:
                all_passed = False
                break  # Stop on first failure

        # Update final trial status
        final_status = 'deployed' if all_passed else 'failed'
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE trials
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE trial_id = ?
            ''', (final_status, trial_id))
            conn.commit()

        logger.info(f"Trial {trial_num} final status: {final_status}")
        return all_passed

    def _execute_stage(self, trial_id: int, stage: Dict) -> bool:
        """
        Execute a single stage with retry logic.

        Args:
            trial_id: Internal trial ID
            stage: Stage info dict

        Returns:
            True if stage passed, False otherwise
        """
        stage_id = stage['id']
        stage_name = stage['stage_name']
        max_attempts = stage['max_attempts']
        current_attempt = stage['attempts']

        # Skip if already passed
        if stage['status'] == 'passed':
            logger.info(f"Stage {stage_name} already passed, skipping")
            return True

        logger.info(f"Executing stage: {stage_name}")

        # Add project root to path for imports
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        # Import stage executors dynamically
        from scripts.optimization.backtest_v2 import BacktestValidator
        from archive.deprecated_scripts.deploy_v2 import PaperTradingDeployer

        executors = {
            'backtest': BacktestValidator(),
            'deploy': PaperTradingDeployer()
        }

        if stage_name not in executors:
            logger.warning(f"No executor for stage: {stage_name}, skipping")
            self._update_stage_status(stage_id, 'skipped')
            return True

        executor = executors[stage_name]

        # Retry loop
        for attempt in range(current_attempt, max_attempts):
            try:
                # Update stage status
                with self._db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE stages
                        SET status = 'running', started_at = CURRENT_TIMESTAMP, attempts = ?
                        WHERE id = ?
                    ''', (attempt + 1, stage_id))
                    conn.commit()

                # Execute stage
                logger.info(f"Stage {stage_name} attempt {attempt + 1}/{max_attempts}")

                # Get trial number for executor
                with self._db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT trial_number FROM trials WHERE trial_id = ?', (trial_id,))
                    trial_num = cursor.fetchone()[0]

                # Execute based on stage type
                if stage_name == 'backtest':
                    result = executor.validate(trial_num)
                elif stage_name == 'deploy':
                    # Check deployment limit before deploying
                    if not self._check_deployment_limit(trial_num):
                        logger.warning(f"Deployment limit ({MAX_DEPLOYED_TRIALS}) reached, skipping deployment of trial {trial_num}")
                        result = {'success': False, 'error': f'Deployment limit ({MAX_DEPLOYED_TRIALS}) reached'}
                    else:
                        result = executor.deploy(trial_num)
                else:
                    result = {'success': True}

                # Check result
                if result.get('success', False):
                    logger.info(f"Stage {stage_name} PASSED")
                    self._update_stage_status(stage_id, 'passed', results=result)
                    return True
                else:
                    error_msg = result.get('error', 'Unknown error')
                    logger.warning(f"Stage {stage_name} failed: {error_msg}")

                    # Retry with backoff
                    if attempt < max_attempts - 1:
                        delay = self.config['retry']['delays'][min(attempt, len(self.config['retry']['delays']) - 1)]
                        logger.info(f"Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        self._update_stage_status(stage_id, 'failed', error_msg=error_msg)
                        return False

            except Exception as e:
                logger.error(f"Stage {stage_name} exception: {str(e)}", exc_info=True)
                if attempt < max_attempts - 1:
                    delay = self.config['retry']['delays'][min(attempt, len(self.config['retry']['delays']) - 1)]
                    logger.info(f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    self._update_stage_status(stage_id, 'failed', error_msg=str(e))
                    return False

        return False

    def _check_deployment_limit(self, new_trial_num: int) -> bool:
        """
        Check if we can deploy a new trial without exceeding MAX_DEPLOYED_TRIALS.

        If limit is reached, stop the worst performing deployed trial to make room.

        Args:
            new_trial_num: Trial number that wants to be deployed

        Returns:
            True if deployment can proceed, False otherwise
        """
        # Count currently deployed trials
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trials WHERE status = 'deployed'")
            deployed_count = cursor.fetchone()[0]

        logger.info(f"Currently deployed: {deployed_count}/{MAX_DEPLOYED_TRIALS}")

        # If under limit, allow deployment
        if deployed_count < MAX_DEPLOYED_TRIALS:
            return True

        # At limit - need to stop worst performer to make room
        logger.info(f"Deployment limit reached, finding worst performer to replace")

        # Get value of new trial from Optuna
        try:
            import optuna
            storage = f"sqlite:///{self.optuna_db}"

            # Find study containing this trial
            conn = sqlite3.connect(self.optuna_db)
            cursor = conn.cursor()
            cursor.execute("SELECT study_name FROM studies")
            study_names = [row[0] for row in cursor.fetchall()]
            conn.close()

            new_trial_value = None
            for study_name in study_names:
                try:
                    study = optuna.load_study(study_name=study_name, storage=storage)
                    for trial in study.trials:
                        if trial.number == new_trial_num and trial.state == optuna.trial.TrialState.COMPLETE:
                            new_trial_value = trial.value
                            break
                    if new_trial_value is not None:
                        break
                except:
                    continue

            if new_trial_value is None:
                logger.warning(f"Could not find value for trial {new_trial_num}, skipping deployment")
                return False

        except Exception as e:
            logger.error(f"Error loading trial value: {e}")
            return False

        # Get all deployed trials with their values
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT trial_number, value FROM trials WHERE status = 'deployed' ORDER BY value ASC")
            deployed_trials = cursor.fetchall()

        if not deployed_trials:
            return True

        # Find worst performer
        worst_trial_num, worst_value = deployed_trials[0]

        # Only replace if new trial is better
        if new_trial_value > worst_value:
            logger.info(f"Trial {new_trial_num} (value={new_trial_value:.6f}) is better than worst trial {worst_trial_num} (value={worst_value:.6f})")
            logger.info(f"Stopping trial {worst_trial_num} to make room")

            # Stop the worst performer
            self._stop_deployed_trial(worst_trial_num)

            return True
        else:
            logger.info(f"Trial {new_trial_num} (value={new_trial_value:.6f}) is not better than worst deployed trial {worst_trial_num} (value={worst_value:.6f})")
            logger.info(f"Skipping deployment")
            return False

    def _stop_deployed_trial(self, trial_num: int):
        """Stop a deployed trial's paper trader process."""
        try:
            import psutil

            # Find process by searching for trial number in command line
            for proc in psutil.process_iter(['pid', 'cmdline']):
                cmdline = proc.info.get('cmdline', [])
                if cmdline and 'paper_trader_alpaca_polling.py' in ' '.join(cmdline):
                    if f'trial_{trial_num}_' in ' '.join(cmdline):
                        logger.info(f"Stopping paper trader for trial {trial_num} (PID {proc.info['pid']})")
                        proc.terminate()
                        proc.wait(timeout=10)

                        # Update trial status to stopped (not deployed)
                        with self._db_connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute("""
                                UPDATE trials
                                SET status = 'completed', updated_at = CURRENT_TIMESTAMP
                                WHERE trial_number = ?
                            """, (trial_num,))
                            conn.commit()

                        logger.info(f"Successfully stopped trial {trial_num}")
                        return True

        except Exception as e:
            logger.error(f"Error stopping trial {trial_num}: {e}")

        return False

    def _update_stage_status(self, stage_id: int, status: str,
                            results: Optional[Dict] = None,
                            error_msg: Optional[str] = None):
        """Update stage status in database."""
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE stages
                SET status = ?,
                    completed_at = CURRENT_TIMESTAMP,
                    results = ?,
                    error_message = ?
                WHERE id = ?
            ''', (status, json.dumps(results) if results else None, error_msg, stage_id))
            conn.commit()

    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        with self._db_connection() as conn:
            cursor = conn.cursor()

            # Count trials by status
            cursor.execute('''
                SELECT status, COUNT(*) as count
                FROM trials
                GROUP BY status
            ''')

            status_counts = {row[0]: row[1] for row in cursor.fetchall()}

            # Get recent activity
            cursor.execute('''
                SELECT trial_number, status, updated_at
                FROM trials
                ORDER BY updated_at DESC
                LIMIT 5
            ''')

            recent = [dict(row) for row in cursor.fetchall()]

        return {
            'pipeline_status': 'running' if self.running else 'stopped',
            'trials_pending': status_counts.get('pending', 0),
            'trials_processing': status_counts.get('processing', 0),
            'trials_deployed': status_counts.get('deployed', 0),
            'trials_failed': status_counts.get('failed', 0),
            'recent_activity': recent,
            'timestamp': datetime.now().isoformat()
        }

    def run_daemon(self):
        """Run pipeline in daemon mode."""
        logger.info("Starting pipeline daemon")
        logger.info(f"Check interval: {self.config['daemon']['check_interval']}s")

        while self.running:
            try:
                # Discover new trials
                self.discover_trials()

                # Process pending trials
                pending = self.get_pending_trials()

                if pending:
                    logger.info(f"Processing {len(pending)} trials")
                    for trial in pending:
                        if not self.running:
                            break
                        self.process_trial(trial['trial_id'])
                else:
                    logger.info("No trials to process")

                # Log status
                status = self.get_status()
                logger.info(f"Status: {status['trials_deployed']} deployed, "
                          f"{status['trials_pending']} pending, "
                          f"{status['trials_failed']} failed")

                # Sleep until next check
                if self.running:
                    logger.info(f"Sleeping for {self.config['daemon']['check_interval']}s...")
                    time.sleep(self.config['daemon']['check_interval'])

            except Exception as e:
                logger.error(f"Daemon error: {str(e)}", exc_info=True)
                time.sleep(60)  # Wait before retrying

        logger.info("Pipeline daemon stopped")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Pipeline V2')
    parser.add_argument('--daemon', action='store_true', help='Run in daemon mode')
    parser.add_argument('--status', action='store_true', help='Show status')
    parser.add_argument('--process-all', action='store_true', help='Process all pending trials once')
    args = parser.parse_args()

    # Create logs directory
    Path('logs').mkdir(exist_ok=True)

    pipeline = PipelineV2()

    if args.status:
        status = pipeline.get_status()
        print(json.dumps(status, indent=2))
    elif args.process_all:
        pipeline.discover_trials()
        pending = pipeline.get_pending_trials()
        logger.info(f"Processing {len(pending)} trials")
        for trial in pending:
            pipeline.process_trial(trial['trial_id'])
    elif args.daemon:
        pipeline.run_daemon()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
