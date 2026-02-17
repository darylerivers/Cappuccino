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
    python auto_model_deployer.py --study cappuccino_week_20251206 --check-interval 3600
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

# Add parent directory to path for imports
PARENT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PARENT_DIR))

import optuna
from utils.path_detector import PathDetector

# Load active study from environment
from dotenv import load_dotenv
load_dotenv('.env.training')
_DEFAULT_STUDY = os.getenv('ACTIVE_STUDY_NAME', 'cappuccino_week_20251206')


class AutoModelDeployer:
    def __init__(
        self,
        study_name: str,
        db_path: str = None,
        check_interval: int = 3600,
        min_improvement: float = 0.01,
        validation_enabled: bool = True,
        auto_deploy: bool = True,
        daemon: bool = False,
        arena_mode: bool = False,  # NEW: Use arena for model evaluation
    ):
        self.study_name = study_name
        # Auto-detect database path if not provided
        if db_path is None:
            detector = PathDetector()
            self.db_path = detector.find_optuna_db()
        else:
            self.db_path = db_path
        self.check_interval = check_interval
        self.min_improvement = min_improvement
        self.validation_enabled = validation_enabled
        self.auto_deploy = auto_deploy
        self.daemon = daemon
        self.arena_mode = arena_mode

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
        # Auto-detect log directory
        detector = PathDetector()
        log_dir = detector.find_log_dir()

        log_format = '[%(asctime)s] [%(levelname)s] %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(f'{log_dir}/auto_deployer.log'),
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
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            query = """
                SELECT t.trial_id, tv.value, t.datetime_complete, t.number
                FROM trials t
                JOIN studies s ON t.study_id = s.study_id
                JOIN trial_values tv ON t.trial_id = tv.trial_id
                WHERE s.study_name = ?
                AND t.state = 'COMPLETE'
                ORDER BY tv.value DESC
                LIMIT ?
            """

            cursor.execute(query, (self.study_name, top_n))
            results = cursor.fetchall()
            conn.close()

            return [(trial_id, value, dt, num) for trial_id, value, dt, num in results]
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                self.logger.info("Fresh study - no trials table yet. Waiting for first trial...")
                return []
            else:
                raise

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

    def check_model_exists(self, trial_num: int) -> Tuple[bool, Optional[Path]]:
        """Check if model files exist for a trial.

        Note: Uses trial number (not trial_id) since directories are named trial_XX_1h.
        """
        # Try different possible paths using trial number
        possible_paths = [
            Path(f"train_results/cwd_tests/trial_{trial_num}_1h"),
            Path(f"train_results/cwd_tests/trial_{trial_num}"),
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
        Validate model performance with backtesting.

        Validation stages:
        1. File existence check
        2. Backtest on validation data
        3. Performance thresholds (adaptive based on trial count)
        """
        if not self.validation_enabled:
            return True

        # Stage 1: Check if model files exist
        required_files = ["actor.pth", "critic.pth"]
        for f in required_files:
            if not (model_dir / f).exists():
                self.logger.warning(f"Missing file {f} in {model_dir}")
                return False

        # Stage 2: Run backtest
        backtest_result = self._run_backtest(trial_id, model_dir)

        if backtest_result is None:
            self.logger.warning(f"Backtest failed for trial {trial_id}")
            return False

        # Stage 3: Check performance thresholds
        total_return = backtest_result.get('total_return', 0)
        sharpe = backtest_result.get('sharpe', 0)
        max_drawdown = backtest_result.get('max_drawdown', 1)
        vs_benchmark = backtest_result.get('vs_benchmark', 0)

        # Get adaptive thresholds based on how many trials we have
        thresholds = self._get_adaptive_thresholds()

        self.logger.info(f"Backtest results for trial {trial_id}:")
        self.logger.info(f"  Total return: {total_return*100:.2f}%")
        self.logger.info(f"  Sharpe ratio: {sharpe:.3f}")
        self.logger.info(f"  Max drawdown: {max_drawdown*100:.2f}%")
        self.logger.info(f"  vs Benchmark: {vs_benchmark*100:.2f}%")
        self.logger.info(f"  Thresholds: min_return={thresholds['min_return']*100:.1f}%, min_sharpe={thresholds['min_sharpe']:.2f}")

        # Check thresholds
        if total_return < thresholds['min_return']:
            self.logger.warning(f"Total return {total_return*100:.2f}% below threshold {thresholds['min_return']*100:.1f}%")
            return False

        if sharpe < thresholds['min_sharpe']:
            self.logger.warning(f"Sharpe {sharpe:.3f} below threshold {thresholds['min_sharpe']:.2f}")
            return False

        if max_drawdown > thresholds['max_drawdown']:
            self.logger.warning(f"Max drawdown {max_drawdown*100:.2f}% exceeds threshold {thresholds['max_drawdown']*100:.1f}%")
            return False

        self.logger.info(f"Model validation PASSED for trial {trial_id}")
        return True

    def _get_adaptive_thresholds(self) -> dict:
        """
        Get adaptive thresholds based on training progress.
        Early in training, be more lenient. As more trials complete, raise the bar.
        """
        # Count completed trials
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM trials t
                JOIN studies s ON t.study_id = s.study_id
                WHERE s.study_name = ? AND t.state = 'COMPLETE'
            """, (self.study_name,))
            trial_count = cursor.fetchone()[0]
            conn.close()
        except:
            trial_count = 0

        # Define threshold phases
        if trial_count < 50:
            # Early phase: Very lenient - just deploy anything that doesn't crash
            return {
                'min_return': -0.50,  # Accept up to 50% loss
                'min_sharpe': -2.0,   # Very negative sharpe OK
                'max_drawdown': 0.80,  # 80% drawdown OK
            }
        elif trial_count < 200:
            # Middle phase: Moderate thresholds
            return {
                'min_return': -0.20,  # Accept up to 20% loss
                'min_sharpe': -0.5,   # Slightly negative OK
                'max_drawdown': 0.50,  # 50% drawdown max
            }
        elif trial_count < 500:
            # Late phase: Stricter
            return {
                'min_return': -0.10,  # Max 10% loss
                'min_sharpe': 0.0,    # Must be positive
                'max_drawdown': 0.35,  # 35% drawdown max
            }
        else:
            # Mature phase: High standards
            return {
                'min_return': 0.0,    # Must be profitable
                'min_sharpe': 0.3,    # Decent risk-adjusted return
                'max_drawdown': 0.25,  # 25% drawdown max
            }

    def _run_backtest(self, trial_id: int, model_dir: Path) -> Optional[dict]:
        """
        Run a quick backtest on validation data.
        Returns dict with performance metrics or None if failed.
        """
        try:
            import numpy as np
            import torch
            from environment_Alpaca import CryptoEnvAlpaca

            # Load validation data (pickled files without extension)
            val_dir = Path("data/1h_1680/val")
            if not val_dir.exists():
                val_dir = Path("data/1h_1680")  # Fallback to full data

            # Load pickled arrays (same format as training uses)
            with open(val_dir / "price_array", 'rb') as f:
                price_array = pickle.load(f)
            with open(val_dir / "tech_array", 'rb') as f:
                tech_array = pickle.load(f)

            # Get trial parameters from database
            params = self._get_trial_params(trial_id)
            if not params:
                self.logger.warning(f"Could not load params for trial {trial_id}")
                return None

            # Create environment
            env_config = {
                "price_array": price_array,
                "tech_array": tech_array,
                "if_train": False,
            }

            env_params = {
                "lookback": int(params.get('lookback', 60)),
                "norm_cash": 2 ** float(params.get('norm_cash_exp', -11)),
                "norm_stocks": 2 ** float(params.get('norm_stocks_exp', -8)),
                "norm_tech": 2 ** float(params.get('norm_tech_exp', -14)),
                "norm_reward": 2 ** float(params.get('norm_reward_exp', -9)),
                "norm_action": float(params.get('norm_action', 100)),
                "time_decay_floor": float(params.get('time_decay_floor', 0.0)),
                "min_cash_reserve": float(params.get('min_cash_reserve', 0.1)),
                "concentration_penalty": float(params.get('concentration_penalty', 0.05)),
            }

            env = CryptoEnvAlpaca(env_config, env_params, if_log=False)

            # Load model using init_agent (same as paper trader)
            from train.run import init_agent
            from train.config import Arguments
            from drl_agents.elegantrl_models import MODELS

            net_dim = int(params.get('net_dimension', 256))
            model_name = "ppo"  # Default model type

            args = Arguments(agent=MODELS[model_name], env=env)
            args.cwd = str(model_dir)
            args.if_remove = False
            args.net_dim = net_dim

            agent = init_agent(args, gpu_id=-1, env=env)  # -1 for CPU
            agent.act.eval()

            # Run backtest
            state = env.reset()
            done = False
            portfolio_values = [env.initial_total_asset]

            while not done:
                state_tensor = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    action = agent.act(state_tensor).cpu().numpy()[0]
                state, reward, done, _ = env.step(action)
                portfolio_values.append(env.total_asset)

            # Calculate metrics
            portfolio_values = np.array(portfolio_values)
            returns = np.diff(portfolio_values) / portfolio_values[:-1]

            total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(24 * 365)  # Annualized

            # Max drawdown
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (peak - portfolio_values) / peak
            max_drawdown = np.max(drawdown)

            # vs buy-and-hold benchmark
            initial_prices = price_array[0]
            final_prices = price_array[-1]
            benchmark_return = np.mean((final_prices - initial_prices) / initial_prices)
            vs_benchmark = total_return - benchmark_return

            return {
                'total_return': total_return,
                'sharpe': sharpe,
                'max_drawdown': max_drawdown,
                'vs_benchmark': vs_benchmark,
                'final_value': portfolio_values[-1],
            }

        except Exception as e:
            self.logger.error(f"Backtest error: {e}", exc_info=True)
            return None

    def _get_trial_params(self, trial_id: int) -> Optional[dict]:
        """Get trial parameters from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT param_name, param_value
                FROM trial_params
                WHERE trial_id = ?
            """, (trial_id,))

            params = {}
            for name, value in cursor.fetchall():
                params[name] = value
            conn.close()
            return params
        except Exception as e:
            self.logger.error(f"Error loading params: {e}")
            return None

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

            # Get list of all studies from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT study_name FROM studies")
            study_names = [row[0] for row in cursor.fetchall()]
            conn.close()

            # Find the trial in any study
            trial_obj = None
            for study_name in study_names:
                try:
                    study = optuna.load_study(study_name=study_name, storage=storage)

                    # Check if this study has the trial
                    for trial in study.trials:
                        if trial.number == trial_id and trial.state == optuna.trial.TrialState.COMPLETE:
                            trial_obj = trial
                            self.logger.info(f"Found trial {trial_id} in study '{study_name}'")
                            break

                    if trial_obj:
                        break
                except Exception as e:
                    self.logger.debug(f"Could not load study {study_name}: {e}")
                    continue

            if trial_obj:
                # Save the frozen trial (no need to add name_folder, paper trader uses model_dir directly)
                with open(best_trial_path, 'wb') as f:
                    pickle.dump(trial_obj, f)
                self.logger.info(f"Created best_trial file for trial {trial_id}")
            else:
                self.logger.warning(f"Could not find completed trial {trial_id} in any study")

        except Exception as e:
            self.logger.warning(f"Failed to create best_trial file: {e}")

    def ensure_stored_agent_directory(self, model_dir: Path):
        """
        Ensure stored_agent directory exists with model weights.
        Paper trader requires weights to be in model_dir/stored_agent/
        """
        stored_agent_dir = model_dir / "stored_agent"

        # If it already exists, verify it has all files
        if stored_agent_dir.exists():
            weight_files = list(stored_agent_dir.glob("*.pth"))
            if len(weight_files) >= 2:  # At least actor.pth and critic.pth
                self.logger.info(f"stored_agent directory already exists with {len(weight_files)} files")
                return

        # Create directory
        stored_agent_dir.mkdir(exist_ok=True)
        self.logger.info(f"Creating stored_agent directory: {stored_agent_dir}")

        # Copy all .pth files from parent directory
        weight_files = list(model_dir.glob("*.pth"))
        if not weight_files:
            self.logger.warning(f"No .pth files found in {model_dir}")
            return

        copied_count = 0
        for weight_file in weight_files:
            dest = stored_agent_dir / weight_file.name
            shutil.copy2(weight_file, dest)
            copied_count += 1

        self.logger.info(f"Copied {copied_count} weight files to stored_agent/")

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

        # Ensure stored_agent directory exists with weights
        self.ensure_stored_agent_directory(model_dir)

        # Get trial parameters to extract risk management settings
        trial_params = self._get_trial_params(trial_id)
        trailing_stop_pct = trial_params.get('trailing_stop_pct', 0.0) if trial_params else 0.0

        # Stop current paper trader
        self.stop_current_paper_trader()
        time.sleep(2)

        # Start new paper trader
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/paper_trading_auto_{timestamp}.log"
        csv_file = f"paper_trades/auto_session_{timestamp}.csv"

        # ALWAYS use ensemble (top N models with voting) - NOT individual trials
        # This ensures ensemble_votes.json is written and dashboard stays updated
        ensemble_model_dir = "train_results/ensemble"
        self.logger.info(f"Deploying ensemble (triggered by trial {trial_id} with value {value:.6f})")

        cmd = [
            "nohup", "python", "-u", "paper_trader_alpaca_polling.py",
            "--model-dir", ensemble_model_dir,
            "--tickers", "BTC/USD", "ETH/USD", "LTC/USD", "BCH/USD", "LINK/USD", "UNI/USD", "AAVE/USD",
            "--timeframe", "1h",
            "--history-hours", "120",
            "--poll-interval", "60",
            "--gpu", "-1",
            "--log-file", csv_file,
            # Risk management - use trial's trailing stop if available
            "--max-position-pct", "0.30",  # Max 30% in single asset
            "--stop-loss-pct", "0.10",  # 10% stop-loss
            "--trailing-stop-pct", str(trailing_stop_pct),  # Use trial's trailing stop
        ]

        if trailing_stop_pct > 0:
            self.logger.info(f"Deploying with trailing stop: {trailing_stop_pct*100:.1f}%")

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
            "model_dir": ensemble_model_dir,  # Always ensemble now
            "triggered_by_trial": str(model_dir),  # Log which trial triggered it
            "log_file": log_file,
            "csv_file": csv_file,
        })

        # Keep only last 20 deployments in history
        if len(self.state["deployment_history"]) > 20:
            self.state["deployment_history"] = self.state["deployment_history"][-20:]

        self._save_state()
        self._log_deployment(trial_id, value, "DEPLOYED", f"PID: {proc.pid}, Log: {log_file}")

    def deploy_single_model(self, trial_num: int, model_dir: Path, value: float):
        """Deploy a SINGLE model to paper trading (not ensemble).

        Used by arena mode when promoting the best performer.
        """
        self.logger.info(f"Deploying SINGLE model trial_{trial_num} (value: {value:.6f})")

        # Create best_trial file if needed
        trial_id = self._get_trial_id_from_number(trial_num)
        if trial_id:
            self.create_best_trial_file(trial_id, model_dir)

        # Ensure stored_agent directory exists with weights
        self.ensure_stored_agent_directory(model_dir)

        # Get trial parameters to extract risk management settings
        trial_params = self._get_trial_params(trial_id) if trial_id else {}
        trailing_stop_pct = trial_params.get('trailing_stop_pct', 0.08) if trial_params else 0.08

        # Stop current paper trader
        self.stop_current_paper_trader()
        time.sleep(2)

        # Start new paper trader with SINGLE model (not ensemble)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/paper_trading_arena_{timestamp}.log"
        csv_file = f"paper_trades/arena_session_{timestamp}.csv"

        cmd = [
            "nohup", "python", "-u", "paper_trader_alpaca_polling.py",
            "--model-dir", str(model_dir),  # Single model directory
            "--tickers", "BTC/USD", "ETH/USD", "LTC/USD", "BCH/USD", "LINK/USD", "UNI/USD", "AAVE/USD",
            "--timeframe", "1h",
            "--history-hours", "120",
            "--poll-interval", "60",
            "--gpu", "-1",
            "--log-file", csv_file,
            # Risk management
            "--max-position-pct", "0.30",
            "--stop-loss-pct", "0.10",
            "--trailing-stop-pct", str(trailing_stop_pct),
        ]

        self.logger.info(f"Starting single model paper trader: trial_{trial_num}")

        # Start process
        with open(log_file, 'w') as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

        self.logger.info(f"Started paper trader PID {proc.pid}")

        # Update state
        self.state["last_deployed_trial"] = trial_num
        self.state["last_deployed_value"] = value
        self.state["last_deployment_time"] = datetime.now().isoformat()
        self.state["current_paper_trader_pid"] = proc.pid
        self.state["deployment_source"] = "arena"  # Mark as arena-promoted
        self.state["deployment_history"].append({
            "trial_id": trial_num,
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "pid": proc.pid,
            "model_dir": str(model_dir),
            "source": "arena_promotion",
            "log_file": log_file,
            "csv_file": csv_file,
        })

        if len(self.state["deployment_history"]) > 20:
            self.state["deployment_history"] = self.state["deployment_history"][-20:]

        self._save_state()
        self._log_deployment(trial_num, value, "ARENA_DEPLOYED", f"PID: {proc.pid}")

    def _get_trial_id_from_number(self, trial_num: int) -> Optional[int]:
        """Get trial_id from trial number."""
        try:
            study = optuna.load_study(
                study_name=self.study_name,
                storage=f"sqlite:///{self.db_path}"
            )
            for trial in study.trials:
                if trial.number == trial_num:
                    return trial._trial_id
        except:
            pass
        return None

    def add_to_arena(self, trial_num: int, model_dir: Path, value: float):
        """Add a model to the arena for evaluation."""
        try:
            from model_arena import ModelArena
            arena = ModelArena()
            if arena.add_model(model_dir, trial_num, value):
                self.logger.info(f"Added trial_{trial_num} to arena")
                return True
            else:
                self.logger.info(f"trial_{trial_num} already in arena or arena full")
                return False
        except Exception as e:
            self.logger.error(f"Failed to add to arena: {e}")
            return False

    def check_arena_promotion(self) -> Optional[dict]:
        """Check if any model in arena is ready for promotion."""
        promotion_file = Path("arena_state/promotion_candidate.json")
        if not promotion_file.exists():
            return None

        try:
            with promotion_file.open() as f:
                candidate = json.load(f)

            # Check if we already deployed this candidate
            last_deployed = self.state.get("last_deployed_trial")
            if last_deployed == candidate.get("trial_number"):
                return None

            # Verify candidate is still valid (file not stale)
            candidate_time = datetime.fromisoformat(candidate["timestamp"].replace('Z', '+00:00'))
            age_hours = (datetime.now(candidate_time.tzinfo) - candidate_time).total_seconds() / 3600

            if age_hours > 24:  # Candidate file is stale
                return None

            return candidate
        except Exception as e:
            self.logger.error(f"Error checking arena promotion: {e}")
            return None

    def check_and_deploy(self):
        """Check for new best models and deploy if needed."""
        # ARENA MODE: Check for promotion candidates first
        if self.arena_mode:
            self.logger.info("Checking arena for promotion candidates...")

            # First, add top models to arena
            best_trials = self.get_best_trials(top_n=10)
            for trial_id, value, _, trial_num in best_trials[:10]:
                model_exists, model_dir = self.check_model_exists(trial_num)
                if model_exists:
                    self.add_to_arena(trial_num, model_dir, value)

            # Check for promotion candidate
            candidate = self.check_arena_promotion()
            if candidate:
                trial_num = candidate["trial_number"]
                metrics = candidate.get("metrics", {})
                self.logger.info(
                    f"Arena promotion candidate: trial_{trial_num} "
                    f"(return={metrics.get('return_pct', 0):.2f}%, "
                    f"sharpe={metrics.get('sharpe_ratio', 0):.2f})"
                )

                model_exists, model_dir = self.check_model_exists(trial_num)
                if model_exists and self.auto_deploy:
                    self.deploy_single_model(trial_num, model_dir, metrics.get("return_pct", 0))
            else:
                self.logger.info("No promotion candidates ready (need 7+ days evaluation)")

            return

        # LEGACY MODE: Direct deployment (not arena)
        self.logger.info("Checking for new best models (legacy mode)...")

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
            improvement = best_value - last_value  # Higher is better (positive values = profit)
            improvement_pct = abs(improvement / (abs(last_value) + 1e-8)) * 100

            self.logger.info(f"Improvement: {improvement:.6f} ({improvement_pct:.2f}%)")

            if improvement_pct < self.min_improvement:
                self.logger.info(f"Improvement {improvement_pct:.2f}% < threshold {self.min_improvement}%")
                return

        # Check if model exists (use trial_num for path lookup)
        model_exists, model_dir = self.check_model_exists(trial_num)

        if not model_exists:
            self.logger.warning(f"Model files not found for trial {trial_num} (ID: {best_trial_id})")
            self._log_deployment(best_trial_id, best_value, "SKIPPED", f"Model files not found for trial_{trial_num}_1h")
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
        self.logger.info(f"Arena mode: {self.arena_mode}")
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
    # Auto-detect database path
    detector = PathDetector()
    default_db = detector.find_optuna_db()

    parser = argparse.ArgumentParser(description="Auto-model deployment pipeline")
    parser.add_argument("--study", default=_DEFAULT_STUDY, help="Study name (default from .env.training)")
    parser.add_argument("--db", default=default_db, help="Database path (auto-detected)")
    parser.add_argument("--check-interval", type=int, default=3600, help="Check interval in seconds")
    parser.add_argument("--min-improvement", type=float, default=1.0, help="Min improvement % to deploy")
    parser.add_argument("--no-validation", action="store_true", help="Disable validation")
    parser.add_argument("--no-auto-deploy", action="store_true", help="Disable auto-deployment")
    parser.add_argument("--daemon", action="store_true", help="Run in daemon mode")
    parser.add_argument("--arena-mode", action="store_true",
                       help="Use Model Arena for evaluation (deploy single models, not ensemble)")

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
        arena_mode=args.arena_mode,
    )

    deployer.run()


if __name__ == "__main__":
    main()
