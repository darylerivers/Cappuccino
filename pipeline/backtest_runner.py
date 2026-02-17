"""
Backtest Runner
Automates backtesting for pipeline validation.
"""

import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional
import time


class BacktestRunner:
    """Runs backtests programmatically for pipeline validation."""

    def __init__(self, config=None):
        self.logger = logging.getLogger(__name__)
        self.metrics_file = Path("plots_and_metrics/test_metrics.txt")
        self.config = config or {}

    def run(self, trial_number: int, retries: int = 3) -> Optional[Dict]:
        """
        Run backtest for a trial.

        Args:
            trial_number: Trial number to backtest
            retries: Number of retry attempts

        Returns:
            Dict with metrics or None if failed
        """
        self.logger.info(f"Running backtest for trial {trial_number}")

        # Check if model exists
        model_dir = self._find_model_dir(trial_number)
        if not model_dir:
            self.logger.error(f"Model directory not found for trial {trial_number}")
            return None

        # Run backtest with retries
        for attempt in range(retries):
            try:
                metrics = self._run_backtest(trial_number, model_dir)
                if metrics:
                    self.logger.info(f"Backtest successful for trial {trial_number}")
                    return metrics

                self.logger.warning(f"Backtest attempt {attempt + 1}/{retries} failed")

                if attempt < retries - 1:
                    time.sleep(300)  # Wait 5 minutes before retry

            except Exception as e:
                self.logger.error(f"Backtest error (attempt {attempt + 1}): {e}", exc_info=True)

                if attempt < retries - 1:
                    time.sleep(300)

        return None

    def _find_model_dir(self, trial_number: int) -> Optional[Path]:
        """Find model directory for a trial."""
        possible_paths = [
            Path(f"train_results/cwd_tests/trial_{trial_number}_rerun_1h"),  # Rerun with weights
            Path(f"train_results/cwd_tests/trial_{trial_number}_1h"),
            Path(f"train_results/cwd_tests/trial_{trial_number}"),
        ]

        for path in possible_paths:
            if not path.exists():
                continue

            # Check if cleaned up (weights removed)
            if (path / ".cleanup_done").exists() and not (path / "actor.pth").exists():
                continue

            # Check for required weight files
            if (path / "actor.pth").exists() and (path / "critic.pth").exists():
                return path

            # Also check stored_agent subdirectory
            stored_agent = path / "stored_agent"
            if stored_agent.exists():
                if (stored_agent / "actor.pth").exists() and (stored_agent / "critic.pth").exists():
                    return path

        return None

    def _run_backtest(self, trial_number: int, model_dir: Path) -> Optional[Dict]:
        """
        Run backtest directly using the same method as auto_model_deployer.py.

        This loads the model and runs it through validation data to calculate metrics.
        """
        self.logger.info(f"Running in-process backtest for trial {trial_number}")

        try:
            import numpy as np
            import torch
            from environment_Alpaca import CryptoEnvAlpaca
            import sqlite3

            # Load validation data
            val_dir = Path("data/1h_1680/val")
            if not val_dir.exists():
                val_dir = Path("data/1h_1680")  # Fallback to full data

            if not val_dir.exists():
                self.logger.error(f"Data directory not found: {val_dir}")
                return None

            price_array = np.load(val_dir / "price_array.npy")
            tech_array = np.load(val_dir / "tech_array.npy")

            self.logger.info(f"Loaded data: price_array {price_array.shape}, tech_array {tech_array.shape}")

            # Get trial parameters from database
            params = self._get_trial_params_from_db(trial_number)
            if not params:
                self.logger.warning(f"Could not load params for trial {trial_number}, using defaults")
                params = self._get_default_params()

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

            # Log environment state space
            state = env.reset()
            self.logger.info(f"Environment created with state space size: {len(state)}")
            self.logger.info(f"Environment params: lookback={env_params['lookback']}, n_assets={price_array.shape[1]}")

            # Load model
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
            actions_taken = []
            step_count = 0

            while not done:
                state_tensor = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    action = agent.act(state_tensor).cpu().numpy()[0]
                actions_taken.append(action)
                state, reward, done, _ = env.step(action)
                portfolio_values.append(env.total_asset)
                step_count += 1

            # Log action statistics
            actions_array = np.array(actions_taken)
            self.logger.info(f"Completed {step_count} steps")
            self.logger.info(f"Action stats: mean={np.mean(np.abs(actions_array)):.4f}, max={np.max(np.abs(actions_array)):.4f}, non-zero={np.count_nonzero(actions_array)}/{len(actions_array)}")

            # Calculate metrics
            portfolio_values = np.array(portfolio_values)
            returns = np.diff(portfolio_values) / portfolio_values[:-1]

            total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(24 * 365)  # Annualized

            # Max drawdown
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (peak - portfolio_values) / peak
            max_drawdown = np.max(drawdown)

            # Volatility
            volatility = np.std(returns) * np.sqrt(24 * 365)  # Annualized

            return {
                'total_return': total_return,
                'sharpe': sharpe,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'final_value': portfolio_values[-1],
            }

        except Exception as e:
            self.logger.error(f"Backtest error: {e}", exc_info=True)
            return None

    def _parse_metrics_file(self) -> Optional[Dict]:
        """Parse metrics from test_metrics.txt file."""
        if not self.metrics_file.exists():
            self.logger.warning(f"Metrics file not found: {self.metrics_file}")
            return None

        try:
            with open(self.metrics_file, 'r') as f:
                content = f.read()

            metrics = {}

            # Parse metrics using regex
            patterns = {
                "total_return": r"Total Return:\s*([-+]?\d*\.?\d+)%",
                "annual_return": r"Annual(?:ized)? Return:\s*([-+]?\d*\.?\d+)%",
                "sharpe": r"Sharpe Ratio:\s*([-+]?\d*\.?\d+)",
                "max_drawdown": r"Max Drawdown:\s*([-+]?\d*\.?\d+)%",
                "volatility": r"Volatility:\s*([-+]?\d*\.?\d+)%",
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    value = float(match.group(1))
                    # Convert percentages to decimals
                    if key in ["total_return", "annual_return", "max_drawdown", "volatility"]:
                        value = value / 100.0
                    metrics[key] = value

            if not metrics:
                self.logger.warning("No metrics parsed from file")
                return None

            self.logger.info(f"Parsed metrics: {metrics}")
            return metrics

        except Exception as e:
            self.logger.error(f"Error parsing metrics file: {e}")
            return None

    def _parse_stdout(self, stdout: str) -> Optional[Dict]:
        """Parse metrics from stdout as fallback."""
        try:
            metrics = {}

            patterns = {
                "total_return": r"Total Return:\s*([-+]?\d*\.?\d+)%",
                "sharpe": r"Sharpe:\s*([-+]?\d*\.?\d+)",
                "max_drawdown": r"Max Drawdown:\s*([-+]?\d*\.?\d+)%",
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, stdout, re.IGNORECASE)
                if match:
                    value = float(match.group(1))
                    if "%" in pattern:
                        value = value / 100.0
                    metrics[key] = value

            return metrics if metrics else None

        except Exception as e:
            self.logger.error(f"Error parsing stdout: {e}")
            return None

    def _get_trial_params_from_db(self, trial_number: int) -> Optional[Dict]:
        """Get trial parameters from Optuna database."""
        try:
            import sqlite3
            db_path = self.config.get('database', {}).get('optuna_db_path', 'databases/optuna_cappuccino.db')
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT param_name, param_value
                FROM trial_params tp
                JOIN trials t ON tp.trial_id = t.trial_id
                WHERE t.number = ?
            """, (trial_number,))

            params = {}
            for name, value in cursor.fetchall():
                params[name] = value
            conn.close()

            return params if params else None
        except Exception as e:
            self.logger.error(f"Error loading params from DB: {e}")
            return None

    def _get_default_params(self) -> Dict:
        """Get default parameters if trial params can't be loaded."""
        return {
            'lookback': 60,
            'norm_cash_exp': -11,
            'norm_stocks_exp': -8,
            'norm_tech_exp': -14,
            'norm_reward_exp': -9,
            'norm_action': 100,
            'time_decay_floor': 0.0,
            'min_cash_reserve': 0.1,
            'concentration_penalty': 0.05,
            'net_dimension': 256,
        }

    def quick_validate(self, trial_number: int) -> bool:
        """
        Quick validation: just check if model files exist.
        Used for faster pipeline checks.
        """
        model_dir = self._find_model_dir(trial_number)
        if not model_dir:
            return False

        required_files = ["actor.pth", "critic.pth"]
        return all((model_dir / f).exists() for f in required_files)
