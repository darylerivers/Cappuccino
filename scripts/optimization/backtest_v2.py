#!/usr/bin/env python3
"""
Backtest Validator V2 - Simple, Robust Model Validation

Validates trained models without complex dependencies.
"""

import logging
import sqlite3
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class BacktestValidator:
    """Validates models with backtesting."""

    def __init__(self, optuna_db: str = "/tmp/optuna_working.db",
                 data_dir: str = "data/1h_1680"):
        """
        Initialize validator.

        Args:
            optuna_db: Path to Optuna database
            data_dir: Path to validation data
        """
        self.optuna_db = optuna_db
        self.data_dir = Path(data_dir)

    def validate(self, trial_num: int) -> Dict[str, Any]:
        """
        Validate a trial's model.

        Args:
            trial_num: Trial number to validate

        Returns:
            Dict with success status and metrics
        """
        try:
            logger.info(f"Validating trial {trial_num}")

            # Load trial parameters
            params = self._load_trial_params(trial_num)
            if not params:
                return {'success': False, 'error': 'Failed to load trial params'}

            # Find model directory
            model_dir = Path(f"train_results/cwd_tests/trial_{trial_num}_1h")
            if not model_dir.exists():
                return {'success': False, 'error': f'Model directory not found: {model_dir}'}

            # Check for actor model file
            actor_path = model_dir / "actor.pth"
            if not actor_path.exists():
                return {'success': False, 'error': f'Actor model not found: {actor_path}'}

            # Load validation data
            val_data = self._load_validation_data()
            if val_data is None:
                return {'success': False, 'error': 'Failed to load validation data'}

            # Create environment with trial params
            env = self._create_environment(val_data, params)
            if env is None:
                return {'success': False, 'error': 'Failed to create environment'}

            # Load model
            agent = self._load_model(model_dir, params, env)
            if agent is None:
                return {'success': False, 'error': 'Failed to load model'}

            # Run backtest
            metrics = self._run_backtest(agent, env)

            # Check if passed basic gates
            passed = self._check_gates(metrics)

            logger.info(f"Trial {trial_num} backtest: {'PASSED' if passed else 'FAILED'}")
            logger.info(f"  Metrics: {metrics}")

            return {
                'success': passed,
                'metrics': metrics,
                'trial_number': trial_num
            }

        except Exception as e:
            logger.error(f"Backtest validation error: {str(e)}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def _load_trial_params(self, trial_num: int) -> Optional[Dict]:
        """Load trial parameters from Optuna database."""
        try:
            conn = sqlite3.connect(self.optuna_db)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT param_name, param_value
                FROM trial_params tp
                JOIN trials t ON tp.trial_id = t.trial_id
                WHERE t.number = ?
            ''', (trial_num,))

            params = {row[0]: row[1] for row in cursor.fetchall()}
            conn.close()

            logger.info(f"Loaded {len(params)} parameters for trial {trial_num}")
            return params

        except Exception as e:
            logger.error(f"Failed to load trial params: {e}")
            return None

    def _load_validation_data(self) -> Optional[Dict]:
        """Load validation data."""
        try:
            # Try validation directory first
            val_dir = self.data_dir / "val"
            if not val_dir.exists():
                val_dir = self.data_dir  # Fallback to main data dir

            price_path = val_dir / "price_array.npy"
            tech_path = val_dir / "tech_array.npy"

            if not price_path.exists() or not tech_path.exists():
                logger.error(f"Validation data not found in {val_dir}")
                return None

            price_array = np.load(price_path)
            tech_array = np.load(tech_path)

            logger.info(f"Loaded validation data: price {price_array.shape}, tech {tech_array.shape}")

            return {
                'price_array': price_array,
                'tech_array': tech_array
            }

        except Exception as e:
            logger.error(f"Failed to load validation data: {e}")
            return None

    def _create_environment(self, val_data: Dict, params: Dict):
        """Create trading environment with trial parameters."""
        try:
            from environment_Alpaca import CryptoEnvAlpaca

            env_config = {
                'price_array': val_data['price_array'],
                'tech_array': val_data['tech_array'],
                'if_train': False
            }

            env_params = {
                'lookback': int(params.get('lookback', 4)),
                'norm_cash': 2 ** float(params.get('norm_cash_exp', -11)),
                'norm_stocks': 2 ** float(params.get('norm_stocks_exp', -8)),
                'norm_tech': 2 ** float(params.get('norm_tech_exp', -14)),
                'norm_reward': 2 ** float(params.get('norm_reward_exp', -9)),
                'norm_action': float(params.get('norm_action', 100)),
                'time_decay_floor': float(params.get('time_decay_floor', 0.0)),
                'min_cash_reserve': float(params.get('min_cash_reserve', 0.1)),
                'concentration_penalty': float(params.get('concentration_penalty', 0.05))
            }

            env = CryptoEnvAlpaca(env_config, env_params, if_log=False)
            logger.info(f"Created environment with lookback={env_params['lookback']}")

            return env

        except Exception as e:
            logger.error(f"Failed to create environment: {e}")
            return None

    def _load_model(self, model_dir: Path, params: Dict, env):
        """Load trained model."""
        try:
            from train.run import init_agent
            from train.config import Arguments
            from drl_agents.elegantrl_models import MODELS

            net_dim = int(params.get('net_dimension', 256))

            args = Arguments(agent=MODELS['ppo'], env=env)
            args.cwd = str(model_dir)
            args.if_remove = False
            args.net_dim = net_dim

            agent = init_agent(args, gpu_id=-1, env=env)  # CPU
            agent.act.eval()

            logger.info(f"Loaded model from {model_dir}")
            return agent

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

    def _run_backtest(self, agent, env) -> Dict[str, float]:
        """Run backtest and calculate metrics."""
        try:
            state = env.reset()
            done = False
            portfolio_values = [env.initial_total_asset]

            steps = 0
            while not done and steps < 10000:  # Safety limit
                state_tensor = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    action = agent.act(state_tensor).cpu().numpy()[0]

                state, reward, done, _ = env.step(action)
                portfolio_values.append(env.total_asset)
                steps += 1

            # Calculate metrics
            portfolio_values = np.array(portfolio_values)
            returns = np.diff(portfolio_values) / portfolio_values[:-1]

            total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252 * 24)  # Annualized for hourly data

            # Calculate max drawdown
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (portfolio_values - peak) / peak
            max_drawdown = np.min(drawdown)

            metrics = {
                'total_return': float(total_return),
                'sharpe': float(sharpe),
                'max_drawdown': float(max_drawdown),
                'final_value': float(portfolio_values[-1]),
                'steps': steps
            }

            return metrics

        except Exception as e:
            logger.error(f"Backtest execution error: {e}")
            return {
                'total_return': -1.0,
                'sharpe': -999.0,
                'max_drawdown': -1.0,
                'final_value': 0.0,
                'error': str(e)
            }

    def _check_gates(self, metrics: Dict) -> bool:
        """
        Check if metrics pass basic quality gates.

        Simple gates:
        - Max loss not worse than -50%
        - Max drawdown not worse than 80%
        """
        if 'error' in metrics:
            return False

        if metrics['total_return'] < -0.5:
            logger.info(f"Failed gate: total_return {metrics['total_return']} < -0.5")
            return False

        if metrics['max_drawdown'] < -0.8:
            logger.info(f"Failed gate: max_drawdown {metrics['max_drawdown']} < -0.8")
            return False

        return True


def main():
    """Test backtest validator."""
    import argparse

    parser = argparse.ArgumentParser(description='Backtest Validator V2')
    parser.add_argument('trial_num', type=int, help='Trial number to validate')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    validator = BacktestValidator()
    result = validator.validate(args.trial_num)

    print(f"\nResult: {'PASSED' if result['success'] else 'FAILED'}")
    if 'metrics' in result:
        print("\nMetrics:")
        for key, value in result['metrics'].items():
            print(f"  {key}: {value}")
    if 'error' in result:
        print(f"\nError: {result['error']}")


if __name__ == '__main__':
    main()
