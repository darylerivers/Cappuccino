#!/usr/bin/env python3
"""
Deploy best trial from Optuna study to production.

Retrains the best trial with exact hyperparameters and saves checkpoint
for paper trading deployment.

Usage:
    python deploy_best_trial.py --study cappuccino_ft_transformer --trial 250
"""

import argparse
import json
import os
import pickle
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Fix stdout buffering
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
os.environ['PYTHONUNBUFFERED'] = '1'

from environment_Alpaca import CryptoEnvAlpaca
from utils.function_train_test import train_and_test


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def get_trial_from_optuna(db_path: str, study_name: str, trial_number: int):
    """Load trial parameters from Optuna database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get study_id
    cursor.execute("SELECT study_id FROM studies WHERE study_name = ?", (study_name,))
    result = cursor.fetchone()
    if not result:
        raise ValueError(f"Study '{study_name}' not found in database")
    study_id = result[0]

    # Get trial
    cursor.execute("""
        SELECT t.trial_id, tv.value
        FROM trials t
        LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE t.study_id = ? AND t.number = ?
    """, (study_id, trial_number))

    trial_data = cursor.fetchone()
    if not trial_data:
        raise ValueError(f"Trial #{trial_number} not found in study '{study_name}'")

    trial_id, sharpe = trial_data

    # Get all parameters
    cursor.execute("""
        SELECT param_name, param_value
        FROM trial_params
        WHERE trial_id = ?
    """, (trial_id,))

    params = {}
    for name, value in cursor.fetchall():
        # Convert Optuna's float storage to appropriate types
        if name in ['lookback', 'net_dimension', 'worker_num', 'thread_num', 'batch_size',
                    'base_target_step', 'base_break_step', 'ppo_epochs', 'eval_time_gap']:
            params[name] = int(value)
        elif name in ['use_ft_encoder', 'ft_use_pretrained', 'ft_freeze_encoder', 'use_lr_schedule']:
            params[name] = bool(int(value))
        else:
            params[name] = float(value)

    # Get user attributes
    cursor.execute("""
        SELECT key, value_json
        FROM trial_user_attributes
        WHERE trial_id = ?
    """, (trial_id,))

    user_attrs = {}
    for key, value_json in cursor.fetchall():
        try:
            user_attrs[key] = json.loads(value_json) if value_json else None
        except:
            user_attrs[key] = value_json

    conn.close()

    return {
        'trial_id': trial_id,
        'number': trial_number,
        'sharpe': sharpe,
        'params': params,
        'user_attrs': user_attrs
    }


# Custom unpickler for numpy compatibility
class NumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'numpy._core.numeric':
            module = 'numpy.core.numeric'
        elif module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)


def load_data(data_dir: str):
    """Load price, tech, and time arrays."""
    data_path = Path(data_dir)

    price_file = data_path / 'price_array'
    tech_file = data_path / 'tech_array'
    time_file = data_path / 'time_array'

    with open(price_file, 'rb') as f:
        price_array = NumpyUnpickler(f).load()
    with open(tech_file, 'rb') as f:
        tech_array = NumpyUnpickler(f).load()
    with open(time_file, 'rb') as f:
        time_array = NumpyUnpickler(f).load()

    return price_array, tech_array, time_array


def create_mock_trial(trial_data):
    """Create a mock Optuna trial object for compatibility."""
    class MockTrial:
        def __init__(self, data):
            self.number = data['number']
            self.params = data['params']
            self.user_attrs = data['user_attrs']
            self.value = data['sharpe']

    return MockTrial(trial_data)


def main():
    parser = argparse.ArgumentParser(description='Deploy best trial to production')
    parser.add_argument('--study', type=str, default='cappuccino_ft_transformer',
                        help='Optuna study name')
    parser.add_argument('--trial', type=int, required=True,
                        help='Trial number to deploy')
    parser.add_argument('--db', type=str, default='databases/optuna_cappuccino.db',
                        help='Optuna database path')
    parser.add_argument('--data-dir', type=str, default='data/1h_1680',
                        help='Data directory')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=312,
                        help='Random seed')

    args = parser.parse_args()

    print(f"{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}DEPLOYING TRIAL #{args.trial} TO PRODUCTION{Colors.END}")
    print(f"{Colors.HEADER}{'='*80}{Colors.END}\n")

    # Load trial from database
    print(f"Loading trial from Optuna database...")
    trial_data = get_trial_from_optuna(args.db, args.study, args.trial)

    print(f"{Colors.GREEN}✓ Trial loaded successfully{Colors.END}")
    print(f"  Trial: #{trial_data['number']}")
    print(f"  Sharpe: {Colors.GREEN}{trial_data['sharpe']:.6f}{Colors.END}")
    print(f"  Model: {trial_data['user_attrs'].get('model_name', 'ppo')}")
    print(f"  FT-Transformer: {trial_data['params'].get('use_ft_encoder', False)}")
    print()

    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f"train_results/deployment_trial{args.trial}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print()

    # Load data
    print("Loading training data...")
    price_array, tech_array, time_array = load_data(args.data_dir)

    print(f"  Price shape: {price_array.shape}")
    print(f"  Tech shape: {tech_array.shape}")
    print(f"  Time samples: {len(time_array)}")
    print(f"  Date range: {pd.Timestamp(time_array[0], unit='s')} to {pd.Timestamp(time_array[-1], unit='s')}")
    print()

    # Setup environment parameters
    params = trial_data['params']
    env_params = {
        'lookback': params['lookback'],
        'norm_cash': 2 ** params.get('norm_cash_exp', -11),
        'norm_stocks': 2 ** params.get('norm_stocks_exp', -8),
        'norm_tech': 2 ** params.get('norm_tech_exp', -14),
        'norm_reward': 2 ** params.get('norm_reward_exp', -9),
        'norm_action': params.get('norm_action', 3000),
        'time_decay_floor': params.get('time_decay_floor', 0.0),
        'min_cash_reserve': params.get('min_cash_reserve', 0.1),
        'concentration_penalty': params.get('concentration_penalty', 0.05),
    }

    # Setup simple train/test split (80/20)
    print("Setting up train/test split...")
    n_samples = len(time_array)
    split_point = int(n_samples * 0.8)

    train_idx = np.arange(0, split_point)
    test_idx = np.arange(split_point, n_samples)

    print(f"  Train samples: {len(train_idx)} ({len(train_idx)/n_samples*100:.1f}%)")
    print(f"  Test samples: {len(test_idx)} ({len(test_idx)/n_samples*100:.1f}%)")
    print()

    # Setup ERL parameters
    # Safety check for batch_size (handle Optuna storage quirks)
    batch_size = int(params.get('batch_size', 2048))
    if batch_size == 0:
        batch_size = 2048  # Use reasonable default
        print(f"{Colors.YELLOW}⚠️  Warning: batch_size was 0, using default: {batch_size}{Colors.END}")

    erl_params = {
        'learning_rate': params['learning_rate'],
        'batch_size': batch_size,
        'gamma': params['gamma'],
        'net_dimension': params['net_dimension'],
        'target_step': params['base_target_step'],
        'eval_time_gap': params.get('eval_time_gap', 60),
        'break_step': params['base_break_step'],
        'if_allow_break': True,
        'clip_range': params.get('clip_range', 0.2),
        'entropy_coef': params.get('entropy_coef', 0.01),
        'value_loss_coef': params.get('value_loss_coef', 0.5),
        'max_grad_norm': params.get('max_grad_norm', 0.5),
        'gae_lambda': params.get('gae_lambda', 0.95),
        'repeat_times': params.get('ppo_epochs', 10),
        'if_use_gae': True,
        **env_params
    }

    # Add FT-Transformer params if applicable
    if params.get('use_ft_encoder', False):
        print(f"{Colors.CYAN}{'='*80}{Colors.END}")
        print(f"{Colors.CYAN}FT-TRANSFORMER CONFIGURATION{Colors.END}")
        print(f"{Colors.CYAN}{'='*80}{Colors.END}")

        erl_params['use_ft_encoder'] = True
        erl_params['ft_config'] = {
            'd_token': int(params.get('ft_d_token', 32)),
            'n_blocks': int(params.get('ft_n_blocks', 2)),
            'n_heads': int(params.get('ft_n_heads', 4)),
            'dropout': params.get('ft_dropout', 0.1),
        }
        erl_params['pretrained_encoder_path'] = params.get('pretrained_encoder_path', None)
        erl_params['ft_freeze_encoder'] = params.get('ft_freeze_encoder', False)

        print(f"  FT Config: {erl_params['ft_config']}")
        print(f"  Pre-trained: {erl_params['pretrained_encoder_path'] is not None}")
        print(f"  Freeze encoder: {erl_params['ft_freeze_encoder']}")
        print(f"{Colors.CYAN}{'='*80}{Colors.END}\n")

    # Train model
    print(f"{Colors.BOLD}Starting model training...{Colors.END}\n")

    model_name = trial_data['user_attrs'].get('model_name', 'ppo')
    mock_trial = create_mock_trial(trial_data)

    sharpe_bot, sharpe_eqw, drl_rets, train_dur, test_dur = train_and_test(
        trial=mock_trial,
        price_array=price_array,
        tech_array=tech_array,
        train_indices=train_idx,
        test_indices=test_idx,
        env=CryptoEnvAlpaca,
        model_name=model_name,
        env_params=env_params,
        erl_params=erl_params,
        break_step=params['base_break_step'],
        cwd=str(output_dir),
        gpu_id=args.gpu,
        sentiment_service=None,
        use_sentiment=False,
        tickers=None,
        use_timeframe_constraint=False,
        timeframe=None,
        data_interval='1h'
    )

    # Save trial object for paper trader
    trial_obj = create_mock_trial(trial_data)
    with open(output_dir / 'best_trial', 'wb') as f:
        pickle.dump(trial_obj, f)

    print(f"\n{Colors.GREEN}{'='*80}{Colors.END}")
    print(f"{Colors.GREEN}{Colors.BOLD}✓ DEPLOYMENT COMPLETE{Colors.END}")
    print(f"{Colors.GREEN}{'='*80}{Colors.END}\n")

    print(f"Results:")
    print(f"  Agent Sharpe:      {Colors.GREEN}{sharpe_bot:.6f}{Colors.END}")
    print(f"  Benchmark Sharpe:  {sharpe_eqw:.6f}")
    print(f"  Train duration:    {train_dur:.1f}s")
    print(f"  Test duration:     {test_dur:.1f}s")
    print()

    print(f"Model saved to: {Colors.CYAN}{output_dir}{Colors.END}")
    print()

    print(f"{Colors.YELLOW}Next steps:{Colors.END}")
    print(f"  1. Stop current paper trader:")
    print(f"     pkill -f paper_trader")
    print()
    print(f"  2. Deploy to paper trading:")
    print(f"     python scripts/deployment/paper_trader_alpaca_polling.py \\")
    print(f"         --model-dir {output_dir} \\")
    print(f"         --tickers BTC/USD ETH/USD LTC/USD \\")
    print(f"         --timeframe 1h \\")
    print(f"         --history-hours 120 \\")
    print(f"         --poll-interval 3600 \\")
    print(f"         --gpu -1")
    print()


if __name__ == '__main__':
    main()
