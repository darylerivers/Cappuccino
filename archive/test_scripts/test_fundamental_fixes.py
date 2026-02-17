#!/usr/bin/env python3
"""
Test Fundamental Fixes - 50 Trial Validation

This script tests the three fundamental fixes applied to environment_Alpaca.py:
1. Enhanced concentration penalty (require 3+ positions, cap at 40%)
2. Revised reward function (hybrid: alpha + absolute + cash management)
3. Cash reserve enforcement

Runs 50 trials for comparison against baseline performance.

Usage:
    python test_fundamental_fixes.py --gpu 0

    # With custom study name
    python test_fundamental_fixes.py --study-name cappuccino_fixes_custom

    # With more trials
    python test_fundamental_fixes.py --n-trials 100
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Fix stdout buffering
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
os.environ['PYTHONUNBUFFERED'] = '1'

import numpy as np
import pandas as pd
import optuna
import sqlite3

# Add parent directory to path
PARENT_DIR = Path(__file__).parent.parent / "ghost/FinRL_Crypto"
sys.path.insert(0, str(PARENT_DIR))

from environment_Alpaca import CryptoEnvAlpaca
from function_CPCV import CombPurgedKFoldCV
from function_train_test import train_and_test

from optuna.storages import RDBStorage


class Colors:
    """ANSI color codes."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_TICKERS = ['BTC', 'ETH', 'LTC', 'DOGE', 'ADA', 'SOL', 'MATIC']
DEFAULT_INDICATORS = ['high', 'low', 'volume', 'macd', 'macd_signal', 'macd_hist', 'rsi', 'cci', 'dx']

SEED = 312


# =============================================================================
# Database Setup
# =============================================================================

def create_optimized_storage(storage_url: str) -> RDBStorage:
    """Create Optuna storage with SQLite optimizations."""
    if storage_url.startswith('sqlite:///'):
        db_path = storage_url.replace('sqlite:///', '')
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            conn = sqlite3.connect(db_path)
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA synchronous=NORMAL')
            conn.execute('PRAGMA cache_size=10000')
            conn.execute('PRAGMA temp_store=MEMORY')
            conn.close()
            print(f"  SQLite WAL mode enabled for: {db_path}")
        except Exception as e:
            print(f"  Warning: Could not enable WAL mode: {e}")

    storage = RDBStorage(
        url=storage_url,
        engine_kwargs={
            "connect_args": {
                "timeout": 60,
                "check_same_thread": False,
            },
            "pool_size": 5,
            "max_overflow": 10,
            "pool_pre_ping": True,
        },
        heartbeat_interval=60,
        grace_period=120,
    )

    return storage


# =============================================================================
# Data Loading
# =============================================================================

def load_data(data_dir: str):
    """Load price, tech, and time arrays."""
    print(f"{Colors.BLUE}Loading data from {data_dir}...{Colors.END}")

    price_array = np.load(Path(data_dir) / 'price_array.npy')
    tech_array = np.load(Path(data_dir) / 'tech_array.npy')
    time_array = np.load(Path(data_dir) / 'time_array.npy')

    # Sentiment array (optional, may not exist)
    sentiment_path = Path(data_dir) / 'sentiment_array.npy'
    if sentiment_path.exists():
        sentiment_array = np.load(sentiment_path)
    else:
        sentiment_array = None

    print(f"  Price array shape: {price_array.shape}")
    print(f"  Tech array shape: {tech_array.shape}")
    print(f"  Time range: {time_array[0]} to {time_array[-1]}")

    return price_array, tech_array, time_array, sentiment_array


# =============================================================================
# Hyperparameter Sampling
# =============================================================================

def sample_hyperparameters(trial: optuna.Trial) -> dict:
    """Sample hyperparameters for PPO agent."""

    # Network architecture
    net_dim = trial.suggest_categorical('net_dim', [256, 512, 1024])
    layer_num = trial.suggest_int('layer_num', 3, 6)

    # PPO hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.90, 0.999)
    batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024, 2048])
    repeat_times = trial.suggest_int('repeat_times', 4, 16)

    # Entropy and ratio clip
    lambda_entropy = trial.suggest_float('lambda_entropy', 0.001, 0.05)
    ratio_clip = trial.suggest_float('ratio_clip', 0.15, 0.35)

    # Advantage normalization
    if_use_adv_norm = trial.suggest_categorical('if_use_adv_norm', [True, False])

    # Environment parameters - CRITICAL FOR TESTING FIXES
    min_cash_reserve = trial.suggest_float('min_cash_reserve', 0.0, 0.3)
    concentration_penalty = trial.suggest_float('concentration_penalty', 0.0, 2.0)
    max_stock_penalty = trial.suggest_float('max_stock_penalty', 0.0, 2.0)
    norm_reward = trial.suggest_float('norm_reward', 100, 10000, log=True)

    # Reward decay for long-term planning
    reward_decay = trial.suggest_float('reward_decay', 0.995, 1.0)

    return {
        'net_dim': net_dim,
        'layer_num': layer_num,
        'learning_rate': learning_rate,
        'gamma': gamma,
        'batch_size': batch_size,
        'repeat_times': repeat_times,
        'lambda_entropy': lambda_entropy,
        'ratio_clip': ratio_clip,
        'if_use_adv_norm': if_use_adv_norm,
        'min_cash_reserve': min_cash_reserve,
        'concentration_penalty': concentration_penalty,
        'max_stock_penalty': max_stock_penalty,
        'norm_reward': norm_reward,
        'reward_decay': reward_decay,
    }


# =============================================================================
# Objective Function
# =============================================================================

def objective(trial: optuna.Trial, price_array, tech_array, time_array, sentiment_array, args) -> float:
    """Objective function for Optuna optimization."""

    print(f"\n{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.HEADER}Trial #{trial.number} - Testing Fundamental Fixes{Colors.END}")
    print(f"{Colors.HEADER}{'='*80}{Colors.END}\n")

    # Sample hyperparameters
    params = sample_hyperparameters(trial)

    print(f"{Colors.CYAN}Hyperparameters:{Colors.END}")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print()

    # Environment configuration
    env_params = {
        'price_array': price_array,
        'tech_array': tech_array,
        'if_train': True,
        'tickers': args.tickers,
        'technical_indicators': DEFAULT_INDICATORS,
        'min_cash_reserve': params['min_cash_reserve'],
        'concentration_penalty': params['concentration_penalty'],
        'max_stock_penalty': params['max_stock_penalty'],
        'norm_reward': params['norm_reward'],
        'reward_decay': params['reward_decay'],
    }

    # Agent configuration
    agent_config = {
        'net_dim': params['net_dim'],
        'layer_num': params['layer_num'],
        'learning_rate': params['learning_rate'],
        'gamma': params['gamma'],
        'batch_size': params['batch_size'],
        'repeat_times': params['repeat_times'],
        'lambda_entropy': params['lambda_entropy'],
        'ratio_clip': params['ratio_clip'],
        'if_use_adv_norm': params['if_use_adv_norm'],
    }

    # CPCV setup
    cpcv = CombPurgedKFoldCV(
        n_splits=args.num_paths,
        n_test_splits=args.k_test_groups,
        embargo_td=pd.Timedelta(hours=24),  # 24 hours embargo for 1h data
    )

    # Create DataFrame for CPCV (requires pandas DataFrame with DatetimeIndex)
    data = pd.DataFrame(tech_array)
    data = data.set_index(pd.DatetimeIndex(pd.to_datetime(time_array, unit='s')))

    # Create pred_times and eval_times (required by CPCV)
    y = pd.Series([0] * len(data), index=data.index)
    pred_times = pd.Series(data.index, index=data.index)
    eval_times = pd.Series(data.index, index=data.index)

    # Run cross-validation
    sharpe_list_bot = []
    sharpe_list_hodl = []

    print(f"{Colors.BLUE}Running {args.num_paths} CPCV splits...{Colors.END}\n")

    for split_idx, (train_idx, test_idx) in enumerate(cpcv.split(data, y, pred_times=pred_times, eval_times=eval_times)):
        print(f"{Colors.YELLOW}Split {split_idx + 1}/{args.num_paths}{Colors.END}")

        try:
            # Train and test
            result = train_and_test(
                config=agent_config,
                env_params=env_params,
                train_idx=train_idx,
                test_idx=test_idx,
                trial_name=f"trial_{trial.number}_split{split_idx}",
                trial_number=trial.number,
                save_best=True,
            )

            sharpe_bot = result.get('sharpe_bot', -999)
            sharpe_hodl = result.get('sharpe_hodl', 0)

            sharpe_list_bot.append(sharpe_bot)
            sharpe_list_hodl.append(sharpe_hodl)

            print(f"    Bot Sharpe: {sharpe_bot:.4f}, HODL Sharpe: {sharpe_hodl:.4f}")

        except Exception as e:
            print(f"    {Colors.RED}Split {split_idx} failed: {e}{Colors.END}")
            continue

    if not sharpe_list_bot:
        raise optuna.exceptions.TrialPruned("All splits failed")

    # Calculate objective
    mean_sharpe_bot = np.mean(sharpe_list_bot)
    mean_sharpe_hodl = np.mean(sharpe_list_hodl)
    std_sharpe_bot = np.std(sharpe_list_bot)

    # Risk-adjusted objective: penalize variance
    objective_value = mean_sharpe_bot - mean_sharpe_hodl - 0.1 * std_sharpe_bot

    # Store metrics for analysis
    trial.set_user_attr('sharpe_list_bot', sharpe_list_bot)
    trial.set_user_attr('sharpe_list_hodl', sharpe_list_hodl)
    trial.set_user_attr('mean_sharpe_bot', mean_sharpe_bot)
    trial.set_user_attr('mean_sharpe_hodl', mean_sharpe_hodl)
    trial.set_user_attr('std_sharpe_bot', std_sharpe_bot)
    trial.set_user_attr('test_type', 'fundamental_fixes')
    trial.set_user_attr('fix_version', '2025-12-15')

    print(f"\n{Colors.GREEN}Trial #{trial.number} Results:{Colors.END}")
    print(f"  Bot Sharpe: {mean_sharpe_bot:.6f} ± {std_sharpe_bot:.6f}")
    print(f"  HODL Sharpe: {mean_sharpe_hodl:.6f}")
    print(f"  Objective: {objective_value:.6f}")
    print(f"{Colors.GREEN}{'='*80}{Colors.END}\n")

    return objective_value


# =============================================================================
# Main Function
# =============================================================================

def main():
    """Run 50-trial validation test."""
    parser = argparse.ArgumentParser(description="Test Fundamental Fixes - 50 Trial Validation")
    parser.add_argument('--n-trials', type=int, default=50, help='Number of trials')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--study-name', type=str, default='cappuccino_fundamentals_test_20251215', help='Study name')
    parser.add_argument('--storage', type=str, default='sqlite:///databases/optuna_cappuccino.db', help='Storage URL')
    parser.add_argument('--data-dir', type=str, default='data/1h_1680', help='Data directory')
    parser.add_argument('--num-paths', type=int, default=3, help='CPCV paths')
    parser.add_argument('--k-test-groups', type=int, default=2, help='CPCV test groups')
    parser.add_argument('--tickers', nargs='+', default=DEFAULT_TICKERS, help='Tickers')
    args = parser.parse_args()

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Banner
    print(f"\n{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.HEADER}FUNDAMENTAL FIXES VALIDATION TEST{Colors.END}")
    print(f"{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.YELLOW}Testing 3 fundamental fixes:{Colors.END}")
    print(f"  1. Enhanced concentration penalty (require 3+ positions, cap at 40%)")
    print(f"  2. Revised reward function (50% alpha + 30% absolute + 20% cash)")
    print(f"  3. Cash reserve enforcement (penalty for violations)")
    print(f"\n{Colors.CYAN}Configuration:{Colors.END}")
    print(f"  Study: {args.study_name}")
    print(f"  Storage: {args.storage}")
    print(f"  Trials: {args.n_trials}")
    print(f"  GPU: {args.gpu}")
    print(f"  Data: {args.data_dir}")
    print(f"  Tickers: {', '.join(args.tickers)}")
    print(f"{Colors.HEADER}{'='*80}{Colors.END}\n")
    sys.stdout.flush()

    # Create directories
    Path('databases').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    Path('train_results').mkdir(exist_ok=True)

    # Load data
    price_array, tech_array, time_array, sentiment_array = load_data(args.data_dir)

    # Create Optuna study
    print(f"\n{Colors.BLUE}Creating Optuna study...{Colors.END}")
    storage = create_optimized_storage(args.storage)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction='maximize',
        load_if_exists=True
    )

    # Objective wrapper
    def objective_wrapper(trial):
        return objective(trial, price_array, tech_array, time_array, sentiment_array, args)

    # Run optimization
    print(f"{Colors.GREEN}Starting {args.n_trials}-trial validation test...{Colors.END}")
    print(f"{Colors.YELLOW}Expected duration: ~8-10 hours{Colors.END}\n")
    sys.stdout.flush()

    study.optimize(objective_wrapper, n_trials=args.n_trials, show_progress_bar=False)

    # Results
    print(f"\n{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.HEADER}VALIDATION TEST COMPLETE{Colors.END}")
    print(f"{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.6f}")
    print(f"\n{Colors.CYAN}Best hyperparameters:{Colors.END}")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"\n{Colors.GREEN}Next step: Analyze results and compare to baseline{Colors.END}")
    print(f"{Colors.YELLOW}Use: python analyze_test_results.py --study {args.study_name}{Colors.END}")
    print(f"{Colors.HEADER}{'='*80}{Colors.END}\n")


if __name__ == '__main__':
    main()
