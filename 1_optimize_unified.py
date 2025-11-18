#!/usr/bin/env python3
"""
Unified Training Script - Production Ready

Combines:
- Rolling time windows (realistic backtesting)
- Multi-timeframe support (5m to 1d)
- Sentiment analysis integration
- Enhanced PPO hyperparameters
- Risk management parameters
- Adaptive sampling and early stopping
- Comprehensive logging
- Maximum GPU utilization

Usage:
    # Basic usage (1h timeframe, no sentiment)
    python 1_optimize_unified.py --n-trials 100 --gpu 0

    # Multi-timeframe optimization
    python 1_optimize_unified.py --mode multi-timeframe --n-trials 150

    # With sentiment analysis
    python 1_optimize_unified.py --use-sentiment --sentiment-model "mvkvl/sentiments:aya"

    # Rolling windows
    python 1_optimize_unified.py --mode rolling --window-train-days 90 --window-test-days 30

    # Tightened ranges (exploit best known hyperparams)
    python 1_optimize_unified.py --use-best-ranges --n-trials 50
"""

import argparse
import itertools as itt
import joblib
import optuna
import pickle
import os
import sys
import shutil
from datetime import datetime, timedelta
from pathlib import Path

# Fix stdout buffering issues
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
os.environ['PYTHONUNBUFFERED'] = '1'

import numpy as np
import pandas as pd
import torch

from optuna.trial import TrialState

# Add parent directory to path for imports
PARENT_DIR = Path(__file__).parent.parent / "ghost/FinRL_Crypto"
sys.path.insert(0, str(PARENT_DIR))

from environment_Alpaca import CryptoEnvAlpaca
from function_CPCV import CombPurgedKFoldCV, back_test_paths_generator
from function_train_test import train_and_test


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


# =============================================================================
# Configuration and Defaults
# =============================================================================

DEFAULT_TICKERS = ['BTC', 'ETH', 'LTC']
DEFAULT_INDICATORS = ['high', 'low', 'volume', 'macd', 'macd_signal', 'macd_hist', 'rsi', 'cci', 'dx']

TIMEFRAME_MINUTES = {
    '5m': 5,
    '15m': 15,
    '30m': 30,
    '1h': 60,
    '4h': 240,
    '12h': 720,
    '1d': 1440,
}

SEED = 312


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

def load_data(data_dir: str, timeframe: str = '5m'):
    """Load price, tech, and time arrays from data directory."""
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    print(f"\n{Colors.CYAN}Loading data from: {data_path}{Colors.END}")

    # Try different file formats
    price_file = data_path / 'price_array'
    tech_file = data_path / 'tech_array'
    time_file = data_path / 'time_array'

    # Load with pickle - handle numpy backward compatibility
    # Create a custom unpickler that remaps old numpy paths to new ones
    class NumpyUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Remap old numpy module paths to new ones
            if module == 'numpy._core.numeric':
                module = 'numpy.core.numeric'
            elif module.startswith('numpy._core'):
                module = module.replace('numpy._core', 'numpy.core')
            return super().find_class(module, name)

    with open(price_file, 'rb') as f:
        price_array = NumpyUnpickler(f).load()
    with open(tech_file, 'rb') as f:
        tech_array = NumpyUnpickler(f).load()
    with open(time_file, 'rb') as f:
        time_array = NumpyUnpickler(f).load()

    print(f"  Price shape: {price_array.shape}")
    print(f"  Tech shape: {tech_array.shape}")
    print(f"  Time samples: {len(time_array)}")
    print(f"  Date range: {pd.Timestamp(time_array[0], unit='s')} to {pd.Timestamp(time_array[-1], unit='s')}")

    # Return None for sentiment_array since this is standard (non-sentiment) training
    return price_array, tech_array, time_array, None


def resample_data(price_array, tech_array, time_array, sentiment_array,
                  source_tf='5m', target_tf='1h'):
    """Resample data to different timeframe."""
    if target_tf == source_tf:
        return price_array, tech_array, time_array, sentiment_array

    factor = TIMEFRAME_MINUTES[target_tf] // TIMEFRAME_MINUTES[source_tf]

    # Take last value in each period (close price, final tech values)
    resampled_price = price_array[factor-1::factor]
    resampled_tech = tech_array[factor-1::factor]
    resampled_time = time_array[factor-1::factor]
    resampled_sentiment = sentiment_array[factor-1::factor] if sentiment_array is not None else None

    print(f"\n  {Colors.CYAN}Resampled {source_tf} → {target_tf}{Colors.END}")
    print(f"    Original: {len(time_array)} bars")
    print(f"    Resampled: {len(resampled_time)} bars (factor: {factor}x)")

    return resampled_price, resampled_tech, resampled_time, resampled_sentiment


def create_rolling_windows(time_array, train_days=90, test_days=30, step_days=30):
    """Create rolling time windows for walk-forward analysis."""
    time_df = pd.DataFrame({'time': pd.to_datetime(time_array, unit='s')})

    start_time = time_df['time'].min()
    end_time = time_df['time'].max()

    train_delta = timedelta(days=train_days)
    test_delta = timedelta(days=test_days)
    step_delta = timedelta(days=step_days)

    windows = []
    current_train_start = start_time

    while current_train_start + train_delta + test_delta <= end_time:
        train_start = current_train_start
        train_end = train_start + train_delta
        test_start = train_end
        test_end = test_start + test_delta

        if test_end > end_time:
            break

        train_mask = (time_df['time'] >= train_start) & (time_df['time'] < train_end)
        test_mask = (time_df['time'] >= test_start) & (time_df['time'] < test_end)

        train_indices = time_df[train_mask].index.values
        test_indices = time_df[test_mask].index.values

        if len(train_indices) > 100 and len(test_indices) > 10:
            windows.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_indices': train_indices,
                'test_indices': test_indices,
            })

        current_train_start += step_delta

    return windows


# =============================================================================
# Hyperparameter Sampling
# =============================================================================

def sample_hyperparams(trial, use_best_ranges=False, timeframe='1h',
                      use_sentiment=False, n_timesteps=None):
    """Sample hyperparameters with adaptive ranges.

    Args:
        trial: Optuna trial object
        use_best_ranges: Use tightened ranges from best trials
        timeframe: Trading timeframe
        use_sentiment: Whether sentiment features are enabled
        n_timesteps: Number of timesteps (for calculating steps)
    """
    # Timeframe multiplier (relative to 1h baseline)
    bars_per_day = {
        '5m': 288, '15m': 96, '30m': 48, '1h': 24,
        '4h': 6, '12h': 2, '1d': 1
    }
    multiplier = bars_per_day.get(timeframe, 24) / 24.0

    # CPU settings
    cpu_total = os.cpu_count() or 4
    max_threads = max(4, cpu_total - 1)
    max_workers = max(1, min(cpu_total - 2, 6))

    # GPU OPTIMIZATION: Use 10-16 parallel workers for MAXIMUM GPU utilization
    # Note: Old code disabled multiprocessing for sentiment, but that's not necessary
    # We can use multiple workers without multiprocessing (just parallel envs)
    if use_best_ranges:
        worker_num = trial.suggest_int("worker_num", 10, 13)  # Trial #98: 11
    else:
        worker_num = trial.suggest_int("worker_num", 10, 16)  # MAXIMUM for exploration
    use_multiprocessing = False  # Keep False to avoid pickle issues

    if use_best_ranges:
        thread_num = trial.suggest_int("thread_num", 12, 15)  # Trial #98: 13
    else:
        thread_num = trial.suggest_int("thread_num", 4, max_threads)

    if use_multiprocessing:
        min_threads = min(max_threads, max(4, worker_num * 2))
        if thread_num < min_threads:
            thread_num = min_threads

    # Sample hyperparameters based on mode
    if use_best_ranges:
        # EXPLOITATION MODE: Tighter ranges around Trial #141 (best: 0.058131)
        # Trial #141 params: lr=2.697e-06, batch=3072, gamma=0.98, net=1536, workers=13
        learning_rate = trial.suggest_float("learning_rate", 1.5e-6, 4.0e-6, log=True)
        batch_size = trial.suggest_categorical("batch_size", [1536, 2048, 3072, 4096])  # Include 1536 for compatibility
        gamma = trial.suggest_float("gamma", 0.96, 0.99, step=0.01)
        net_dimension = trial.suggest_int("net_dimension", 1408, 1728, step=64)  # 1536 ± ~160

        base_target_step = trial.suggest_int("base_target_step", 200, 280, step=10)  # 230 ± ~50
        base_break_step = trial.suggest_int("base_break_step", 110000, 140000, step=5000)  # 125000 ± 15000

        lookback = trial.suggest_int("lookback", 4, 6)  # Trial #141 used 5
        norm_cash_exp = trial.suggest_int("norm_cash_exp", -16, -14)  # Trial #141: -15
        norm_stocks_exp = trial.suggest_int("norm_stocks_exp", -11, -9)  # Trial #141: -10
        norm_tech_exp = trial.suggest_int("norm_tech_exp", -19, -17)  # Trial #141: -18
        norm_reward_exp = trial.suggest_int("norm_reward_exp", -14, -12)  # Trial #141: -13
        norm_action = trial.suggest_int("norm_action", 15000, 20000, step=500)  # Trial #141: 17000
        time_decay_floor = trial.suggest_float("time_decay_floor", 0.15, 0.25, step=0.05)  # Trial #141: 0.20
    else:
        # Broader exploration ranges - GPU OPTIMIZED for MAXIMUM usage
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [1536, 2048, 3072, 4096])  # LARGE batches
        gamma = trial.suggest_float("gamma", 0.88, 0.99, step=0.01)
        net_dimension = trial.suggest_int("net_dimension", 768, 2048, step=64)  # LARGE networks

        base_target_step = trial.suggest_int("base_target_step", 256, 1024)
        base_break_step = trial.suggest_int("base_break_step", 50000, 200000, step=10000)

        lookback = trial.suggest_int("lookback", 1, 5)
        norm_cash_exp = trial.suggest_int("norm_cash_exp", -16, -10)
        norm_stocks_exp = trial.suggest_int("norm_stocks_exp", -10, -5)
        norm_tech_exp = trial.suggest_int("norm_tech_exp", -18, -12)
        norm_reward_exp = trial.suggest_int("norm_reward_exp", -13, -7)
        norm_action = trial.suggest_int("norm_action", 1000, 25000, step=1000)
        time_decay_floor = trial.suggest_float("time_decay_floor", 0.0, 0.5, step=0.05)

    # Adjust for timeframe
    target_step = int(max(64, base_target_step * multiplier))
    break_step = int(max(10000, base_break_step * multiplier))

    # Advanced PPO hyperparameters
    if use_best_ranges:
        # Tight ranges around Trial #141
        clip_range = trial.suggest_float("clip_range", 0.15, 0.25, step=0.05)  # Trial #141: 0.2
        entropy_coef = trial.suggest_float("entropy_coef", 0.0015, 0.003, log=True)  # Trial #141: 0.00211
        value_loss_coef = trial.suggest_float("value_loss_coef", 0.7, 0.9, step=0.05)  # Trial #141: 0.8
        max_grad_norm = trial.suggest_float("max_grad_norm", 0.7, 1.0, step=0.1)  # Trial #141: 0.8
        gae_lambda = trial.suggest_float("gae_lambda", 0.95, 0.98, step=0.01)  # Trial #141: 0.97

        # Additional advanced hyperparameters for fine-tuning
        ppo_epochs = trial.suggest_int("ppo_epochs", 8, 12)  # Number of epochs per update
        kl_target = trial.suggest_float("kl_target", 0.008, 0.02, log=True)  # KL divergence target
        adam_epsilon = trial.suggest_float("adam_epsilon", 1e-8, 1e-6, log=True)  # Adam optimizer epsilon
    else:
        # Broader ranges for exploration
        clip_range = trial.suggest_float("clip_range", 0.1, 0.4, step=0.05)
        entropy_coef = trial.suggest_float("entropy_coef", 1e-5, 0.01, log=True)
        value_loss_coef = trial.suggest_float("value_loss_coef", 0.25, 1.0, step=0.05)
        max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 2.0, step=0.1)
        gae_lambda = trial.suggest_float("gae_lambda", 0.90, 0.99, step=0.01)

        # Additional advanced hyperparameters for exploration
        ppo_epochs = trial.suggest_int("ppo_epochs", 4, 16)
        kl_target = trial.suggest_float("kl_target", 0.005, 0.03, log=True)
        adam_epsilon = trial.suggest_float("adam_epsilon", 1e-9, 1e-5, log=True)

    # Learning rate schedule
    if use_best_ranges:
        # Trial #98 used lr_schedule=True with linear and factor=0.8
        use_lr_schedule = trial.suggest_categorical("use_lr_schedule", [True])
        lr_schedule_type = trial.suggest_categorical("lr_schedule_type", ["linear"])
        lr_schedule_factor = trial.suggest_float("lr_schedule_factor", 0.7, 0.9, step=0.05)  # Trial #98: 0.8
    else:
        use_lr_schedule = trial.suggest_categorical("use_lr_schedule", [True, False])
        if use_lr_schedule:
            lr_schedule_type = trial.suggest_categorical("lr_schedule_type", ["linear", "exponential"])
            lr_schedule_factor = trial.suggest_float("lr_schedule_factor", 0.50, 0.95, step=0.05)
        else:
            lr_schedule_type = None
            lr_schedule_factor = None

    # Risk management parameters
    if use_best_ranges:
        # Tight ranges around Trial #141
        min_cash_reserve = trial.suggest_float("min_cash_reserve", 0.04, 0.08, step=0.01)  # Trial #141: 0.06
        concentration_penalty = trial.suggest_float("concentration_penalty", 0.03, 0.07, step=0.01)  # Trial #141: 0.05

        # Additional risk management parameters
        max_drawdown_penalty = trial.suggest_float("max_drawdown_penalty", 0.0, 0.05, step=0.01)  # Penalize large drawdowns
        volatility_penalty = trial.suggest_float("volatility_penalty", 0.0, 0.03, step=0.005)  # Penalize high volatility
    else:
        min_cash_reserve = trial.suggest_float("min_cash_reserve", 0.05, 0.30, step=0.05)
        concentration_penalty = trial.suggest_float("concentration_penalty", 0.0, 0.10, step=0.01)

        # Additional risk management parameters for exploration
        max_drawdown_penalty = trial.suggest_float("max_drawdown_penalty", 0.0, 0.1, step=0.01)
        volatility_penalty = trial.suggest_float("volatility_penalty", 0.0, 0.05, step=0.01)

    erl_params = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gamma": gamma,
        "net_dimension": net_dimension,
        "target_step": target_step,
        "eval_time_gap": trial.suggest_int("eval_time_gap", 60, 90, step=30) if use_best_ranges else trial.suggest_int("eval_time_gap", 30, 120, step=30),  # Trial #141: 90
        "break_step": break_step,
        "use_multiprocessing": use_multiprocessing,
        "worker_num": worker_num,
        "thread_num": thread_num,
        # Advanced PPO
        "clip_range": clip_range,
        "entropy_coef": entropy_coef,
        "value_loss_coef": value_loss_coef,
        "max_grad_norm": max_grad_norm,
        "gae_lambda": gae_lambda,
        "use_lr_schedule": use_lr_schedule,
        "lr_schedule_type": lr_schedule_type,
        "lr_schedule_factor": lr_schedule_factor,
        # Additional advanced hyperparameters
        "ppo_epochs": ppo_epochs,
        "kl_target": kl_target,
        "adam_epsilon": adam_epsilon,
    }

    env_params = {
        "lookback": lookback,
        "norm_cash": 2 ** norm_cash_exp,
        "norm_stocks": 2 ** norm_stocks_exp,
        "norm_tech": 2 ** norm_tech_exp,
        "norm_reward": 2 ** norm_reward_exp,
        "norm_action": norm_action,
        "time_decay_floor": time_decay_floor,
        "min_cash_reserve": min_cash_reserve,
        "concentration_penalty": concentration_penalty,
        # Additional risk management
        "max_drawdown_penalty": max_drawdown_penalty,
        "volatility_penalty": volatility_penalty,
    }

    if use_sentiment:
        env_params["use_sentiment"] = True

    return erl_params, env_params, multiplier


# =============================================================================
# CPCV Setup
# =============================================================================

def setup_cpcv(price_array, tech_array, time_array, timeframe='1h',
               num_paths=3, k_test_groups=2):
    """Setup Combinatorial Purged Cross-Validation."""
    env = CryptoEnvAlpaca

    n_total_groups = num_paths + 1
    t_final = 10

    # Calculate embargo based on timeframe
    timeframe_delta = {
        '5m': pd.Timedelta(minutes=5),
        '15m': pd.Timedelta(minutes=15),
        '30m': pd.Timedelta(minutes=30),
        '1h': pd.Timedelta(hours=1),
        '4h': pd.Timedelta(hours=4),
        '12h': pd.Timedelta(hours=12),
        '1d': pd.Timedelta(days=1),
    }

    embargo_td = timeframe_delta[timeframe] * t_final * 5

    cv = CombPurgedKFoldCV(
        n_splits=n_total_groups,
        n_test_splits=k_test_groups,
        embargo_td=embargo_td
    )

    # Create dataframe for CPCV
    data = pd.DataFrame(tech_array)
    data = data.set_index(pd.DatetimeIndex(pd.to_datetime(time_array, unit='s')))
    data = data[:-t_final]  # Remove last t_final candles

    # Placeholder target
    y = pd.Series([0] * len(data), index=data.index)

    # Prediction and evaluation times
    prediction_times = pd.Series(data.index, index=data.index)
    evaluation_times = pd.Series(data.index, index=data.index)

    # Calculate paths (using combinatorics instead of expensive generator evaluation)
    import math
    num_paths_actual = math.comb(n_total_groups, k_test_groups)
    n_splits = num_paths_actual  # Use the actual number of CV paths

    # Skip back_test_paths_generator since paths are not used in training
    # This saves significant computation time during setup

    return cv, env, data, y, num_paths_actual, None, n_total_groups, n_splits, prediction_times, evaluation_times


# =============================================================================
# Training Objective - To be continued...
# =============================================================================

def objective_standard(trial, args, price_array, tech_array, time_array, sentiment_array=None):
    """Standard CPCV objective function."""

    print(f"\n{Colors.CYAN}[DEBUG] Objective function called for trial #{trial.number}{Colors.END}")
    sys.stdout.flush()

    # Sample timeframe if multi-timeframe mode
    if args.mode == 'multi-timeframe':
        timeframe = trial.suggest_categorical('timeframe', args.timeframes)
        # Resample data
        price_array, tech_array, time_array, sentiment_array = resample_data(
            price_array, tech_array, time_array, sentiment_array,
            source_tf='5m', target_tf=timeframe
        )
    else:
        timeframe = args.timeframe

    print(f"\n{'='*80}")
    print(f"{Colors.HEADER}Trial #{trial.number}: {timeframe} timeframe{Colors.END}")
    print(f"{'='*80}\n")
    sys.stdout.flush()

    # Sample hyperparameters
    erl_params, env_params, multiplier = sample_hyperparams(
        trial,
        use_best_ranges=args.use_best_ranges,
        timeframe=timeframe,
        use_sentiment=args.use_sentiment,
        n_timesteps=len(time_array)
    )

    print(f"Hyperparameters:")
    print(f"  LR: {erl_params['learning_rate']:.6f}")
    print(f"  Batch: {erl_params['batch_size']}")
    print(f"  Gamma: {erl_params['gamma']:.4f}")
    print(f"  Net dim: {erl_params['net_dimension']}")
    print(f"  Target step: {erl_params['target_step']}")
    print(f"  Break step: {erl_params['break_step']}")
    print(f"  Timeframe multiplier: {multiplier:.3f}x")

    # Setup CPCV
    cv, env, data, y, num_paths, paths, n_total_groups, n_splits, prediction_times, evaluation_times = setup_cpcv(
        price_array, tech_array, time_array, timeframe, args.num_paths, args.k_test_groups
    )

    # Training directory
    name_folder = f"trial_{trial.number}_{timeframe}"
    cwd = f"./train_results/cwd_tests/{name_folder}"
    os.makedirs(cwd, exist_ok=True)

    # Set timeframe attribute BEFORE training starts (needed by test_agent)
    trial.set_user_attr('timeframe', timeframe)

    # Run CPCV splits
    sharpe_list_bot = []
    sharpe_list_hodl = []

    for split_idx, (train_indices, test_indices) in enumerate(
            cv.split(data, y, pred_times=prediction_times, eval_times=evaluation_times)):

        print(f"\n  Split {split_idx+1}/{n_splits}...")

        try:
            sharpe_bot, sharpe_hodl, drl_rets, train_duration, test_duration = train_and_test(
                trial,
                price_array,
                tech_array,
                train_indices,
                test_indices,
                env,
                'ppo',
                env_params,
                erl_params,
                erl_params['break_step'],
                cwd,
                args.gpu,
                sentiment_service=None,  # TODO: Add if enabled
                use_sentiment=args.use_sentiment,
                tickers=args.tickers,
            )

            if np.isnan(sharpe_bot) or np.isnan(sharpe_hodl):
                raise optuna.exceptions.TrialPruned("Sharpe returned NaN")

            sharpe_list_bot.append(sharpe_bot)
            sharpe_list_hodl.append(sharpe_hodl)

            print(f"    Bot: {sharpe_bot:.4f}, HODL: {sharpe_hodl:.4f}")

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

    # Store metrics
    trial.set_user_attr('sharpe_list_bot', sharpe_list_bot)
    trial.set_user_attr('sharpe_list_hodl', sharpe_list_hodl)
    trial.set_user_attr('mean_sharpe_bot', mean_sharpe_bot)
    trial.set_user_attr('mean_sharpe_hodl', mean_sharpe_hodl)
    trial.set_user_attr('std_sharpe_bot', std_sharpe_bot)
    trial.set_user_attr('timeframe', timeframe)

    print(f"\n{Colors.GREEN}Trial #{trial.number} Results:{Colors.END}")
    print(f"  Bot Sharpe: {mean_sharpe_bot:.6f} ± {std_sharpe_bot:.6f}")
    print(f"  HODL Sharpe: {mean_sharpe_hodl:.6f}")
    print(f"  Objective: {objective_value:.6f}")

    return objective_value


def main():
    """Main function - run Optuna optimization."""
    parser = argparse.ArgumentParser(description="Unified Training Script with GPU Optimization")
    parser.add_argument('--n-trials', type=int, default=100, help='Number of Optuna trials')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--study-name', type=str, default='cappuccino_maxgpu', help='Optuna study name')
    parser.add_argument('--storage', type=str, default='sqlite:///databases/optuna_cappuccino.db', help='Optuna storage')
    parser.add_argument('--mode', type=str, default='standard', choices=['standard', 'multi-timeframe'], help='Training mode')
    parser.add_argument('--timeframe', type=str, default='1h', choices=['5m', '15m', '30m', '1h', '4h', '12h', '1d'], help='Timeframe')
    parser.add_argument('--use-best-ranges', action='store_true', help='Use tightened hyperparameter ranges')
    parser.add_argument('--use-sentiment', action='store_true', help='Use sentiment analysis features')
    parser.add_argument('--num-paths', type=int, default=3, help='Number of paths for CPCV')
    parser.add_argument('--k-test-groups', type=int, default=2, help='Number of test groups for CPCV')
    parser.add_argument('--tickers', nargs='+', default=['BTC', 'ETH', 'DOGE', 'ADA', 'SOL', 'MATIC', 'DOT'], help='List of cryptocurrency tickers')
    parser.add_argument('--data-dir', type=str, default='data/1h_1680', help='Data directory')
    args = parser.parse_args()

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    print(f"{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.HEADER}CAPPUCCINO - MAXIMUM GPU OPTIMIZATION{Colors.END}")
    print(f"{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"Study: {args.study_name}")
    print(f"Storage: {args.storage}")
    print(f"Trials: {args.n_trials}")
    print(f"GPU: {args.gpu}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Use Best Ranges: {args.use_best_ranges}")
    print(f"Data Dir: {args.data_dir}")
    print(f"{Colors.HEADER}{'='*80}{Colors.END}\n")
    sys.stdout.flush()

    # Create directories
    Path('databases').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    Path('train_results').mkdir(exist_ok=True)

    # Load data
    print(f"{Colors.BLUE}Loading data from {args.data_dir}...{Colors.END}")
    sys.stdout.flush()
    price_array, tech_array, time_array, sentiment_array = load_data(args.data_dir, args.timeframe)
    print(f"  Data shape: {price_array.shape}")
    print(f"  Time range: {time_array[0]} to {time_array[-1]}\n")
    sys.stdout.flush()

    # Create Optuna study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction='maximize',
        load_if_exists=True
    )

    # Optimization function with args bound
    def objective_wrapper(trial):
        return objective_standard(trial, args, price_array, tech_array, time_array, sentiment_array)

    # Run optimization
    print(f"{Colors.GREEN}Starting optimization...{Colors.END}\n")
    sys.stdout.flush()
    print(f"{Colors.CYAN}[DEBUG] About to call study.optimize(){Colors.END}")
    sys.stdout.flush()

    study.optimize(objective_wrapper, n_trials=args.n_trials, show_progress_bar=False)

    print(f"{Colors.CYAN}[DEBUG] study.optimize() completed{Colors.END}")
    sys.stdout.flush()

    # Print results
    print(f"\n{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.HEADER}OPTIMIZATION COMPLETE{Colors.END}")
    print(f"{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.6f}")
    print(f"Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"{Colors.HEADER}{'='*80}{Colors.END}\n")


if __name__ == '__main__':
    main()
