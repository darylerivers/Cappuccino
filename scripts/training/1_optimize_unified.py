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

import os
import sys
from pathlib import Path
# Fix stdout buffering issues
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
os.environ['PYTHONUNBUFFERED'] = '1'

# Add parent directory to path for imports (MUST be before importing local modules)
PARENT_DIR = Path(__file__).parent.parent.parent  # Points to /opt/user-data/experiment/cappuccino
sys.path.insert(0, str(PARENT_DIR))

from environment_Alpaca import CryptoEnvAlpaca
from environment_Alpaca_vectorized import VectorizedCryptoEnvAlpacaOptimized
from environment_Alpaca_batch_vectorized import BatchVectorizedCryptoEnv
from environment_Alpaca_gpu import GPUBatchCryptoEnv
from utils.function_CPCV import CombPurgedKFoldCV, back_test_paths_generator
from utils.function_train_test import train_and_test
from utils.memory_monitor import check_memory, get_monitor

import argparse
import itertools as itt
import joblib
import optuna
import pickle
import torch
from optuna.trial import TrialState
import traceback
import time

import numpy as np
import pandas as pd
import gc
import psutil
from datetime import datetime, timedelta

# === GPU VRAM MONITORING ===
if torch.cuda.is_available():
    try:
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU VRAM available: {vram_total:.1f}GB")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"Warning: Could not query GPU: {e}")


def cleanup_gpu_memory():
    """Aggressively free GPU memory between trials."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    gc.collect()
    gc.collect()
    # Force Python's allocator to return free pages to the OS.
    # Without this, RSS stays high even after gc.collect(), causing
    # the memory-limit check to fire on every subsequent trial.
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def check_vram_usage(label=""):
    """Print current VRAM usage for debugging."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  [VRAM {label}] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.1f}GB")


# Discord notifications for training progress
try:
    from dotenv import load_dotenv
    load_dotenv()
    from integrations.discord_notifier import DiscordNotifier
    from constants import DISCORD
    discord_notifier = DiscordNotifier() if DISCORD.ENABLED and DISCORD.NOTIFY_TRAINING else None
except Exception as e:
    discord_notifier = None
    print(f"⚠️  Discord training notifications disabled: {e}")

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
                      use_sentiment=False, n_timesteps=None, args=None):
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

    # MEMORY-SAFE: Limit workers to prevent RAM exhaustion
    # Balance between parallelism and RAM/VRAM usage
    if use_best_ranges:
        worker_num = trial.suggest_int("worker_num", 2, 3)  # Conservative for stability
    else:
        worker_num = trial.suggest_int("worker_num", 2, 3)  # Conservative for stability
    use_multiprocessing = False  # Keep False to avoid pickle issues

    if use_best_ranges:
        thread_num = trial.suggest_int("thread_num", 4, 8)  # Reduced for RAM efficiency
    else:
        thread_num = trial.suggest_int("thread_num", 4, 8)  # Reduced for RAM efficiency

    if use_multiprocessing:
        min_threads = min(max_threads, max(4, worker_num * 2))
        if thread_num < min_threads:
            thread_num = min_threads

    # =============================================================================
    # FT-Transformer Configuration (MUST sample BEFORE lookback!)
    # =============================================================================
    # TEMPORARY WORKAROUND: Disable FT-Transformer until bug is fixed
    # TODO: Re-enable after fixing dimension mismatch issue
    use_ft_encoder = False
    use_pretrained = False

    # Original code (disabled):
    # if hasattr(args, 'force_ft') and args.force_ft:
    #     use_ft_encoder = True
    # elif hasattr(args, 'force_baseline') and args.force_baseline:
    #     use_ft_encoder = False
    # else:
    #     use_ft_encoder = trial.suggest_categorical("use_ft_encoder", [False, True])
    # if use_ft_encoder:
    #     use_pretrained = trial.suggest_categorical("ft_use_pretrained", [True, False])
    # else:
    #     use_pretrained = False

    # Sample hyperparameters based on mode
    if use_best_ranges:
        # EXPLOITATION MODE: Tighter ranges around Trial #141 (best: 0.058131)
        # Trial #141 params: lr=2.697e-06, batch=3072, gamma=0.98, net=1536, workers=13
        learning_rate = trial.suggest_float("learning_rate", 1.5e-6, 4.0e-6, log=True)
        batch_size = trial.suggest_categorical("batch_size", [2048, 4096])  # Ultra-low for RAM constraints
        gamma = trial.suggest_float("gamma", 0.96, 0.99, step=0.01)
        net_dimension = trial.suggest_int("net_dimension", 512, 1024, step=128)  # Reduced for RAM constraints

        base_target_step = trial.suggest_int("base_target_step", 200, 280, step=10)  # 230 ± ~50
        base_break_step = trial.suggest_int("base_break_step", 110000, 140000, step=5000)  # 125000 ± 15000

        # Lookback: Fixed at 10 if using pre-trained FT encoder, otherwise sample
        if use_ft_encoder and use_pretrained:
            lookback = 10  # MUST match pre-trained encoder
        else:
            lookback = trial.suggest_int("lookback", 4, 6)  # Trial #141 used 5
        norm_cash_exp = trial.suggest_int("norm_cash_exp", -16, -14)  # Trial #141: -15
        norm_stocks_exp = trial.suggest_int("norm_stocks_exp", -11, -9)  # Trial #141: -10
        norm_tech_exp = trial.suggest_int("norm_tech_exp", -19, -17)  # Trial #141: -18
        norm_reward_exp = trial.suggest_int("norm_reward_exp", -14, -12)  # Trial #141: -13
        norm_action = trial.suggest_int("norm_action", 15000, 20000, step=500)  # Trial #141: 17000
        time_decay_floor = trial.suggest_float("time_decay_floor", 0.15, 0.25, step=0.05)  # Trial #141: 0.20
    else:
        # Broader exploration ranges - VRAM OPTIMIZED for better GPU utilization
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [2048, 4096, 8192])  # Reduced max for GPU stability (was 16384)
        gamma = trial.suggest_float("gamma", 0.88, 0.99, step=0.01)
        net_dimension = trial.suggest_int("net_dimension", 512, 1536, step=256)  # Reduced max for GPU stability (was 2048)

        # Balanced exploration: Sufficient training time for learning
        base_target_step = trial.suggest_int("base_target_step", 256, 1024, step=128)
        base_break_step = trial.suggest_int("base_break_step", 30000, 60000, step=5000)

        # Lookback: Fixed at 10 if using pre-trained FT encoder, otherwise sample
        if use_ft_encoder and use_pretrained:
            lookback = 10  # MUST match pre-trained encoder
        else:
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

    # =============================================================================
    # FT-Transformer Architecture Configuration
    # =============================================================================
    # (use_ft_encoder and use_pretrained already sampled above)

    if use_ft_encoder:
        # FT-Transformer hyperparameters
        # Based on A/B test: pre-trained encoder with lookback=10 performed best

        # Load pre-trained encoder path if using pre-trained
        if use_pretrained:
            # Find most recent pre-trained encoder
            pretrained_dir = Path("train_results/pretrained_encoders")
            if pretrained_dir.exists():
                encoder_dirs = sorted(pretrained_dir.glob("ft_encoder_*"))
                if encoder_dirs:
                    pretrained_encoder_path = str(encoder_dirs[-1] / "best_encoder.pth")
                else:
                    pretrained_encoder_path = None
                    use_pretrained = False  # No pre-trained encoder found
            else:
                pretrained_encoder_path = None
                use_pretrained = False
        else:
            pretrained_encoder_path = None

        # FT-Transformer architecture
        if use_best_ranges:
            # Conservative ranges (from A/B test winner)
            ft_d_token = trial.suggest_categorical("ft_d_token", [32, 64])
            ft_n_blocks = trial.suggest_int("ft_n_blocks", 1, 2)
            ft_n_heads = trial.suggest_categorical("ft_n_heads", [2, 4])
            ft_dropout = trial.suggest_float("ft_dropout", 0.05, 0.15, step=0.05)
            ft_freeze_encoder = trial.suggest_categorical("ft_freeze_encoder", [False, True])
        else:
            # Broader exploration
            ft_d_token = trial.suggest_categorical("ft_d_token", [16, 32, 64, 96])
            ft_n_blocks = trial.suggest_int("ft_n_blocks", 1, 3)
            ft_n_heads = trial.suggest_categorical("ft_n_heads", [2, 4, 8])
            ft_dropout = trial.suggest_float("ft_dropout", 0.0, 0.3, step=0.05)
            ft_freeze_encoder = trial.suggest_categorical("ft_freeze_encoder", [False, True])

        ft_config = {
            'd_token': ft_d_token,
            'n_blocks': ft_n_blocks,
            'n_heads': ft_n_heads,
            'dropout': ft_dropout
        }
    else:
        use_pretrained = False
        pretrained_encoder_path = None
        ft_freeze_encoder = False
        ft_config = None

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
        # FT-Transformer configuration
        "use_ft_encoder": use_ft_encoder,
        "ft_config": ft_config,
        "pretrained_encoder_path": pretrained_encoder_path if use_ft_encoder else None,
        "ft_freeze_encoder": ft_freeze_encoder if use_ft_encoder else False,
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
        # Vectorization
        "n_envs": getattr(args, 'n_envs', 1),  # Number of parallel environments
    }

    if use_sentiment:
        env_params["use_sentiment"] = True

    return erl_params, env_params, multiplier


# =============================================================================
# CPCV Setup
# =============================================================================

def setup_cpcv(price_array, tech_array, time_array, timeframe='1h',
               num_paths=3, k_test_groups=2, n_envs=1):
    """Setup Combinatorial Purged Cross-Validation."""
    # DEBUG: Print n_envs value
    print(f"{Colors.YELLOW}[DEBUG] setup_cpcv called with n_envs={n_envs}, type={type(n_envs)}{Colors.END}")
    sys.stdout.flush()

    # Use GPU-accelerated environment if n_envs > 1, otherwise standard env
    if n_envs > 1:
        print(f"{Colors.GREEN}Using GPU-Accelerated Environment: {n_envs} parallel envs (PyTorch on GPU){Colors.END}")
        env = GPUBatchCryptoEnv
    else:
        print(f"{Colors.YELLOW}Using Standard Environment (n_envs={n_envs}){Colors.END}")
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

    # Memory safety check at trial start.
    # If RAM is below threshold, sleep up to 3 times (90s total) to let it recover
    # before raising TrialPruned.  This prevents the rapid-fire prune cascade where
    # low RAM after a crash causes hundreds of instant prunes per minute.
    for _mem_wait in range(3):
        mem_avail_gb = psutil.virtual_memory().available / (1024 ** 3)
        if mem_avail_gb >= 2.0:
            break
        print(f"{Colors.YELLOW}[Trial #{trial.number}] Low RAM ({mem_avail_gb:.1f} GB free), "
              f"sleeping 30s before starting (attempt {_mem_wait+1}/3)...{Colors.END}")
        sys.stdout.flush()
        time.sleep(30)
    check_memory(trial, f"[Trial #{trial.number} start]", safe_threshold_gb=2.0)

    # Track trial start time for timeout detection
    trial_start_time = time.time()
    max_trial_duration = 30 * 60  # 30 minutes maximum per trial
    process = psutil.Process()  # create once; reused for memory checks inside the loop

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

    # Clean up GPU memory from previous trial
    cleanup_gpu_memory()

    print(f"\n{'='*80}")
    print(f"{Colors.HEADER}Trial #{trial.number}: {timeframe} timeframe{Colors.END}")
    print(f"{'='*80}\n")
    check_vram_usage("pre-trial")
    sys.stdout.flush()

    # Sample hyperparameters
    erl_params, env_params, multiplier = sample_hyperparams(
        trial,
        use_best_ranges=args.use_best_ranges,
        timeframe=timeframe,
        use_sentiment=args.use_sentiment,
        n_timesteps=len(time_array),
        args=args
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
        price_array, tech_array, time_array, timeframe, args.num_paths, args.k_test_groups, args.n_envs
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

        # TIMEOUT CHECK: Prevent infinite trials
        elapsed_time = time.time() - trial_start_time
        if elapsed_time > max_trial_duration:
            print(f"{Colors.RED}Trial timeout after {elapsed_time/60:.1f} minutes{Colors.END}")
            raise optuna.exceptions.TrialPruned(f"Trial timeout: {elapsed_time/60:.1f} minutes exceeded {max_trial_duration/60} minute limit")

        # MEMORY CHECK: Prevent memory leaks
        mem_gb = process.memory_info().rss / (1024**3)
        if mem_gb > 10.0:  # 8 GB limit per process
            print(f"{Colors.RED}Memory limit exceeded: {mem_gb:.1f}GB{Colors.END}")
            cleanup_gpu_memory()
            raise optuna.exceptions.TrialPruned(f"Memory limit exceeded: {mem_gb:.1f}GB > 8GB")

        # Memory safety check before each split
        check_memory(trial, f"[Trial #{trial.number} split {split_idx+1}/{n_splits}]", safe_threshold_gb=2.0)

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

        except ValueError as e:
            # Handle state dimension mismatch gracefully
            if "State dimension mismatch" in str(e):
                print(f"    {Colors.YELLOW}Split {split_idx} skipped: Different state dimension (lookback mismatch){Colors.END}")
            else:
                print(f"    {Colors.RED}Split {split_idx} failed: {e}{Colors.END}")
                traceback.print_exc()
            trial.report(float('nan'), step=trial.number)  # Log failure
            cleanup_gpu_memory()
            continue
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    {Colors.RED}Split {split_idx} OOM! Cleaning up and pruning trial.{Colors.END}")
                cleanup_gpu_memory()
                raise optuna.exceptions.TrialPruned(f"GPU OOM at split {split_idx}: net_dim={erl_params['net_dimension']}, batch={erl_params['batch_size']}")
            print(f"    {Colors.RED}Split {split_idx} failed (RuntimeError): {e}{Colors.END}")
            print(f"{Colors.YELLOW}=== FULL TRACEBACK ==={Colors.END}")
            traceback.print_exc()
            print(f"{Colors.YELLOW}=== END TRACEBACK ==={Colors.END}")
            sys.stdout.flush()
            trial.report(float('nan'), step=trial.number)
            cleanup_gpu_memory()
            continue
        except Exception as e:
            print(f"    {Colors.RED}Split {split_idx} failed: {e}{Colors.END}")
            # Print full traceback for debugging
            print(f"{Colors.YELLOW}Full traceback:{Colors.END}")
            traceback.print_exc()
            trial.report(float('nan'), step=trial.number)  # Log failure
            cleanup_gpu_memory()
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
    check_vram_usage("post-trial")

    # AGGRESSIVE CLEANUP: Prevent memory accumulation
    cleanup_gpu_memory()

    # Force multiple GC passes to clear cyclic references
    gc.collect()
    gc.collect()
    gc.collect()

    # Log final memory state
    final_mem_gb = process.memory_info().rss / (1024**3)
    print(f"{Colors.CYAN}Trial #{trial.number} final memory: {final_mem_gb:.2f}GB{Colors.END}")

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
    parser.add_argument('--force-ft', action='store_true', help='Force FT-Transformer encoder (for FT-only studies)')
    parser.add_argument('--force-baseline', action='store_true', help='Force baseline MLP (no FT-Transformer, for ensemble)')
    parser.add_argument('--num-paths', type=int, default=3, help='Number of paths for CPCV')
    parser.add_argument('--k-test-groups', type=int, default=2, help='Number of test groups for CPCV')
    parser.add_argument('--tickers', nargs='+', default=['BTC', 'ETH', 'DOGE', 'ADA', 'SOL', 'MATIC', 'DOT'], help='List of cryptocurrency tickers')
    parser.add_argument('--data-dir', type=str, default='data/1h_1680', help='Data directory')
    parser.add_argument('--n-envs', type=int, default=1, help='Number of parallel vectorized environments (1=disable, 8-16 recommended for GPU speedup)')
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

    # Store target trials for dashboard
    study.set_user_attr('target_trials', args.n_trials)

    # Optimization function with args bound
    def objective_wrapper(trial):
        return objective_standard(trial, args, price_array, tech_array, time_array, sentiment_array)

    # Discord notification callback
    def discord_callback(study, trial):
        """Send Discord notification after each trial completes."""
        if discord_notifier and trial.state == TrialState.COMPLETE:
            # Send every 10 trials to avoid spam
            if trial.number % 10 == 0 or trial.number < 5:
                try:
                    # Get best trial so far
                    best_trial = study.best_trial
                    best_sharpe = best_trial.value if best_trial else 0.0

                    # Calculate progress
                    total_trials = study.trials
                    complete_trials = [t for t in total_trials if t.state == TrialState.COMPLETE]
                    progress_pct = len(complete_trials) / args.n_trials * 100 if args.n_trials > 0 else 0

                    # Determine study type
                    study_type = "🤖 FT" if "ft_transformer" in args.study_name.lower() else "📊 Ensemble"

                    # Send notification
                    discord_notifier.send_message(
                        content='',
                        embed={
                            'title': f'{study_type} Training Progress',
                            'description': f'Study: {args.study_name}',
                            'color': 0x00ff00 if trial.value > 0 else 0xff9900,
                            'fields': [
                                {'name': '🎯 Progress', 'value': f'{len(complete_trials)}/{args.n_trials} trials ({progress_pct:.1f}%)', 'inline': False},
                                {'name': '📊 This Trial', 'value': f'Trial #{trial.number}\nSharpe: {trial.value:.4f}', 'inline': True},
                                {'name': '🏆 Best So Far', 'value': f'Trial #{best_trial.number if best_trial else "N/A"}\nSharpe: {best_sharpe:.4f}', 'inline': True},
                            ],
                            'timestamp': datetime.now().isoformat()
                        }
                    )
                except Exception as e:
                    print(f"⚠️  Discord notification failed: {e}")

    # Run optimization
    print(f"{Colors.GREEN}Starting optimization...{Colors.END}\n")
    sys.stdout.flush()
    print(f"{Colors.CYAN}[DEBUG] About to call study.optimize(){Colors.END}")
    sys.stdout.flush()

    try:
        study.optimize(objective_wrapper, n_trials=args.n_trials, show_progress_bar=True, callbacks=[discord_callback])
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"\n{Colors.RED}GPU Out of Memory Error: {e}{Colors.END}")
            print("Attempting to free up GPU memory and retry...")
            torch.cuda.empty_cache()
            study.optimize(objective_wrapper, n_trials=args.n_trials, show_progress_bar=True, callbacks=[discord_callback])
        else:
            raise e

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
