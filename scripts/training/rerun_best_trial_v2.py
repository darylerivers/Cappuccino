#!/usr/bin/env python3
"""
Re-run Trial #965 with exact hyperparameters and force model save.
Uses the same data loading approach as 1_optimize_unified.py
"""

import json
import os
import sys
import pickle
from pathlib import Path
import torch
import numpy as np
import pandas as pd

# Fix stdout buffering
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
os.environ['PYTHONUNBUFFERED'] = '1'

# Add parent directory to path
PARENT_DIR = Path(__file__).parent.parent / "ghost/FinRL_Crypto"
sys.path.insert(0, str(PARENT_DIR))

from environment_Alpaca import CryptoEnvAlpaca
from utils.function_CPCV import CombPurgedKFoldCV, back_test_paths_generator
from utils.function_train_test import train_and_test

print("="*70)
print("RE-RUNNING TRIAL #965 WITH EXACT HYPERPARAMETERS")
print("="*70)

# Load best trial info
with open('best_trial_info.json', 'r') as f:
    trial_info = json.load(f)

print(f"\nOriginal Trial: #{trial_info['trial_number']}")
print(f"Original Sharpe: {trial_info['sharpe']:.6f}")
print(f"Total Trials: {trial_info['total_trials']}")

params = trial_info['params']

# Configuration
TIMEFRAME = '1h'
DATA_DIR = 'data/1h_1680'  # Points to CGE-augmented data
GPU_ID = 0
SEED = 312

print(f"\nConfiguration:")
print(f"  Data: {DATA_DIR}")
print(f"  Timeframe: {TIMEFRAME}")
print(f"  GPU: {GPU_ID}")

# Set output directory
cwd = f'train_results/cwd_tests/trial_965_rerun_{TIMEFRAME}'
os.makedirs(cwd, exist_ok=True)
print(f"  Output: {cwd}")

# Load data using same method as 1_optimize_unified.py
print("\nLoading data...")
data_path = Path(DATA_DIR)

price_file = data_path / 'price_array'
tech_file = data_path / 'tech_array'
time_file = data_path / 'time_array'

# Custom unpickler for numpy compatibility
class NumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
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

# Setup CPCV
print("\nSetting up cross-validation...")
env = CryptoEnvAlpaca
n_total_groups = 3 + 1  # num_paths + 1
t_final = 10
embargo_td = pd.Timedelta(hours=params['eval_time_gap'])

cv = CombPurgedKFoldCV(
    n_splits=6,
    n_test_splits=1,
    embargo_td=embargo_td
)

# Create dataframe for CV splitting
data = pd.DataFrame(tech_array)
y = data.iloc[:, 0]  # Dummy target
prediction_times = pd.Series(pd.to_datetime(time_array, unit='s'))
evaluation_times = prediction_times.copy()

print(f"  CV folds: 6")
print(f"  Embargo: {embargo_td}")
n_splits = 6  # Known from CV configuration

# Setup environment parameters
# Note: Convert exponent parameters to actual values using 2**exp
env_params = {
    'min_cash_reserve': params['min_cash_reserve'],
    'concentration_penalty': params['concentration_penalty'],
    'max_drawdown_penalty': params['max_drawdown_penalty'],
    'volatility_penalty': params['volatility_penalty'],
    'trailing_stop_pct': params['trailing_stop_pct'],
    'norm_cash': 2 ** params['norm_cash_exp'],
    'norm_stocks': 2 ** params['norm_stocks_exp'],
    'norm_tech': 2 ** params['norm_tech_exp'],
    'norm_reward': 2 ** params['norm_reward_exp'],
    'norm_action': params['norm_action'],
    'time_decay_floor': params['time_decay_floor'],
    'lookback': params['lookback'],
}

# Setup ElegantRL parameters
erl_params = {
    'gamma': params['gamma'],
    'net_dimension': params['net_dimension'],
    'batch_size': params['batch_size'],
    'target_step': params['base_target_step'],
    'learning_rate': params['learning_rate'],
    'eval_time_gap': params['eval_time_gap'],
    'max_memo': int(2**18),
    'repeat_times': 1.5,
    'n_step_return': 512,
    'if_per_or_gae': True,
    'break_step': params['base_break_step'],
    'random_seed': SEED,
    'if_allow_break': True,
    'use_multiprocessing': False,  # Add this parameter
    'worker_num': params['worker_num'],
    'thread_num': params['thread_num'],
    'clip_range': params['clip_range'],
    'entropy_coef': params['entropy_coef'],
    'value_loss_coef': params['value_loss_coef'],
    'max_grad_norm': params['max_grad_norm'],
    'gae_lambda': params['gae_lambda'],
    'use_lr_schedule': params.get('use_lr_schedule', False),
    'lr_schedule_type': params.get('lr_schedule_type', 'linear'),
    'lr_schedule_factor': params.get('lr_schedule_factor', 0.75),
    'ppo_epochs': params['ppo_epochs'],
    'kl_target': params['kl_target'],
    'adam_epsilon': params['adam_epsilon'],
}

print(f"\nKey Hyperparameters:")
print(f"  Learning Rate: {erl_params['learning_rate']:.2e}")
print(f"  Batch Size: {erl_params['batch_size']}")
print(f"  Net Dim: {erl_params['net_dimension']}")
print(f"  Gamma: {erl_params['gamma']}")
print(f"  Target Step: {erl_params['target_step']}")
print(f"  Break Step: {erl_params['break_step']}")

print("\nStarting training...")
print("This will take approximately 50 minutes for 6 folds...")
print("="*70)

# Run cross-validation
sharpe_list_bot = []
sharpe_list_hodl = []

for split_idx, (train_indices, test_indices) in enumerate(
        cv.split(data, y, pred_times=prediction_times, eval_times=evaluation_times)):

    print(f"\nFOLD {split_idx + 1}/{n_splits}")
    print("-"*70)
    print(f"Train samples: {len(train_indices)}")
    print(f"Test samples:  {len(test_indices)}")

    try:
        sharpe_bot, sharpe_hodl, drl_rets, train_duration, test_duration = train_and_test(
            trial=None,  # No trial object needed for standalone run
            price_array=price_array,
            tech_array=tech_array,
            train_indices=train_indices,
            test_indices=test_indices,
            env=env,
            model_name='ppo',
            env_params=env_params,
            erl_params=erl_params,
            break_step=erl_params['break_step'],
            cwd=cwd,
            gpu_id=GPU_ID,
            sentiment_service=None,
            use_sentiment=False,
            tickers=None,
        )

        if np.isnan(sharpe_bot) or np.isnan(sharpe_hodl):
            print(f"  ⚠️  NaN Sharpe detected, skipping fold")
            continue

        sharpe_list_bot.append(sharpe_bot)
        sharpe_list_hodl.append(sharpe_hodl)

        print(f"  ✓ Bot Sharpe:  {sharpe_bot:.6f}")
        print(f"  ✓ HODL Sharpe: {sharpe_hodl:.6f}")
        print(f"  Duration: train={train_duration:.1f}s, test={test_duration:.1f}s")

    except Exception as e:
        print(f"  ✗ Fold {split_idx + 1} failed: {e}")
        import traceback
        traceback.print_exc()
        continue

if not sharpe_list_bot:
    print("\n✗ ERROR: All folds failed!")
    sys.exit(1)

# Calculate final metrics
mean_sharpe_bot = np.mean(sharpe_list_bot)
mean_sharpe_hodl = np.mean(sharpe_list_hodl)
std_sharpe_bot = np.std(sharpe_list_bot)
objective = mean_sharpe_bot - mean_sharpe_hodl

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"Objective Sharpe:  {objective:.6f}")
print(f"Bot Sharpe (mean): {mean_sharpe_bot:.6f}")
print(f"Bot Sharpe (std):  {std_sharpe_bot:.6f}")
print(f"HODL Sharpe:       {mean_sharpe_hodl:.6f}")
print(f"Outperformance:    {mean_sharpe_bot / mean_sharpe_hodl if mean_sharpe_hodl != 0 else float('inf'):.1f}x")
print(f"Completed folds:   {len(sharpe_list_bot)}/{n_splits}")

# Save results
results = {
    'trial_number': '965_rerun',
    'objective': float(objective),
    'mean_sharpe_bot': float(mean_sharpe_bot),
    'mean_sharpe_hodl': float(mean_sharpe_hodl),
    'std_sharpe_bot': float(std_sharpe_bot),
    'sharpe_list_bot': [float(x) for x in sharpe_list_bot],
    'sharpe_list_hodl': [float(x) for x in sharpe_list_hodl],
    'hyperparameters': params,
}

results_file = f'{cwd}/results.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved: {results_file}")

# Check for model files
print("\nModel files:")
actor_path = f'{cwd}/actor.pth'
for filename in ['actor.pth', 'critic.pth', 'act_target.pth', 'cri_target.pth']:
    filepath = f'{cwd}/{filename}'
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024**2)
        print(f"  ✓ {filename} ({size_mb:.1f} MB)")
    else:
        print(f"  ✗ {filename} (not found)")

# List all .pth files
pth_files = [f for f in os.listdir(cwd) if f.endswith('.pth')]
if pth_files:
    print(f"\nAll .pth files in {cwd}:")
    for f in pth_files:
        size_mb = os.path.getsize(f'{cwd}/{f}') / (1024**2)
        print(f"  • {f} ({size_mb:.1f} MB)")

print("\n" + "="*70)
print("✓ TRAINING COMPLETE!")
print("="*70)
