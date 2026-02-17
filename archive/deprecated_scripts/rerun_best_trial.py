#!/usr/bin/env python3
"""
Re-run Trial #965 with exact hyperparameters and force model save.
"""

import json
import os
import sys
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
from function_CPCV import CombPurgedKFoldCV, back_test_paths_generator
from function_train_test import train_and_test

print("="*70)
print("RE-RUNNING TRIAL #965 WITH EXACT HYPERPARAMETERS")
print("="*70)

# Load best trial info
with open('best_trial_info.json', 'r') as f:
    trial_info = json.load(f)

print(f"\nOriginal Trial: #{trial_info['trial_number']}")
print(f"Original Sharpe: {trial_info['sharpe']:.6f}")
print(f"Total Trials: {trial_info['total_trials']}")

# Extract hyperparameters
params = trial_info['params']
print("\nHyperparameters loaded:")
for key, value in list(params.items())[:10]:
    print(f"  {key}: {value}")
print("  ... (and more)")

# Configuration
TIMEFRAME = '1h'
DATA_DIR = Path('data/1h_1680')  # Points to CGE-augmented data
TICKERS = ['AAVE', 'AVAX', 'BTC', 'LINK', 'ETH', 'LTC', 'UNI']
INDICATORS = ['high', 'low', 'volume', 'macd', 'macd_signal', 'macd_hist', 'rsi', 'cci', 'dx']
GPU_ID = 0
SEED = 312

print(f"\nConfiguration:")
print(f"  Data: {DATA_DIR}")
print(f"  Tickers: {TICKERS}")
print(f"  Timeframe: {TIMEFRAME}")
print(f"  GPU: {GPU_ID}")

# Set output directory
cwd = f'train_results/cwd_tests/trial_965_rerun_{TIMEFRAME}'
os.makedirs(cwd, exist_ok=True)
print(f"\nOutput directory: {cwd}")

# Load data
print("\nLoading data...")
file_list = [str(DATA_DIR / f"{ticker}_{TIMEFRAME}.csv") for ticker in TICKERS]
df = pd.DataFrame()
for file in file_list:
    tmp = pd.read_csv(file, index_col=0)
    tmp['tic'] = file.split('/')[-1].split('_')[0]
    df = pd.concat([df, tmp], ignore_index=True)

df = df.sort_values(['date', 'tic']).reset_index(drop=True)
print(f"Data loaded: {len(df)} rows, {df['date'].nunique()} unique dates")

# Setup environment config
env_config = {
    'price_array': 'close',
    'tech_array': INDICATORS,
    'if_train': True,
}

# Extract training parameters
train_config = {
    'gamma': params['gamma'],
    'net_dimension': params['net_dimension'],
    'batch_size': params['batch_size'],
    'target_step': params['base_target_step'],
    'lr': params['learning_rate'],
    'max_memo': int(2**18),  # 262k
    'repeat_times': 1.5,
    'n_step_return': 512,
    'if_per_or_gae': True,
    'break_step': params['base_break_step'],
    'random_seed': SEED,
    'if_allow_break': True,
    'worker_num': params['worker_num'],
    'thread_num': params['thread_num'],
    'lookback': params['lookback'],
    'if_discrete': False,
}

# PPO-specific parameters
ppo_params = {
    'clip_range': params['clip_range'],
    'entropy_coef': params['entropy_coef'],
    'value_loss_coef': params['value_loss_coef'],
    'max_grad_norm': params['max_grad_norm'],
    'gae_lambda': params['gae_lambda'],
    'ppo_epochs': params['ppo_epochs'],
    'kl_target': params['kl_target'],
    'adam_epsilon': params['adam_epsilon'],
}

# Environment parameters
env_params = {
    'min_cash_reserve': params['min_cash_reserve'],
    'concentration_penalty': params['concentration_penalty'],
    'max_drawdown_penalty': params['max_drawdown_penalty'],
    'volatility_penalty': params['volatility_penalty'],
    'trailing_stop_pct': params['trailing_stop_pct'],
}

# Normalization
norm_config = {
    'norm_cash_exp': params['norm_cash_exp'],
    'norm_stocks_exp': params['norm_stocks_exp'],
    'norm_tech_exp': params['norm_tech_exp'],
    'norm_reward_exp': params['norm_reward_exp'],
    'norm_action': params['norm_action'],
}

# Time decay
time_decay_config = {
    'time_decay_floor': params['time_decay_floor'],
}

# Learning rate schedule
lr_schedule = None
if params.get('use_lr_schedule', False):
    lr_schedule = {
        'type': params['lr_schedule_type'],
        'factor': params['lr_schedule_factor'],
    }

print("\nStarting training with CombPurgedKFoldCV...")
print("This will take approximately 50 minutes...")

# Setup cross-validation
cv = CombPurgedKFoldCV(
    n_splits=6,
    n_test_splits=1,
    embargo_td=pd.Timedelta(hours=params['eval_time_gap'])
)

# Run cross-validation
sharpe_list_bot = []
sharpe_list_hodl = []

for fold_idx, (train_idx, test_idx) in enumerate(cv.split(df, pred_times=df['date'], eval_times=df['date'])):
    print(f"\n{'='*70}")
    print(f"FOLD {fold_idx + 1}/6")
    print(f"{'='*70}")

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    print(f"Train: {len(train_df)} rows, {train_df['date'].nunique()} dates")
    print(f"Test:  {len(test_df)} rows, {test_df['date'].nunique()} dates")

    # Train environment
    env_train = CryptoEnvAlpaca(
        df=train_df,
        **env_config,
        **env_params,
        **norm_config,
        **time_decay_config,
        timeframe=TIMEFRAME,
        gpu_id=GPU_ID,
    )

    # Test environment
    env_test = CryptoEnvAlpaca(
        df=test_df,
        **env_config,
        **env_params,
        **norm_config,
        **time_decay_config,
        timeframe=TIMEFRAME,
        gpu_id=GPU_ID,
        if_train=False,
    )

    # Train and test
    sharpe_bot, sharpe_hodl = train_and_test(
        env_train=env_train,
        env_test=env_test,
        agent_name='ppo',
        cwd=cwd,
        **train_config,
        **ppo_params,
        lr_schedule=lr_schedule,
    )

    sharpe_list_bot.append(sharpe_bot)
    sharpe_list_hodl.append(sharpe_hodl)

    print(f"\nFold {fold_idx + 1} Results:")
    print(f"  Bot Sharpe:  {sharpe_bot:.6f}")
    print(f"  HODL Sharpe: {sharpe_hodl:.6f}")

# Calculate final metrics
mean_sharpe_bot = np.mean(sharpe_list_bot)
mean_sharpe_hodl = np.mean(sharpe_list_hodl)
std_sharpe_bot = np.std(sharpe_list_bot)

# Objective: mean_bot - mean_hodl
objective = mean_sharpe_bot - mean_sharpe_hodl

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"Objective Sharpe: {objective:.6f}")
print(f"Bot Sharpe (mean): {mean_sharpe_bot:.6f}")
print(f"Bot Sharpe (std):  {std_sharpe_bot:.6f}")
print(f"HODL Sharpe:       {mean_sharpe_hodl:.6f}")
print(f"Outperformance:    {mean_sharpe_bot / mean_sharpe_hodl:.1f}x")

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

print(f"\nResults saved to: {results_file}")

# Check for model file
actor_path = f'{cwd}/actor.pth'
if os.path.exists(actor_path):
    print(f"✓ Model saved: {actor_path}")
    file_size = os.path.getsize(actor_path) / (1024**2)
    print(f"  File size: {file_size:.1f} MB")
else:
    print(f"⚠️  Model not found at: {actor_path}")
    print("   Checking for other model files...")
    for f in os.listdir(cwd):
        if f.endswith('.pth'):
            print(f"   Found: {f}")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
