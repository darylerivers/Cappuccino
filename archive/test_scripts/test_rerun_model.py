#!/usr/bin/env python3
"""
Test the retrained Trial #965 model and calculate proper Sharpe ratios.
"""

import json
import os
import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

# Fix stdout buffering
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
os.environ['PYTHONUNBUFFERED'] = '1'

# Add parent directory to path
PARENT_DIR = Path(__file__).parent.parent / "ghost/FinRL_Crypto"
sys.path.insert(0, str(PARENT_DIR))

from environment_Alpaca import CryptoEnvAlpaca
from function_CPCV import CombPurgedKFoldCV
from function_finance_metrics import compute_data_points_per_year, sharpe_iid

print("="*70)
print("TESTING TRIAL #965 RERUN MODEL")
print("="*70)

# Load configuration
with open('best_trial_info.json', 'r') as f:
    trial_info = json.load(f)

params = trial_info['params']
TIMEFRAME = '1h'
DATA_DIR = 'data/1h_1680'
GPU_ID = 0
cwd = 'train_results/cwd_tests/trial_965_rerun_1h'

print(f"\nModel directory: {cwd}")
print(f"Timeframe: {TIMEFRAME}")

# Check model exists
model_path = f'{cwd}/actor.pth'
if not os.path.exists(model_path):
    print(f"✗ Model not found: {model_path}")
    sys.exit(1)

model_size = os.path.getsize(model_path) / (1024**2)
print(f"✓ Model found: {model_size:.1f} MB")

# Load data
print("\nLoading data...")
data_path = Path(DATA_DIR)
price_file = data_path / 'price_array'
tech_file = data_path / 'tech_array'
time_file = data_path / 'time_array'

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

# Setup CV for testing
embargo_td = pd.Timedelta(hours=params['eval_time_gap'])
cv = CombPurgedKFoldCV(n_splits=6, n_test_splits=1, embargo_td=embargo_td)

data = pd.DataFrame(tech_array)
y = data.iloc[:, 0]
prediction_times = pd.Series(pd.to_datetime(time_array, unit='s'))
evaluation_times = prediction_times.copy()

# Environment parameters
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

# Calculate data points per year for Sharpe calculation
timeframe_minutes = {'5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '12h': 720, '1d': 1440}
minutes_per_year = 365.25 * 24 * 60
data_points_per_year = minutes_per_year / timeframe_minutes[TIMEFRAME]

print(f"\nData points per year: {data_points_per_year:.0f}")
print("\nTesting model on 6 folds...")
print("="*70)

sharpe_list_bot = []
sharpe_list_hodl = []

# Import agent for testing
from drl_agents.elegantrl_models import DRLAgent as DRLAgent_erl
from function_finance_metrics import compute_eqw

for split_idx, (train_indices, test_indices) in enumerate(
        cv.split(data, y, pred_times=prediction_times, eval_times=evaluation_times)):

    print(f"\nFOLD {split_idx + 1}/6")
    print("-"*70)

    # Create test environment
    test_price = price_array[test_indices]
    test_tech = tech_array[test_indices]

    data_config = {
        "price_array": test_price,
        "tech_array": test_tech,
        "if_train": False,
    }

    env_test = CryptoEnvAlpaca(
        config=data_config,
        env_params=env_params,
        if_log=False,
        sentiment_service=None,
        use_sentiment=False,
        tickers=None
    )

    # Test using DRL_prediction
    account_value_list = DRLAgent_erl.DRL_prediction(
        model_name='ppo',
        cwd=cwd,
        net_dimension=params['net_dimension'],
        environment=env_test,
        gpu_id=GPU_ID
    )

    # Calculate returns
    lookback = params['lookback']
    indice_start = lookback - 1
    indice_end = len(test_price) - lookback

    # Calculate HODL returns
    account_value_eqw, eqw_rets, eqw_cumrets = compute_eqw(test_price, indice_start, indice_end)

    # Calculate bot returns
    account_value_array = np.array(account_value_list)
    base_account = np.maximum(account_value_array[:-1], 1e-12)
    drl_rets = account_value_array[1:] / base_account - 1

    # Calculate Sharpe ratios with proper factor
    dataset_size = np.shape(eqw_rets)[0]
    factor = data_points_per_year / dataset_size

    sharpe_bot, _ = sharpe_iid(drl_rets, bench=0, factor=factor, log=False)
    sharpe_hodl, _ = sharpe_iid(eqw_rets, bench=0, factor=factor, log=False)

    sharpe_list_bot.append(sharpe_bot)
    sharpe_list_hodl.append(sharpe_hodl)

    print(f"  Bot Sharpe:  {sharpe_bot:.6f}")
    print(f"  HODL Sharpe: {sharpe_hodl:.6f}")
    print(f"  Final value: ${account_value_array[-1]:.2f}")

# Calculate final metrics
mean_sharpe_bot = np.mean(sharpe_list_bot)
mean_sharpe_hodl = np.mean(sharpe_list_hodl)
std_sharpe_bot = np.std(sharpe_list_bot)
objective = mean_sharpe_bot - mean_sharpe_hodl

print("\n" + "="*70)
print("FINAL RESULTS - TRIAL #965 RERUN")
print("="*70)
print(f"\nObjective Sharpe:  {objective:.6f}")
print(f"Bot Sharpe (mean): {mean_sharpe_bot:.6f}")
print(f"Bot Sharpe (std):  {std_sharpe_bot:.6f}")
print(f"HODL Sharpe:       {mean_sharpe_hodl:.6f}")
print(f"Outperformance:    {mean_sharpe_bot / mean_sharpe_hodl if mean_sharpe_hodl != 0 else float('inf'):.1f}x")

print(f"\nComparison to original Trial #965:")
print(f"  Original Sharpe: {trial_info['sharpe']:.6f}")
print(f"  Rerun Sharpe:    {objective:.6f}")
print(f"  Match:           {'✓ Close match' if abs(objective - trial_info['sharpe']) < 0.002 else '✗ Different'}")

# Save results
results = {
    'trial_number': '965_rerun',
    'objective': float(objective),
    'mean_sharpe_bot': float(mean_sharpe_bot),
    'mean_sharpe_hodl': float(mean_sharpe_hodl),
    'std_sharpe_bot': float(std_sharpe_bot),
    'sharpe_list_bot': [float(x) for x in sharpe_list_bot],
    'sharpe_list_hodl': [float(x) for x in sharpe_list_hodl],
    'model_path': model_path,
    'hyperparameters': params,
}

results_file = f'{cwd}/test_results.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved: {results_file}")
print(f"✓ Model ready: {model_path}")

print("\n" + "="*70)
print("✓ TESTING COMPLETE!")
print("="*70)
