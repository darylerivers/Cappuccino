#!/usr/bin/env python3
"""
Real Stress Test for Trial #965 Rerun Model

Tests the actual trained model against 200 CGE-generated scenarios
using the real Cappuccino trading environment.
"""

import sys
import os
sys.path.insert(0, '/opt/user-data/experiment/cappuccino')
sys.path.insert(0, str('/opt/user-data/experiment/cappuccino'.replace('cappuccino', 'ghost/FinRL_Crypto')))

import numpy as np
import pandas as pd
import json
from pathlib import Path
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

# Import Cappuccino components
from environment_Alpaca import CryptoEnvAlpaca
from drl_agents.elegantrl_models import DRLAgent as DRLAgent_erl
from function_finance_metrics import sharpe_iid, compute_eqw

print("="*70)
print("TRIAL #965 REAL STRESS TEST")
print("="*70)

# Configuration
MODEL_PATH = '/opt/user-data/experiment/cappuccino/train_results/cwd_tests/trial_965_rerun_1h'
CGE_DATA_PATH = Path('/opt/user-data/experiment/cappuccino/data/cge_synthetic')
TRIAL_INFO_PATH = '/opt/user-data/experiment/cappuccino/best_trial_info.json'
TIMEFRAME = '1h'
GPU_ID = 0

# Load model configuration
print("\nLoading model configuration...")
with open(TRIAL_INFO_PATH, 'r') as f:
    trial_info = json.load(f)

params = trial_info['params']
print(f"✓ Loaded Trial #{trial_info['trial_number']} config")
print(f"  Original Sharpe: {trial_info['sharpe']:.6f}")
print(f"  Model path: {MODEL_PATH}")

# Check model exists
model_file = f"{MODEL_PATH}/actor.pth"
if not os.path.exists(model_file):
    print(f"✗ Model file not found: {model_file}")
    sys.exit(1)

model_size = os.path.getsize(model_file) / (1024**2)
print(f"✓ Model file found: {model_size:.1f} MB")

# Setup environment parameters
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

# Calculate data points per year for Sharpe
timeframe_minutes = {'5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '12h': 720, '1d': 1440}
minutes_per_year = 365.25 * 24 * 60
data_points_per_year = minutes_per_year / timeframe_minutes[TIMEFRAME]

print(f"\nEnvironment configuration:")
print(f"  Lookback: {env_params['lookback']}")
print(f"  Min cash reserve: {env_params['min_cash_reserve']:.1%}")
print(f"  Trailing stop: {env_params['trailing_stop_pct']:.1%}")
print(f"  Data points/year: {data_points_per_year:.0f}")

# Load CGE scenarios
print("\n" + "="*70)
print("Loading CGE Scenarios")
print("="*70)

macro_df = pd.read_csv(CGE_DATA_PATH / 'macro_scenarios.csv')
print(f"✓ Loaded macro scenarios: {len(macro_df)} scenarios")

scenarios = []
for i in range(200):
    try:
        prices = np.load(CGE_DATA_PATH / f'synthetic_{i:04d}.npy')

        meta_file = CGE_DATA_PATH / f'synthetic_meta_{i:04d}.json'
        with open(meta_file, 'r') as f:
            metadata = json.load(f)

        scenarios.append({
            'id': i,
            'prices': prices,
            'regime': metadata.get('regime', 'unknown'),
            'volatility': metadata.get('volatility', 0),
            'expected_return': metadata.get('expected_return', 0),
            'macro': macro_df.iloc[i].to_dict() if i < len(macro_df) else {}
        })
    except Exception as e:
        print(f"  Warning: Could not load scenario {i}: {e}")
        continue

print(f"✓ Loaded {len(scenarios)} price scenarios")
print(f"  Shape example: {scenarios[0]['prices'].shape}")

# Classify regimes from macro conditions
def classify_regime(macro_state):
    """Classify economic scenario into regime"""
    stress = macro_state.get('financial_stress', 0)
    gdp = macro_state.get('GDP_Growth', 2.5)
    rates = macro_state.get('Interest_Rate', 2.5)
    risk = macro_state.get('Risk_Appetite', 50.0)

    if stress > 0.5:
        return 'crisis'
    elif risk > 70 and rates < 3.5:
        return 'bull'
    elif rates > 5.5 or gdp < 0:
        return 'bear'
    else:
        return 'normal'

print("\nClassifying scenarios by macro conditions...")
for scenario in scenarios:
    scenario['regime'] = classify_regime(scenario['macro'])

regimes = [s['regime'] for s in scenarios]
regime_counts = pd.Series(regimes).value_counts()
print(f"\nRegime distribution:")
for regime, count in regime_counts.items():
    print(f"  {regime}: {count} ({100*count/len(scenarios):.1f}%)")

# Run stress tests
print("\n" + "="*70)
print("RUNNING STRESS TESTS")
print("="*70)
print(f"\nTesting {len(scenarios)} scenarios with real model...")
print("This will take several minutes...")

results = []

for i, scenario in enumerate(scenarios):
    try:
        # Get price data
        prices = scenario['prices']  # Shape: (timesteps, n_assets)

        # Create synthetic technical indicators (simplified)
        # In production, you'd calculate real indicators
        n_timesteps, n_assets = prices.shape
        n_indicators = 9  # high, low, volume, macd, macd_signal, macd_hist, rsi, cci, dx

        # Create basic tech array
        tech_array = np.zeros((n_timesteps, n_assets * n_indicators))
        for j in range(n_assets):
            asset_prices = prices[:, j]

            # High/Low (simple approximation)
            tech_array[:, j*n_indicators + 0] = asset_prices * 1.02  # high
            tech_array[:, j*n_indicators + 1] = asset_prices * 0.98  # low

            # Volume (normalized random)
            tech_array[:, j*n_indicators + 2] = np.random.uniform(0.8, 1.2, n_timesteps)

            # Simple moving average indicators
            sma_20 = np.convolve(asset_prices, np.ones(20)/20, mode='same')
            sma_50 = np.convolve(asset_prices, np.ones(50)/50, mode='same')

            # MACD (simplified)
            tech_array[:, j*n_indicators + 3] = sma_20 - sma_50  # macd
            tech_array[:, j*n_indicators + 4] = np.convolve(tech_array[:, j*n_indicators + 3], np.ones(9)/9, mode='same')  # signal
            tech_array[:, j*n_indicators + 5] = tech_array[:, j*n_indicators + 3] - tech_array[:, j*n_indicators + 4]  # hist

            # RSI (simplified momentum)
            returns = np.diff(asset_prices, prepend=asset_prices[0])
            tech_array[:, j*n_indicators + 6] = (returns > 0).astype(float) * 50 + 25  # rsi

            # CCI and DX (placeholder)
            tech_array[:, j*n_indicators + 7] = np.random.normal(0, 100, n_timesteps)  # cci
            tech_array[:, j*n_indicators + 8] = np.random.uniform(20, 40, n_timesteps)  # dx

        # Create environment
        data_config = {
            "price_array": prices,
            "tech_array": tech_array,
            "if_train": False,
        }

        env = CryptoEnvAlpaca(
            config=data_config,
            env_params=env_params,
            if_log=False,
            sentiment_service=None,
            use_sentiment=False,
            tickers=None
        )

        # Run model
        account_values = DRLAgent_erl.DRL_prediction(
            model_name='ppo',
            cwd=MODEL_PATH,
            net_dimension=params['net_dimension'],
            environment=env,
            gpu_id=GPU_ID
        )

        # Calculate metrics
        lookback = params['lookback']
        indice_start = lookback - 1
        indice_end = len(prices) - lookback

        # Bot returns
        account_array = np.array(account_values)
        base_account = np.maximum(account_array[:-1], 1e-12)
        drl_rets = account_array[1:] / base_account - 1

        # HODL returns
        _, eqw_rets, _ = compute_eqw(prices, indice_start, indice_end)

        # Calculate Sharpe
        dataset_size = len(drl_rets)
        factor = data_points_per_year / dataset_size

        sharpe_bot, _ = sharpe_iid(drl_rets, bench=0, factor=factor, log=False)
        sharpe_hodl, _ = sharpe_iid(eqw_rets, bench=0, factor=factor, log=False)

        # Calculate drawdown
        cumulative = np.cumprod(1 + drl_rets)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()

        # Total return
        total_return = (account_array[-1] / 1000000 - 1) * 100  # Percent return

        # Win rate
        win_rate = (drl_rets > 0).mean()

        result = {
            'scenario_id': scenario['id'],
            'regime': scenario['regime'],
            'sharpe_bot': sharpe_bot,
            'sharpe_hodl': sharpe_hodl,
            'sharpe_diff': sharpe_bot - sharpe_hodl,
            'max_drawdown': max_dd * 100,  # Percentage
            'total_return': total_return,
            'win_rate': win_rate * 100,
            'final_value': account_array[-1],
            **scenario['macro']
        }

        results.append(result)

        if (i + 1) % 25 == 0:
            mean_sharpe = np.mean([r['sharpe_bot'] for r in results])
            print(f"  Progress: {i + 1}/{len(scenarios)} - Mean Sharpe: {mean_sharpe:.3f}")

    except Exception as e:
        print(f"  ✗ Scenario {i} failed: {e}")
        continue

print(f"\n✓ Completed {len(results)}/{len(scenarios)} scenarios")

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save results
output_csv = '/opt/user-data/experiment/cappuccino/stress_test_trial965_results.csv'
results_df.to_csv(output_csv, index=False)
print(f"\n✓ Results saved: {output_csv}")

# Analyze results
print("\n" + "="*70)
print("STRESS TEST RESULTS ANALYSIS")
print("="*70)

print("\n【Overall Performance】")
print(f"  Mean Sharpe (Bot):  {results_df['sharpe_bot'].mean():.3f}")
print(f"  Median Sharpe:      {results_df['sharpe_bot'].median():.3f}")
print(f"  Std Dev:            {results_df['sharpe_bot'].std():.3f}")
print(f"  Min Sharpe:         {results_df['sharpe_bot'].min():.3f}")
print(f"  Max Sharpe:         {results_df['sharpe_bot'].max():.3f}")
print(f"\n  Mean Max Drawdown:  {results_df['max_drawdown'].mean():.1f}%")
print(f"  Worst Drawdown:     {results_df['max_drawdown'].min():.1f}%")
print(f"\n  Mean Win Rate:      {results_df['win_rate'].mean():.1f}%")
print(f"  Mean Return:        {results_df['total_return'].mean():.1f}%")

print("\n【Performance by Regime】")
regime_stats = results_df.groupby('regime').agg({
    'sharpe_bot': ['count', 'mean', 'std', 'min', 'max'],
    'max_drawdown': 'mean',
    'win_rate': 'mean'
}).round(3)
print(regime_stats)

print("\n【Comparison to HODL】")
print(f"  Bot mean Sharpe:    {results_df['sharpe_bot'].mean():.3f}")
print(f"  HODL mean Sharpe:   {results_df['sharpe_hodl'].mean():.3f}")
print(f"  Outperformance:     {results_df['sharpe_diff'].mean():.3f}")
print(f"  Bot wins:           {(results_df['sharpe_bot'] > results_df['sharpe_hodl']).sum()}/{len(results_df)} scenarios ({100*(results_df['sharpe_bot'] > results_df['sharpe_hodl']).mean():.1f}%)")

print("\n【Failure Analysis】")
failures = results_df[results_df['sharpe_bot'] < 0]
print(f"  Negative Sharpe scenarios: {len(failures)}/{len(results_df)} ({100*len(failures)/len(results_df):.1f}%)")

if len(failures) > 0:
    print(f"\n  Worst 5 scenarios:")
    worst = failures.nsmallest(5, 'sharpe_bot')[['scenario_id', 'regime', 'sharpe_bot', 'max_drawdown', 'total_return']]
    print(worst.to_string(index=False))

print("\n【Tail Risk】")
p5 = results_df['sharpe_bot'].quantile(0.05)
p1 = results_df['sharpe_bot'].quantile(0.01)
print(f"  5th percentile Sharpe:  {p5:.3f}")
print(f"  1st percentile Sharpe:  {p1:.3f}")

# Create visualization
print("\n" + "="*70)
print("Creating visualizations...")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Trial #965 CGE Stress Test Results', fontsize=16, fontweight='bold')

# 1. Sharpe distribution
ax1 = axes[0, 0]
ax1.hist(results_df['sharpe_bot'], bins=30, alpha=0.7, color='blue', edgecolor='black')
ax1.axvline(results_df['sharpe_bot'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {results_df["sharpe_bot"].mean():.3f}')
ax1.axvline(0, color='black', linestyle='-', linewidth=1)
ax1.set_xlabel('Sharpe Ratio')
ax1.set_ylabel('Frequency')
ax1.set_title('Bot Sharpe Distribution')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Sharpe by regime
ax2 = axes[0, 1]
results_df.boxplot(column='sharpe_bot', by='regime', ax=ax2)
ax2.set_xlabel('Regime')
ax2.set_ylabel('Sharpe Ratio')
ax2.set_title('Performance by Regime')
ax2.axhline(0, color='red', linestyle='--', linewidth=1)
ax2.get_figure().suptitle('')

# 3. Bot vs HODL
ax3 = axes[0, 2]
ax3.scatter(results_df['sharpe_hodl'], results_df['sharpe_bot'], alpha=0.5)
max_val = max(results_df['sharpe_bot'].max(), results_df['sharpe_hodl'].max())
min_val = min(results_df['sharpe_bot'].min(), results_df['sharpe_hodl'].min())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', label='Equal performance')
ax3.set_xlabel('HODL Sharpe')
ax3.set_ylabel('Bot Sharpe')
ax3.set_title('Bot vs HODL Performance')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Return vs Drawdown
ax4 = axes[1, 0]
colors_map = {'crisis': 'red', 'bear': 'orange', 'normal': 'blue', 'bull': 'green'}
for regime in results_df['regime'].unique():
    data = results_df[results_df['regime'] == regime]
    ax4.scatter(data['max_drawdown'], data['total_return'],
               alpha=0.6, label=regime, c=colors_map.get(regime, 'gray'))
ax4.set_xlabel('Max Drawdown (%)')
ax4.set_ylabel('Total Return (%)')
ax4.set_title('Risk-Return Profile')
ax4.legend()
ax4.grid(alpha=0.3)
ax4.axhline(0, color='black', linestyle='-', linewidth=0.5)

# 5. Win rate distribution
ax5 = axes[1, 1]
ax5.hist(results_df['win_rate'], bins=25, alpha=0.7, color='green', edgecolor='black')
ax5.axvline(results_df['win_rate'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {results_df["win_rate"].mean():.1f}%')
ax5.axvline(50, color='black', linestyle='-', linewidth=1, label='50%')
ax5.set_xlabel('Win Rate (%)')
ax5.set_ylabel('Frequency')
ax5.set_title('Win Rate Distribution')
ax5.legend()
ax5.grid(alpha=0.3)

# 6. Regime performance summary
ax6 = axes[1, 2]
regime_means = results_df.groupby('regime')['sharpe_bot'].mean().sort_values()
colors_bar = [colors_map.get(r, 'gray') for r in regime_means.index]
regime_means.plot(kind='barh', ax=ax6, color=colors_bar)
ax6.set_xlabel('Mean Sharpe Ratio')
ax6.set_title('Average Performance by Regime')
ax6.axvline(0, color='red', linestyle='--', linewidth=1)
ax6.grid(alpha=0.3, axis='x')

plt.tight_layout()

output_png = '/opt/user-data/experiment/cappuccino/stress_test_trial965_results.png'
plt.savefig(output_png, dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved: {output_png}")
plt.close()

print("\n" + "="*70)
print("✓ STRESS TEST COMPLETE!")
print("="*70)

print(f"\nFiles created:")
print(f"  • {output_csv}")
print(f"  • {output_png}")

print("\n【Summary】")
mean_sharpe = results_df['sharpe_bot'].mean()
if mean_sharpe > 1.0:
    print("  ✓ Excellent performance across scenarios!")
elif mean_sharpe > 0.5:
    print("  ✓ Good performance, model shows robustness")
elif mean_sharpe > 0:
    print("  ⚠️  Modest performance, consider improvements")
else:
    print("  ✗ Poor performance, model needs retraining")

print(f"\n  Mean Sharpe: {mean_sharpe:.3f}")
print(f"  Success rate: {100*(results_df['sharpe_bot'] > 0).mean():.1f}% positive Sharpe")
print(f"  Beats HODL: {100*(results_df['sharpe_bot'] > results_df['sharpe_hodl']).mean():.1f}% of time")
