#!/usr/bin/env python3
"""
FT-Transformer A/B Comparison Test

Compares three approaches:
1. Baseline: No FT-Transformer, lookback=60
2. FT-Transformer Pre-trained: With pre-trained encoder, lookback=10
3. FT-Transformer From Scratch: No pre-training, lookback=60 (small config to fit in memory)

All trials use identical hyperparameters for fair comparison.
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Add parent directory to path
PARENT_DIR = Path(__file__).parent
sys.path.insert(0, str(PARENT_DIR))

from environment_Alpaca import CryptoEnvAlpaca
from drl_agents.elegantrl_models import DRLAgent
from drl_agents.agents import AgentPPO, AgentPPO_FT
from utils.function_train_test import train_and_test
import pickle


class NumpyUnpickler(pickle.Unpickler):
    """Handle numpy backward compatibility."""
    def find_class(self, module, name):
        if module == 'numpy._core.numeric':
            module = 'numpy.core.numeric'
        elif module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)


def load_data(data_dir):
    """Load price and tech arrays."""
    data_path = Path(data_dir)

    with open(data_path / 'price_array', 'rb') as f:
        price_array = NumpyUnpickler(f).load()
    with open(data_path / 'tech_array', 'rb') as f:
        tech_array = NumpyUnpickler(f).load()
    with open(data_path / 'time_array', 'rb') as f:
        time_array = NumpyUnpickler(f).load()

    return price_array, tech_array, time_array


def create_env_params(lookback):
    """Create environment parameters."""
    return {
        'lookback': lookback,
        'norm_cash': 2e-5,
        'norm_stocks': 1.0,
        'norm_tech': 1e-4,
        'norm_reward': 2**-7,
        'norm_action': 2000
    }


def run_trial(trial_name, config, price_array, tech_array, gpu_id=0):
    """Run a single trial with given configuration."""
    print("\n" + "="*70)
    print(f"TRIAL: {trial_name}")
    print("="*70)
    print(f"Config: {json.dumps(config, indent=2)}")
    print("="*70 + "\n")

    # Create environment parameters
    env_params = create_env_params(config['lookback'])

    # Split data: 80% train, 20% test
    split_idx = int(len(price_array) * 0.8)
    price_train = price_array[:split_idx]
    tech_train = tech_array[:split_idx]
    price_test = price_array[split_idx:]
    tech_test = tech_array[split_idx:]

    print(f"Data split:")
    print(f"  Train: {len(price_train)} timesteps")
    print(f"  Test:  {len(price_test)} timesteps")
    print(f"  Lookback: {config['lookback']}")
    print(f"  State dim: {1 + 7 + (98 * config['lookback'])}")

    # Create DRL agent
    env_config = {
        'price_array': price_train,
        'tech_array': tech_train,
        'if_train': True
    }

    env = CryptoEnvAlpaca(
        config=env_config,
        env_params=env_params,
        if_log=False
    )

    # Model hyperparameters (same for all trials)
    model_kwargs = {
        'learning_rate': 3e-4,
        'batch_size': 128,
        'gamma': 0.99,
        'net_dimension': 512,
        'target_step': 2048,
        'eval_time_gap': 30,
        'thread_num': 4,
        'worker_num': 1,
        'use_multiprocessing': False
    }

    # Create agent based on config
    if config['use_ft_encoder']:
        print(f"\nUsing FT-Transformer encoder")
        print(f"  Pre-trained: {config.get('pretrained_encoder_path') is not None}")
        print(f"  FT config: {config['ft_config']}")

        agent_class = AgentPPO_FT

        # Create mock args object
        class Args:
            use_ft_encoder = True
            ft_config = config['ft_config']
            pretrained_encoder_path = config.get('pretrained_encoder_path')
            freeze_encoder = config.get('freeze_encoder', False)

        args = Args()
    else:
        print(f"\nUsing standard MLP (baseline)")
        agent_class = AgentPPO
        args = None

    # Create agent
    agent = DRLAgent(
        env=CryptoEnvAlpaca,
        price_array=price_train,
        tech_array=tech_train,
        env_params=env_params,
        if_log=False
    )

    model = agent.get_model(
        model_name='ppo',
        gpu_id=gpu_id,
        model_kwargs=model_kwargs
    )

    # Override agent class if using FT-Transformer
    if config['use_ft_encoder']:
        model.agent = agent_class
        model.args = args

    # Train
    print(f"\n{'='*70}")
    print(f"Training for {config['total_timesteps']} timesteps...")
    print(f"{'='*70}\n")

    cwd = f"train_results/ft_comparison/{trial_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    Path(cwd).mkdir(parents=True, exist_ok=True)

    try:
        agent.train_model(
            model=model,
            cwd=cwd,
            total_timesteps=config['total_timesteps']
        )

        # Test
        print(f"\n{'='*70}")
        print(f"Testing on held-out data...")
        print(f"{'='*70}\n")

        # Create test environment
        env_config_test = {
            'price_array': price_test,
            'tech_array': tech_test,
            'if_train': False
        }

        env_test = CryptoEnvAlpaca(
            config=env_config_test,
            env_params=env_params,
            if_log=True
        )

        # Run test
        total_assets = agent.DRL_prediction(
            model_name='ppo',
            cwd=cwd,
            net_dimension=model_kwargs['net_dimension'],
            environment=env_test,
            gpu_id=gpu_id
        )

        # Calculate metrics
        final_return = total_assets[-1] / env_test.initial_total_asset
        returns = np.diff(total_assets) / total_assets[:-1]
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24)  # Annualized for hourly data
        max_drawdown = np.min(np.minimum.accumulate(total_assets) / np.maximum.accumulate(total_assets) - 1)

        results = {
            'trial_name': trial_name,
            'config': config,
            'final_return': float(final_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'total_timesteps': config['total_timesteps'],
            'cwd': cwd,
            'success': True
        }

        print(f"\n{'='*70}")
        print(f"RESULTS: {trial_name}")
        print(f"{'='*70}")
        print(f"Final Return:  {(final_return - 1) * 100:+.2f}%")
        print(f"Sharpe Ratio:  {sharpe_ratio:.4f}")
        print(f"Max Drawdown:  {max_drawdown * 100:.2f}%")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\n❌ Trial failed with error: {e}")
        import traceback
        traceback.print_exc()

        results = {
            'trial_name': trial_name,
            'config': config,
            'success': False,
            'error': str(e)
        }

    return results


def main():
    parser = argparse.ArgumentParser(description='FT-Transformer A/B Comparison')
    parser.add_argument('--data-dir', type=str, default='data/1h_1680',
                        help='Data directory')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID (-1 for CPU)')
    parser.add_argument('--timesteps', type=int, default=50000,
                        help='Training timesteps per trial')
    parser.add_argument('--pretrained-encoder', type=str,
                        default='train_results/pretrained_encoders/ft_encoder_20260206_175932/best_encoder.pth',
                        help='Path to pre-trained encoder')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("FT-TRANSFORMER A/B COMPARISON TEST")
    print("="*70)
    print(f"\nData directory: {args.data_dir}")
    print(f"GPU: {args.gpu}")
    print(f"Timesteps per trial: {args.timesteps:,}")
    print(f"Pre-trained encoder: {args.pretrained_encoder}")
    print("\n" + "="*70 + "\n")

    # Load data
    print("Loading data...")
    price_array, tech_array, time_array = load_data(args.data_dir)
    print(f"✓ Loaded {len(price_array)} timesteps\n")

    # Define trials
    trials = [
        {
            'name': 'baseline',
            'config': {
                'use_ft_encoder': False,
                'lookback': 60,
                'total_timesteps': args.timesteps
            }
        },
        {
            'name': 'ft_pretrained',
            'config': {
                'use_ft_encoder': True,
                'lookback': 10,
                'ft_config': {
                    'd_token': 32,
                    'n_blocks': 2,
                    'n_heads': 4,
                    'dropout': 0.1
                },
                'pretrained_encoder_path': args.pretrained_encoder,
                'freeze_encoder': False,
                'total_timesteps': args.timesteps
            }
        },
        {
            'name': 'ft_from_scratch',
            'config': {
                'use_ft_encoder': True,
                'lookback': 60,
                'ft_config': {
                    'd_token': 16,  # Smaller to fit in memory with lookback=60
                    'n_blocks': 1,
                    'n_heads': 2,
                    'dropout': 0.1
                },
                'pretrained_encoder_path': None,
                'freeze_encoder': False,
                'total_timesteps': args.timesteps
            }
        }
    ]

    # Run trials
    all_results = []

    for trial in trials:
        result = run_trial(
            trial_name=trial['name'],
            config=trial['config'],
            price_array=price_array,
            tech_array=tech_array,
            gpu_id=args.gpu
        )
        all_results.append(result)

    # Save results
    results_file = f"train_results/ft_comparison/comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(results_file).parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70 + "\n")

    successful_results = [r for r in all_results if r.get('success', False)]

    if successful_results:
        # Create comparison table
        df_data = []
        for r in successful_results:
            df_data.append({
                'Trial': r['trial_name'],
                'Lookback': r['config']['lookback'],
                'FT-Encoder': 'Yes' if r['config']['use_ft_encoder'] else 'No',
                'Pre-trained': 'Yes' if r['config'].get('pretrained_encoder_path') else 'No',
                'Return (%)': f"{(r['final_return'] - 1) * 100:+.2f}",
                'Sharpe': f"{r['sharpe_ratio']:.4f}",
                'Max DD (%)': f"{r['max_drawdown'] * 100:.2f}"
            })

        df = pd.DataFrame(df_data)
        print(df.to_string(index=False))
        print("\n")

        # Find winner
        best_idx = np.argmax([r['final_return'] for r in successful_results])
        winner = successful_results[best_idx]
        print(f"🏆 Best Performer: {winner['trial_name'].upper()}")
        print(f"   Return: {(winner['final_return'] - 1) * 100:+.2f}%")
        print(f"   Sharpe: {winner['sharpe_ratio']:.4f}")
        print("\n")

    print(f"Results saved to: {results_file}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
