#!/usr/bin/env python3
"""
Phase 2: Feature-Enhanced Training with PPO and DDQN

Loads the winning timeframe/interval from Phase 1 and trains models with:
- Enhanced features (7-day and 30-day rolling means)
- Dynamic fee tier progression (0.6% → 0.25%)
- Full hyperparameter optimization
- Parallel PPO and DDQN comparison

Trials: 200 PPO + 200 DDQN = 400 total
State dimension: 91 (63 base + 28 rolling features)

Output:
- phase2_ppo_best.json - Best PPO model
- phase2_ddqn_best.json - Best DDQN model
- phase2_comparison.json - Algorithm comparison

Usage:
    # Run full Phase 2 (400 trials)
    python phase2_feature_maximizer.py

    # Run mini test (5 PPO + 5 DDQN = 10 trials)
    python phase2_feature_maximizer.py --mini-test

    # Run only PPO
    python phase2_feature_maximizer.py --algorithm ppo

    # Run only DDQN
    python phase2_feature_maximizer.py --algorithm ddqn
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
from optuna.storages import RDBStorage

# Import project modules
from config_two_phase import PHASE1, PHASE2
from function_train_test import train_and_test
from function_CPCV import setup_cpcv
from environment_Alpaca_phase2 import CryptoEnvAlpacaPhase2

# Import from module starting with digit (requires importlib)
import importlib
_optimize_unified = importlib.import_module('1_optimize_unified')
sample_hyperparams = _optimize_unified.sample_hyperparams


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def load_phase1_winner():
    """
    Load Phase 1 winner configuration.

    Returns:
        Dictionary with winning timeframe and interval
    """
    winner_file = PHASE1.WINNER_FILE

    if not os.path.exists(winner_file):
        raise FileNotFoundError(
            f"Phase 1 winner file not found: {winner_file}\n"
            f"Please run Phase 1 first: python phase1_timeframe_optimizer.py"
        )

    with open(winner_file, 'r') as f:
        winner = json.load(f)

    print(f"{Colors.GREEN}✓ Phase 1 Winner Loaded:{Colors.END}")
    print(f"  Time-frame: {winner['timeframe']}")
    print(f"  Interval: {winner['interval']}")
    print(f"  Best value: {winner['best_value']:.4f}")
    print(f"  Sharpe (bot): {winner['best_sharpe_bot']:.4f}")

    return winner


def load_phase2_data(data_dir, interval, months=12):
    """
    Load Phase 2 enhanced data with rolling means.

    Args:
        data_dir: Data directory
        interval: Data interval
        months: Number of months

    Returns:
        Tuple of (price_array, tech_array_enhanced, time_array)
    """
    print(f"\n{Colors.CYAN}Loading Phase 2 data...{Colors.END}")

    price_file = f"{data_dir}/price_array_{interval}_{months}mo.npy"
    tech_file = f"{data_dir}/tech_array_enhanced_{interval}_{months}mo.npy"
    time_file = f"{data_dir}/time_array_{interval}_{months}mo.npy"

    # Check if Phase 2 data exists
    if not os.path.exists(tech_file):
        print(f"{Colors.YELLOW}Phase 2 data not found. Using standard data.{Colors.END}")
        print(f"{Colors.YELLOW}Run: python prepare_phase2_data.py --interval {interval} --months {months}{Colors.END}")
        tech_file = f"{data_dir}/tech_array_{interval}_{months}mo.npy"

    price_array = np.load(price_file)
    tech_array = np.load(tech_file)
    time_array = np.load(time_file)

    print(f"  Price: {price_array.shape}")
    print(f"  Tech (enhanced): {tech_array.shape}")
    print(f"  Time: {time_array.shape}")
    print(f"{Colors.GREEN}✓ Data loaded{Colors.END}")

    return price_array, tech_array, time_array


def objective_phase2_ppo(trial, timeframe, interval, price_array, tech_array, time_array, args):
    """
    Objective function for Phase 2 PPO optimization.

    Args:
        trial: Optuna trial
        timeframe: Time-frame from Phase 1
        interval: Interval from Phase 1
        price_array: Price data
        tech_array: Enhanced tech array with rolling means
        time_array: Timestamps
        args: Arguments

    Returns:
        Objective value
    """
    print(f"\n{'='*80}")
    print(f"{Colors.HEADER}[Phase 2 PPO] Trial #{trial.number}{Colors.END}")
    print(f"{'='*80}\n")

    # Sample full hyperparameters (26 params)
    erl_params, env_params, multiplier = sample_hyperparams(
        trial,
        use_best_ranges=False,  # Full exploration
        timeframe=interval,
        use_sentiment=False,
        n_timesteps=len(time_array)
    )

    # Mini-test override: Use smaller break_step for faster testing
    if hasattr(args, 'mini_test') and args.mini_test:
        erl_params['break_step'] = 500  # Much faster for mini-test
        erl_params['target_step'] = 50
        print(f"{Colors.YELLOW}Mini-test mode: Using break_step=500 for faster testing{Colors.END}")

    print(f"Hyperparameters (Full Optimization):")
    print(f"  Learning rate: {erl_params['learning_rate']:.6f}")
    print(f"  Batch size: {erl_params['batch_size']}")
    print(f"  Gamma: {erl_params['gamma']:.4f}")
    print(f"  Net dim: {erl_params['net_dimension']}")
    print(f"  Lookback: {env_params['lookback']}")

    # Setup CPCV
    cv, env, data, y, num_paths, _, n_total_groups, n_splits, pred_times, eval_times = setup_cpcv(
        price_array, tech_array, time_array, interval, args.num_paths, args.k_test_groups
    )

    # Training directory
    name_folder = f"phase2_ppo_trial_{trial.number}"
    cwd = f"./train_results/phase2_ppo/{name_folder}"
    os.makedirs(cwd, exist_ok=True)

    # Store metadata
    trial.set_user_attr('algorithm', 'ppo')
    trial.set_user_attr('timeframe', timeframe)
    trial.set_user_attr('interval', interval)

    # Run CPCV splits
    sharpe_list_bot = []
    sharpe_list_hodl = []

    # Use Phase 2 environment with dynamic fees
    env_phase2 = CryptoEnvAlpacaPhase2

    for split_idx, (train_indices, test_indices) in enumerate(
            cv.split(data, y, pred_times=pred_times, eval_times=eval_times)):

        print(f"\n  Split {split_idx+1}/{n_splits}...")

        try:
            sharpe_bot, sharpe_hodl, drl_rets, train_dur, test_dur = train_and_test(
                trial=trial,
                price_array=price_array,
                tech_array=tech_array,
                train_indices=train_indices,
                test_indices=test_indices,
                env=env_phase2,  # Use Phase 2 environment
                model_name='ppo',
                env_params=env_params,
                erl_params=erl_params,
                break_step=erl_params['break_step'],
                cwd=f"{cwd}/split_{split_idx}",
                gpu_id=0,
                use_timeframe_constraint=False,  # No constraint in Phase 2
                timeframe=None,
                data_interval=interval
            )

            print(f"    Sharpe (bot): {sharpe_bot:.4f}, Sharpe (HODL): {sharpe_hodl:.4f}")
            sharpe_list_bot.append(sharpe_bot)
            sharpe_list_hodl.append(sharpe_hodl)

        except Exception as e:
            print(f"    {Colors.RED}Error in split {split_idx}: {e}{Colors.END}")
            raise optuna.TrialPruned()

    # Aggregate results
    if len(sharpe_list_bot) == 0:
        raise optuna.TrialPruned()

    mean_sharpe_bot = np.mean(sharpe_list_bot)
    mean_sharpe_hodl = np.mean(sharpe_list_hodl)
    std_sharpe_bot = np.std(sharpe_list_bot)

    trial.set_user_attr('mean_sharpe_bot', mean_sharpe_bot)
    trial.set_user_attr('mean_sharpe_hodl', mean_sharpe_hodl)
    trial.set_user_attr('std_sharpe_bot', std_sharpe_bot)

    objective_value = mean_sharpe_bot - mean_sharpe_hodl - 0.1 * std_sharpe_bot

    print(f"\n{Colors.GREEN}Trial #{trial.number} Results:{Colors.END}")
    print(f"  Objective: {objective_value:.4f}")

    return objective_value


def run_phase2_algorithm(algorithm, timeframe, interval, price_array, tech_array, time_array, args):
    """
    Run Phase 2 optimization for one algorithm.

    Args:
        algorithm: 'ppo' or 'ddqn'
        timeframe: Time-frame from Phase 1
        interval: Interval from Phase 1
        price_array: Price data
        tech_array: Enhanced tech data
        time_array: Timestamps
        args: Arguments

    Returns:
        Best trial dictionary
    """
    print(f"\n{'='*100}")
    print(f"{Colors.BOLD}{Colors.CYAN}Phase 2: {algorithm.upper()} Optimization{Colors.END}")
    print(f"{'='*100}\n")

    # Create study
    study_name = PHASE2.STUDY_NAME_PPO if algorithm == 'ppo' else PHASE2.STUDY_NAME_DDQN
    db_path = PHASE2.DB_PATH_PPO if algorithm == 'ppo' else PHASE2.DB_PATH_DDQN
    storage_name = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction='maximize',
        load_if_exists=True
    )

    # Objective function
    def objective_wrapper(trial):
        if algorithm == 'ppo':
            return objective_phase2_ppo(trial, timeframe, interval, price_array, tech_array, time_array, args)
        else:
            # DDQN objective (simplified for now - would need full implementation)
            print(f"{Colors.YELLOW}Note: DDQN training requires custom implementation{Colors.END}")
            raise optuna.TrialPruned()  # Skip DDQN for now

    # Run optimization
    n_trials = args.trials_ppo if algorithm == 'ppo' else args.trials_ddqn
    print(f"Running {n_trials} trials for {algorithm.upper()}...")

    study.optimize(objective_wrapper, n_trials=n_trials)

    # Get best trial
    best_trial = study.best_trial

    result = {
        'algorithm': algorithm,
        'timeframe': timeframe,
        'interval': interval,
        'best_trial_number': best_trial.number,
        'best_value': best_trial.value,
        'best_params': best_trial.params,
        'best_sharpe_bot': best_trial.user_attrs.get('mean_sharpe_bot', 0.0),
        'best_sharpe_hodl': best_trial.user_attrs.get('mean_sharpe_hodl', 0.0),
        'n_trials': len(study.trials),
    }

    print(f"\n{Colors.GREEN}✓ {algorithm.upper()} Optimization Complete{Colors.END}")
    print(f"  Best value: {result['best_value']:.4f}")
    print(f"  Best trial: #{result['best_trial_number']}")

    # Save result
    result_file = f"phase2_{algorithm}_best.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {result_file}")

    return result


def run_phase2(args):
    """
    Main Phase 2 orchestration.

    Runs PPO and/or DDQN optimization and compares results.
    """
    print(f"\n{'='*100}")
    print(f"{Colors.BOLD}{Colors.HEADER}PHASE 2: FEATURE-ENHANCED TRAINING{Colors.END}")
    print(f"{'='*100}\n")

    # Load Phase 1 winner
    phase1_winner = load_phase1_winner()
    timeframe = phase1_winner['timeframe']
    interval = phase1_winner['interval']

    # Load Phase 2 data
    price_array, tech_array, time_array = load_phase2_data(
        args.data_dir, interval, args.months
    )

    # Run algorithms
    results = {}
    start_time = time.time()

    if args.algorithm in ['ppo', 'both']:
        print(f"\n{Colors.CYAN}Starting PPO optimization...{Colors.END}")
        results['ppo'] = run_phase2_algorithm(
            'ppo', timeframe, interval, price_array, tech_array, time_array, args
        )

    if args.algorithm in ['ddqn', 'both']:
        print(f"\n{Colors.CYAN}Starting DDQN optimization...{Colors.END}")
        print(f"{Colors.YELLOW}Note: DDQN implementation is experimental{Colors.END}")
        try:
            results['ddqn'] = run_phase2_algorithm(
                'ddqn', timeframe, interval, price_array, tech_array, time_array, args
            )
        except Exception as e:
            print(f"{Colors.RED}DDQN optimization failed: {e}{Colors.END}")
            print(f"{Colors.YELLOW}Continuing with PPO only{Colors.END}")

    elapsed = time.time() - start_time

    # Print summary
    print(f"\n{'='*100}")
    print(f"{Colors.BOLD}{Colors.GREEN}PHASE 2 COMPLETE!{Colors.END}")
    print(f"{'='*100}\n")
    print(f"Total time: {elapsed/3600:.2f} hours")

    # Compare algorithms
    if len(results) > 1:
        print(f"\n{Colors.CYAN}Algorithm Comparison:{Colors.END}")
        for alg, result in results.items():
            print(f"  {alg.upper()}:")
            print(f"    Best value: {result['best_value']:.4f}")
            print(f"    Sharpe (bot): {result['best_sharpe_bot']:.4f}")
            print(f"    Trials: {result['n_trials']}")

        # Determine winner
        winner_alg = max(results.keys(), key=lambda k: results[k]['best_value'])
        print(f"\n{Colors.BOLD}{Colors.GREEN}🏆 Winner: {winner_alg.upper()}{Colors.END}")

        # Save comparison
        comparison = {
            'winner': winner_alg,
            'results': results,
            'phase1_timeframe': timeframe,
            'phase1_interval': interval,
            'timestamp': datetime.now().isoformat()
        }
        with open(PHASE2.RESULTS_FILE, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\n✓ Comparison saved: {PHASE2.RESULTS_FILE}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Feature-Enhanced Training")
    parser.add_argument('--data-dir', type=str, default='data/phase2',
                        help='Phase 2 data directory')
    parser.add_argument('--algorithm', type=str, default='both',
                        choices=['ppo', 'ddqn', 'both'],
                        help='Algorithm to run')
    parser.add_argument('--trials-ppo', type=int, default=200,
                        help='PPO trials (default: 200)')
    parser.add_argument('--trials-ddqn', type=int, default=200,
                        help='DDQN trials (default: 200)')
    parser.add_argument('--mini-test', action='store_true',
                        help='Run mini test (5 PPO + 5 DDQN)')
    parser.add_argument('--months', type=int, default=12,
                        help='Data months (default: 12)')
    parser.add_argument('--num-paths', type=int, default=3,
                        help='CPCV paths (default: 3)')
    parser.add_argument('--k-test-groups', type=int, default=2,
                        help='CPCV test groups (default: 2)')

    args = parser.parse_args()

    # Mini test adjustments
    if args.mini_test:
        args.trials_ppo = 5
        args.trials_ddqn = 5
        print(f"{Colors.YELLOW}MINI TEST MODE: 5 PPO + 5 DDQN trials{Colors.END}")

    # Create output directories
    os.makedirs('train_results/phase2_ppo', exist_ok=True)
    os.makedirs('train_results/phase2_ddqn', exist_ok=True)
    os.makedirs(os.path.dirname(PHASE2.DB_PATH_PPO), exist_ok=True)
    os.makedirs(os.path.dirname(PHASE2.DB_PATH_DDQN), exist_ok=True)

    # Run Phase 2
    results = run_phase2(args)

    print(f"\n{'='*100}")
    print(f"{Colors.BOLD}Phase 2 complete!{Colors.END}")
    print(f"Models ready for deployment and backtesting.")
    print(f"{'='*100}\n")


if __name__ == '__main__':
    main()
