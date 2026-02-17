#!/usr/bin/env python3
"""
Phase 1: Time-Frame Optimization

Systematically searches across time-frame and interval combinations to find
the optimal trading horizon. Tests 25 combinations (5 timeframes × 5 intervals)
with simplified hyperparameter search to focus on validating time-frame constraints.

Time-frames: 3d, 5d, 7d, 10d, 14d
Intervals: 5m, 15m, 30m, 1h, 4h
Trials per combination: 20
Total trials: 500

Output: phase1_winner.json with best time-frame/interval combination

Usage:
    # Run full Phase 1 (500 trials)
    python phase1_timeframe_optimizer.py

    # Run mini test (2 combos, 5 trials each)
    python phase1_timeframe_optimizer.py --mini-test

    # Resume from checkpoint
    python phase1_timeframe_optimizer.py --resume phase1_checkpoint.json
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
from config_two_phase import PHASE1, TIMEFRAME
from function_train_test import train_and_test
from function_CPCV import setup_cpcv
from environment_Alpaca import CryptoEnvAlpaca


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def sample_phase1_hyperparams(trial, timeframe, interval):
    """
    Sample simplified hyperparameters for Phase 1.

    Uses fixed parameters for non-critical settings and optimizes only
    essential hyperparameters to focus on timeframe validation.

    Args:
        trial: Optuna trial
        timeframe: Time-frame constraint (e.g., '3d', '5d')
        interval: Data interval (e.g., '5m', '1h')

    Returns:
        Tuple of (erl_params, env_params)
    """
    # Simplified hyperparameter search
    erl_params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-4, log=True),
        'batch_size': trial.suggest_categorical('batch_size', PHASE1.BATCH_SIZES),
        'gamma': trial.suggest_float('gamma', PHASE1.GAMMA_MIN, PHASE1.GAMMA_MAX, step=PHASE1.GAMMA_STEP),
        'net_dimension': trial.suggest_int('net_dimension', PHASE1.NET_DIM_MIN, PHASE1.NET_DIM_MAX, step=PHASE1.NET_DIM_STEP),

        # Fixed parameters
        'target_step': PHASE1.FIXED_TARGET_STEP,
        'break_step': PHASE1.FIXED_BREAK_STEP,
        'worker_num': PHASE1.FIXED_WORKER_NUM,
        'thread_num': PHASE1.FIXED_THREAD_NUM,
        'eval_time_gap': 60,

        # Default PPO parameters
        'clip_range': 0.2,
        'entropy_coef': 0.01,
        'value_loss_coef': 0.5,
        'max_grad_norm': 0.5,
        'gae_lambda': 0.95,
        'ppo_epochs': 10,
        'kl_target': 0.02,
        'adam_epsilon': 1e-7,
        'use_lr_schedule': False,
    }

    env_params = {
        'lookback': trial.suggest_int('lookback', PHASE1.LOOKBACK_MIN, PHASE1.LOOKBACK_MAX),
        'norm_cash': 2**-11,
        'norm_stocks': 2**-8,
        'norm_tech': 2**-14,
        'norm_reward': 2**-9,
        'norm_action': 100.0,
        'time_decay_floor': 0.0,
        'min_cash_reserve': 0.05,
        'concentration_penalty': 0.0,
        'trailing_stop_pct': trial.suggest_float('trailing_stop_pct',
                                                   PHASE1.TRAILING_STOP_MIN,
                                                   PHASE1.TRAILING_STOP_MAX,
                                                   step=PHASE1.TRAILING_STOP_STEP),
    }

    return erl_params, env_params


def objective_phase1(trial, timeframe, interval, price_array, tech_array, time_array, args):
    """
    Objective function for Phase 1 timeframe optimization.

    Args:
        trial: Optuna trial
        timeframe: Time-frame constraint
        interval: Data interval
        price_array: Price data
        tech_array: Technical indicators
        time_array: Timestamps
        args: Command-line arguments

    Returns:
        Objective value (higher is better)
    """
    print(f"\n{'='*80}")
    print(f"{Colors.HEADER}[Phase 1] Trial #{trial.number}: {timeframe} @ {interval}{Colors.END}")
    print(f"{'='*80}\n")

    # Sample hyperparameters
    erl_params, env_params = sample_phase1_hyperparams(trial, timeframe, interval)

    print(f"Hyperparameters:")
    print(f"  Learning rate: {erl_params['learning_rate']:.6f}")
    print(f"  Batch size: {erl_params['batch_size']}")
    print(f"  Gamma: {erl_params['gamma']:.4f}")
    print(f"  Net dim: {erl_params['net_dimension']}")
    print(f"  Lookback: {env_params['lookback']}")
    print(f"  Trailing stop: {env_params['trailing_stop_pct']:.2%}")
    print(f"\nTimeframe Constraint:")
    print(f"  Time-frame: {timeframe}")
    print(f"  Interval: {interval}")
    candles = TIMEFRAME.get_candles_in_timeframe(timeframe, interval)
    print(f"  Max candles: {candles}")

    # Setup CPCV
    cv, env, data, y, num_paths, _, n_total_groups, n_splits, pred_times, eval_times = setup_cpcv(
        price_array, tech_array, time_array, interval, args.num_paths, args.k_test_groups
    )

    # Training directory
    name_folder = f"phase1_trial_{trial.number}_{timeframe}_{interval}"
    cwd = f"./train_results/phase1/{name_folder}"
    os.makedirs(cwd, exist_ok=True)

    # Store trial metadata
    trial.set_user_attr('timeframe', timeframe)
    trial.set_user_attr('interval', interval)
    trial.set_user_attr('max_candles', candles)

    # Run CPCV splits
    sharpe_list_bot = []
    sharpe_list_hodl = []

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
                env=env,
                model_name='ppo',
                env_params=env_params,
                erl_params=erl_params,
                break_step=erl_params['break_step'],
                cwd=f"{cwd}/split_{split_idx}",
                gpu_id=0,
                use_timeframe_constraint=True,  # Enable timeframe constraint
                timeframe=timeframe,
                data_interval=interval
            )

            print(f"    Sharpe (bot): {sharpe_bot:.4f}, Sharpe (HODL): {sharpe_hodl:.4f}")
            print(f"    Training: {train_dur:.1f}s, Testing: {test_dur:.1f}s")

            sharpe_list_bot.append(sharpe_bot)
            sharpe_list_hodl.append(sharpe_hodl)

        except Exception as e:
            print(f"    {Colors.RED}Error in split {split_idx}: {e}{Colors.END}")
            import traceback
            traceback.print_exc()
            # Prune trial on error
            raise optuna.TrialPruned()

    # Aggregate results
    if len(sharpe_list_bot) == 0:
        raise optuna.TrialPruned()

    mean_sharpe_bot = np.mean(sharpe_list_bot)
    mean_sharpe_hodl = np.mean(sharpe_list_hodl)
    std_sharpe_bot = np.std(sharpe_list_bot)

    # Store aggregated metrics
    trial.set_user_attr('mean_sharpe_bot', mean_sharpe_bot)
    trial.set_user_attr('mean_sharpe_hodl', mean_sharpe_hodl)
    trial.set_user_attr('std_sharpe_bot', std_sharpe_bot)
    trial.set_user_attr('sharpe_list_bot', sharpe_list_bot)
    trial.set_user_attr('sharpe_list_hodl', sharpe_list_hodl)

    # Objective: maximize outperformance vs HODL, penalize inconsistency
    objective_value = mean_sharpe_bot - mean_sharpe_hodl - 0.1 * std_sharpe_bot

    print(f"\n{Colors.GREEN}Trial #{trial.number} Results:{Colors.END}")
    print(f"  Mean Sharpe (bot): {mean_sharpe_bot:.4f}")
    print(f"  Mean Sharpe (HODL): {mean_sharpe_hodl:.4f}")
    print(f"  Std Sharpe (bot): {std_sharpe_bot:.4f}")
    print(f"  Objective: {objective_value:.4f}")

    return objective_value


def run_combination(timeframe, interval, price_array, tech_array, time_array, args):
    """
    Run Optuna optimization for a specific timeframe/interval combination.

    Args:
        timeframe: Time-frame constraint
        interval: Data interval
        price_array: Price data
        tech_array: Technical indicators
        time_array: Timestamps
        args: Command-line arguments

    Returns:
        Dictionary with results
    """
    print(f"\n{'='*100}")
    print(f"{Colors.BOLD}{Colors.CYAN}Starting Combination: {timeframe} @ {interval}{Colors.END}")
    print(f"{'='*100}\n")

    # Create Optuna study
    study_name = f"phase1_{timeframe}_{interval}"
    storage_name = f"sqlite:///{PHASE1.DB_PATH}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction='maximize',
        load_if_exists=True
    )

    # Objective wrapper
    def objective_wrapper(trial):
        return objective_phase1(trial, timeframe, interval, price_array, tech_array, time_array, args)

    # Run optimization
    n_trials = args.trials_per_combo
    print(f"Running {n_trials} trials for {timeframe} @ {interval}...")

    study.optimize(objective_wrapper, n_trials=n_trials)

    # Get best trial
    best_trial = study.best_trial

    result = {
        'timeframe': timeframe,
        'interval': interval,
        'best_trial_number': best_trial.number,
        'best_value': best_trial.value,
        'best_params': best_trial.params,
        'best_sharpe_bot': best_trial.user_attrs.get('mean_sharpe_bot', 0.0),
        'best_sharpe_hodl': best_trial.user_attrs.get('mean_sharpe_hodl', 0.0),
        'n_trials': len(study.trials),
    }

    print(f"\n{Colors.GREEN}✓ Combination Complete: {timeframe} @ {interval}{Colors.END}")
    print(f"  Best value: {result['best_value']:.4f}")
    print(f"  Best trial: #{result['best_trial_number']}")

    return result


def run_phase1(args):
    """
    Main Phase 1 orchestration function.

    Runs optimization across all timeframe/interval combinations
    and selects the best one.
    """
    print(f"\n{'='*100}")
    print(f"{Colors.BOLD}{Colors.HEADER}PHASE 1: TIME-FRAME OPTIMIZATION{Colors.END}")
    print(f"{'='*100}\n")

    # Load data
    print("Loading data...")
    data_dir = args.data_dir
    interval_default = '1h'  # Use 1h data for Phase 1
    months = 12

    price_file = f"{data_dir}/price_array_{interval_default}_{months}mo.npy"
    tech_file = f"{data_dir}/tech_array_{interval_default}_{months}mo.npy"
    time_file = f"{data_dir}/time_array_{interval_default}_{months}mo.npy"

    price_array = np.load(price_file)
    tech_array = np.load(tech_file)
    time_array = np.load(time_file)

    print(f"✓ Data loaded: {price_array.shape[0]} timesteps, {price_array.shape[1]} assets")

    # Get combinations
    if args.mini_test:
        timeframes = ['5d', '7d']
        intervals = ['1h']
        print(f"\n{Colors.YELLOW}MINI TEST MODE: 2 combinations × {args.trials_per_combo} trials{Colors.END}\n")
    else:
        timeframes = PHASE1.TIME_FRAMES
        intervals = PHASE1.INTERVALS

    print(f"Time-frames: {list(timeframes)}")
    print(f"Intervals: {list(intervals)}")
    print(f"Trials per combination: {args.trials_per_combo}")
    print(f"Total combinations: {len(timeframes) * len(intervals)}")
    print(f"Total trials: {len(timeframes) * len(intervals) * args.trials_per_combo}")

    # Run combinations
    results = []
    start_time = time.time()

    for timeframe in timeframes:
        for interval in intervals:
            combo_result = run_combination(
                timeframe, interval, price_array, tech_array, time_array, args
            )
            results.append(combo_result)

            # Save checkpoint
            checkpoint = {
                'results': results,
                'completed_combinations': len(results),
                'total_combinations': len(timeframes) * len(intervals),
                'timestamp': datetime.now().isoformat()
            }
            checkpoint_file = 'phase1_checkpoint.json'
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)

            print(f"\n  Checkpoint saved: {checkpoint_file}")

    # Select winner
    results.sort(key=lambda x: x['best_value'], reverse=True)
    winner = results[0]

    elapsed = time.time() - start_time

    # Print summary
    print(f"\n{'='*100}")
    print(f"{Colors.BOLD}{Colors.GREEN}PHASE 1 COMPLETE!{Colors.END}")
    print(f"{'='*100}\n")
    print(f"Total time: {elapsed/3600:.2f} hours")
    print(f"Combinations tested: {len(results)}")
    print(f"\n{Colors.CYAN}Top 5 Combinations:{Colors.END}")
    for i, result in enumerate(results[:5]):
        print(f"  {i+1}. {result['timeframe']} @ {result['interval']}: "
              f"value={result['best_value']:.4f}, "
              f"sharpe={result['best_sharpe_bot']:.4f}")

    print(f"\n{Colors.BOLD}{Colors.GREEN}🏆 WINNER: {winner['timeframe']} @ {winner['interval']}{Colors.END}")
    print(f"  Best value: {winner['best_value']:.4f}")
    print(f"  Sharpe (bot): {winner['best_sharpe_bot']:.4f}")
    print(f"  Sharpe (HODL): {winner['best_sharpe_hodl']:.4f}")

    # Save winner
    winner_file = PHASE1.WINNER_FILE
    with open(winner_file, 'w') as f:
        json.dump(winner, f, indent=2)

    print(f"\n✓ Winner saved: {winner_file}")

    # Save full results
    results_file = 'phase1_all_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ All results saved: {results_file}")

    return winner


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Time-Frame Optimization")
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--trials-per-combo', type=int, default=20,
                        help='Trials per combination (default: 20)')
    parser.add_argument('--mini-test', action='store_true',
                        help='Run mini test with 2 combinations')
    parser.add_argument('--num-paths', type=int, default=3,
                        help='CPCV paths (default: 3)')
    parser.add_argument('--k-test-groups', type=int, default=2,
                        help='CPCV test groups (default: 2)')

    args = parser.parse_args()

    # Create output directories
    os.makedirs('train_results/phase1', exist_ok=True)
    os.makedirs(os.path.dirname(PHASE1.DB_PATH), exist_ok=True)

    # Run Phase 1
    winner = run_phase1(args)

    print(f"\n{'='*100}")
    print(f"{Colors.BOLD}Phase 1 optimization complete!{Colors.END}")
    print(f"Proceed to Phase 2 with: python phase2_feature_maximizer.py")
    print(f"{'='*100}\n")


if __name__ == '__main__':
    main()
