#!/usr/bin/env python3
"""
Analyze Fundamental Fixes Test Results

Compares the 50-trial validation test against baseline performance to determine
if the fundamental fixes improved model performance.

Usage:
    python analyze_test_results.py --study cappuccino_fundamentals_test_20251215
    python analyze_test_results.py --baseline cappuccino_3workers_20251102_2325
"""

import argparse
import sqlite3
from pathlib import Path
from collections import defaultdict
import numpy as np


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def get_study_trials(db_path: str, study_name: str):
    """Get all completed trials from a study."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get study ID
    cursor.execute("SELECT study_id FROM studies WHERE study_name = ?", (study_name,))
    result = cursor.fetchone()
    if not result:
        conn.close()
        return None

    study_id = result[0]

    # Get completed trials
    cursor.execute("""
        SELECT
            t.trial_id,
            t.number,
            t.value,
            t.state
        FROM trials t
        WHERE t.study_id = ? AND t.state = 'COMPLETE'
        ORDER BY t.number
    """, (study_id,))

    trials = cursor.fetchall()

    # Get trial parameters and attributes
    trial_data = []
    for trial_id, number, value, state in trials:
        # Get parameters
        cursor.execute("""
            SELECT param_name, param_value
            FROM trial_params
            WHERE trial_id = ?
        """, (trial_id,))
        params = dict(cursor.fetchall())

        # Get user attributes
        cursor.execute("""
            SELECT key, value_json
            FROM trial_user_attributes
            WHERE trial_id = ?
        """, (trial_id,))
        attrs = dict(cursor.fetchall())

        trial_data.append({
            'trial_id': trial_id,
            'number': number,
            'value': value,
            'params': params,
            'attrs': attrs,
        })

    conn.close()
    return trial_data


def analyze_study(trials, study_name):
    """Analyze trial statistics."""
    if not trials:
        return None

    values = [t['value'] for t in trials]

    # Extract specific metrics from user attributes
    sharpe_bots = []
    sharpe_hodls = []
    min_cash_reserves = []
    concentration_penalties = []

    for t in trials:
        # Try to parse user attributes (stored as JSON strings)
        attrs = t.get('attrs', {})

        # Extract mean_sharpe_bot if available
        if 'mean_sharpe_bot' in attrs:
            try:
                sharpe_bot = float(attrs['mean_sharpe_bot'].strip('"'))
                sharpe_bots.append(sharpe_bot)
            except:
                pass

        if 'mean_sharpe_hodl' in attrs:
            try:
                sharpe_hodl = float(attrs['mean_sharpe_hodl'].strip('"'))
                sharpe_hodls.append(sharpe_hodl)
            except:
                pass

        # Extract hyperparameters
        params = t.get('params', {})
        if 'min_cash_reserve' in params:
            try:
                min_cash_reserves.append(float(params['min_cash_reserve']))
            except:
                pass

        if 'concentration_penalty' in params:
            try:
                concentration_penalties.append(float(params['concentration_penalty']))
            except:
                pass

    stats = {
        'name': study_name,
        'n_trials': len(trials),
        'mean_objective': np.mean(values),
        'std_objective': np.std(values),
        'max_objective': np.max(values),
        'min_objective': np.min(values),
        'median_objective': np.median(values),
    }

    if sharpe_bots:
        stats['mean_sharpe_bot'] = np.mean(sharpe_bots)
        stats['std_sharpe_bot'] = np.std(sharpe_bots)

    if sharpe_hodls:
        stats['mean_sharpe_hodl'] = np.mean(sharpe_hodls)

    if min_cash_reserves:
        stats['mean_min_cash'] = np.mean(min_cash_reserves)

    if concentration_penalties:
        stats['mean_concentration_penalty'] = np.mean(concentration_penalties)

    return stats


def compare_studies(test_stats, baseline_stats):
    """Compare test study against baseline."""
    print(f"\n{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.HEADER}COMPARISON: TEST vs BASELINE{Colors.END}")
    print(f"{Colors.HEADER}{'='*80}{Colors.END}\n")

    print(f"{Colors.CYAN}Study Statistics:{Colors.END}\n")

    # Objective comparison
    print(f"{'Metric':<30} {'Test':>15} {'Baseline':>15} {'Improvement':>15}")
    print("-" * 80)

    def print_metric(name, test_val, baseline_val, is_higher_better=True):
        if test_val is None or baseline_val is None:
            return

        diff = test_val - baseline_val
        pct_change = (diff / abs(baseline_val) * 100) if baseline_val != 0 else 0

        # Determine if improvement
        is_improvement = (diff > 0) if is_higher_better else (diff < 0)
        color = Colors.GREEN if is_improvement else Colors.RED
        sign = '+' if diff > 0 else ''

        print(f"{name:<30} {test_val:>15.6f} {baseline_val:>15.6f} "
              f"{color}{sign}{diff:>7.6f} ({sign}{pct_change:.2f}%){Colors.END}")

    print_metric("Mean Objective", test_stats['mean_objective'], baseline_stats['mean_objective'])
    print_metric("Max Objective", test_stats['max_objective'], baseline_stats['max_objective'])
    print_metric("Std Objective", test_stats['std_objective'], baseline_stats['std_objective'], is_higher_better=False)

    if 'mean_sharpe_bot' in test_stats and 'mean_sharpe_bot' in baseline_stats:
        print_metric("Mean Sharpe (Bot)", test_stats['mean_sharpe_bot'], baseline_stats['mean_sharpe_bot'])

    print()

    # Hyperparameter comparison
    if 'mean_min_cash' in test_stats and 'mean_min_cash' in baseline_stats:
        print(f"{Colors.CYAN}Hyperparameter Trends:{Colors.END}\n")
        print(f"{'Parameter':<30} {'Test':>15} {'Baseline':>15}")
        print("-" * 80)
        print(f"{'Mean Min Cash Reserve':<30} {test_stats['mean_min_cash']:>15.4f} {baseline_stats['mean_min_cash']:>15.4f}")
        if 'mean_concentration_penalty' in test_stats:
            print(f"{'Mean Concentration Penalty':<30} {test_stats['mean_concentration_penalty']:>15.4f} {baseline_stats.get('mean_concentration_penalty', 0):>15.4f}")
        print()

    # Overall assessment
    print(f"{Colors.HEADER}ASSESSMENT:{Colors.END}\n")

    obj_improvement = test_stats['mean_objective'] - baseline_stats['mean_objective']
    obj_improvement_pct = (obj_improvement / abs(baseline_stats['mean_objective']) * 100) if baseline_stats['mean_objective'] != 0 else 0

    if obj_improvement > 0.01:  # Significant improvement threshold
        print(f"{Colors.GREEN}✓ SIGNIFICANT IMPROVEMENT{Colors.END}")
        print(f"  Test study shows {obj_improvement_pct:.2f}% improvement over baseline")
        print(f"  Mean objective: {test_stats['mean_objective']:.6f} vs {baseline_stats['mean_objective']:.6f}")
        print(f"\n{Colors.GREEN}Recommendation: Proceed to Step 3 (rolling mean features){Colors.END}")

    elif obj_improvement > 0:
        print(f"{Colors.YELLOW}~ MARGINAL IMPROVEMENT{Colors.END}")
        print(f"  Test study shows {obj_improvement_pct:.2f}% improvement over baseline")
        print(f"  Improvement is small but positive")
        print(f"\n{Colors.YELLOW}Recommendation: Analyze top trials, then proceed cautiously to Step 3{Colors.END}")

    elif obj_improvement > -0.01:
        print(f"{Colors.YELLOW}~ NO SIGNIFICANT CHANGE{Colors.END}")
        print(f"  Test study similar to baseline ({obj_improvement_pct:.2f}% change)")
        print(f"\n{Colors.YELLOW}Recommendation: Review implementation, test fixes in isolation{Colors.END}")

    else:
        print(f"{Colors.RED}✗ PERFORMANCE REGRESSION{Colors.END}")
        print(f"  Test study shows {obj_improvement_pct:.2f}% decline from baseline")
        print(f"\n{Colors.RED}Recommendation: Investigate what went wrong, revert fixes{Colors.END}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze fundamental fixes test results")
    parser.add_argument('--study', type=str, default='cappuccino_fundamentals_test_20251215', help='Test study name')
    parser.add_argument('--baseline', type=str, default='cappuccino_3workers_20251102_2325', help='Baseline study name')
    parser.add_argument('--db', type=str, default='databases/optuna_cappuccino.db', help='Database path')
    args = parser.parse_args()

    print(f"\n{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.HEADER}FUNDAMENTAL FIXES - RESULTS ANALYSIS{Colors.END}")
    print(f"{Colors.HEADER}{'='*80}{Colors.END}\n")

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"{Colors.RED}ERROR: Database not found: {db_path}{Colors.END}")
        return

    # Load test study
    print(f"{Colors.BLUE}Loading test study: {args.study}...{Colors.END}")
    test_trials = get_study_trials(str(db_path), args.study)

    if not test_trials:
        print(f"{Colors.RED}ERROR: No trials found for study '{args.study}'{Colors.END}")
        print(f"{Colors.YELLOW}Has the test completed? Check with: sqlite3 {db_path} \"SELECT COUNT(*) FROM trials WHERE study_id = (SELECT study_id FROM studies WHERE study_name = '{args.study}');\"{Colors.END}")
        return

    print(f"{Colors.GREEN}Found {len(test_trials)} completed trials{Colors.END}\n")

    # Load baseline study
    print(f"{Colors.BLUE}Loading baseline study: {args.baseline}...{Colors.END}")
    baseline_trials = get_study_trials(str(db_path), args.baseline)

    if not baseline_trials:
        print(f"{Colors.YELLOW}WARNING: No trials found for baseline study '{args.baseline}'{Colors.END}")
        print(f"{Colors.YELLOW}Continuing with test-only analysis...{Colors.END}\n")
        baseline_trials = None
    else:
        print(f"{Colors.GREEN}Found {len(baseline_trials)} completed trials{Colors.END}\n")

    # Analyze test study
    test_stats = analyze_study(test_trials, args.study)

    print(f"{Colors.CYAN}Test Study Summary:{Colors.END}")
    print(f"  Trials: {test_stats['n_trials']}")
    print(f"  Mean Objective: {test_stats['mean_objective']:.6f} ± {test_stats['std_objective']:.6f}")
    print(f"  Best Objective: {test_stats['max_objective']:.6f}")
    print(f"  Worst Objective: {test_stats['min_objective']:.6f}")
    if 'mean_sharpe_bot' in test_stats:
        print(f"  Mean Sharpe (Bot): {test_stats['mean_sharpe_bot']:.6f}")
    print()

    # Compare if baseline available
    if baseline_trials:
        baseline_stats = analyze_study(baseline_trials, args.baseline)

        print(f"{Colors.CYAN}Baseline Study Summary:{Colors.END}")
        print(f"  Trials: {baseline_stats['n_trials']}")
        print(f"  Mean Objective: {baseline_stats['mean_objective']:.6f} ± {baseline_stats['std_objective']:.6f}")
        print(f"  Best Objective: {baseline_stats['max_objective']:.6f}")
        if 'mean_sharpe_bot' in baseline_stats:
            print(f"  Mean Sharpe (Bot): {baseline_stats['mean_sharpe_bot']:.6f}")
        print()

        # Detailed comparison
        compare_studies(test_stats, baseline_stats)

    print(f"{Colors.HEADER}{'='*80}{Colors.END}\n")


if __name__ == '__main__':
    main()
