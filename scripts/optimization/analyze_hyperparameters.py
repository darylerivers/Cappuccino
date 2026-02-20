#!/usr/bin/env python3
"""
Analyze hyperparameters of top vs bottom performing trials
"""
import sqlite3
import json
import sys
from collections import defaultdict
from pathlib import Path
import statistics

# Ensure project root is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.study_config import get_current_study

DB_PATH = "databases/optuna_cappuccino.db"
STUDY_NAME = get_current_study()

def get_trial_params(conn, trial_numbers):
    """Get all hyperparameters for given trial numbers"""
    placeholders = ','.join('?' * len(trial_numbers))
    query = f"""
    SELECT t.number, tp.param_name, tp.param_value
    FROM trials t
    JOIN trial_params tp ON t.trial_id = tp.trial_id
    WHERE t.study_id = (SELECT study_id FROM studies WHERE study_name = ?)
    AND t.number IN ({placeholders})
    ORDER BY t.number, tp.param_name
    """

    cursor = conn.execute(query, [STUDY_NAME] + trial_numbers)

    # Organize by trial number
    trials = defaultdict(dict)
    for row in cursor:
        trial_num, param_name, param_value = row
        trials[trial_num][param_name] = param_value

    return trials

def analyze_param_distribution(param_name, top_values, bottom_values):
    """Analyze a single parameter's distribution"""
    if not top_values or not bottom_values:
        return None

    top_mean = statistics.mean(top_values)
    top_median = statistics.median(top_values)
    bottom_mean = statistics.mean(bottom_values)
    bottom_median = statistics.median(bottom_values)

    # Calculate difference
    mean_diff_pct = ((top_mean - bottom_mean) / bottom_mean * 100) if bottom_mean != 0 else 0

    return {
        'param': param_name,
        'top_mean': top_mean,
        'top_median': top_median,
        'top_min': min(top_values),
        'top_max': max(top_values),
        'bottom_mean': bottom_mean,
        'bottom_median': bottom_median,
        'bottom_min': min(bottom_values),
        'bottom_max': max(bottom_values),
        'mean_diff_pct': mean_diff_pct
    }

def main():
    conn = sqlite3.connect(DB_PATH)

    # Get top 20 and bottom 20 trial numbers
    cursor = conn.execute("""
        SELECT t.number, tv.value as sharpe
        FROM trials t
        JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE t.study_id = (SELECT study_id FROM studies WHERE study_name = ?)
        AND t.state = 'COMPLETE'
        ORDER BY tv.value DESC
        LIMIT 20
    """, [STUDY_NAME])
    top_trials = [(row[0], row[1]) for row in cursor]
    top_trial_numbers = [t[0] for t in top_trials]

    cursor = conn.execute("""
        SELECT t.number, tv.value as sharpe
        FROM trials t
        JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE t.study_id = (SELECT study_id FROM studies WHERE study_name = ?)
        AND t.state = 'COMPLETE'
        ORDER BY tv.value ASC
        LIMIT 20
    """, [STUDY_NAME])
    bottom_trials = [(row[0], row[1]) for row in cursor]
    bottom_trial_numbers = [t[0] for t in bottom_trials]

    # Get parameters
    top_params = get_trial_params(conn, top_trial_numbers)
    bottom_params = get_trial_params(conn, bottom_trial_numbers)

    # Get all parameter names
    all_params = set()
    for trial_params in top_params.values():
        all_params.update(trial_params.keys())

    # Analyze each parameter
    results = []
    for param_name in sorted(all_params):
        top_values = [trial_params.get(param_name) for trial_params in top_params.values()
                      if trial_params.get(param_name) is not None]
        bottom_values = [trial_params.get(param_name) for trial_params in bottom_params.values()
                         if trial_params.get(param_name) is not None]

        analysis = analyze_param_distribution(param_name, top_values, bottom_values)
        if analysis:
            results.append(analysis)

    # Sort by absolute mean difference
    results.sort(key=lambda x: abs(x['mean_diff_pct']), reverse=True)

    # Print report
    print("=" * 100)
    print("HYPERPARAMETER ANALYSIS: TOP 20 vs BOTTOM 20 TRIALS")
    print("=" * 100)
    print(f"\nStudy: {STUDY_NAME}")
    print(f"Top 20 Sharpe range: {top_trials[0][1]:.6f} to {top_trials[-1][1]:.6f}")
    print(f"Bottom 20 Sharpe range: {bottom_trials[0][1]:.6f} to {bottom_trials[-1][1]:.6f}")
    print(f"\n{'=' * 100}")
    print("\nMOST SIGNIFICANT DIFFERENCES (sorted by impact)")
    print("=" * 100)

    # Header
    print(f"\n{'Parameter':<25} {'Top Mean':<12} {'Top Med':<12} {'Bot Mean':<12} {'Bot Med':<12} {'Diff %':<10}")
    print("-" * 100)

    # Top 15 most different
    for result in results[:15]:
        diff_indicator = "🟢" if result['mean_diff_pct'] > 0 else "🔴"
        print(f"{result['param']:<25} {result['top_mean']:<12.6f} {result['top_median']:<12.6f} "
              f"{result['bottom_mean']:<12.6f} {result['bottom_median']:<12.6f} "
              f"{diff_indicator} {result['mean_diff_pct']:>8.1f}%")

    print("\n" + "=" * 100)
    print("ALL PARAMETERS (sorted by impact)")
    print("=" * 100)
    print(f"\n{'Parameter':<25} {'Top Range':<30} {'Bottom Range':<30} {'Diff %':<10}")
    print("-" * 100)

    for result in results:
        top_range = f"{result['top_min']:.4f} - {result['top_max']:.4f}"
        bottom_range = f"{result['bottom_min']:.4f} - {result['bottom_max']:.4f}"
        print(f"{result['param']:<25} {top_range:<30} {bottom_range:<30} {result['mean_diff_pct']:>8.1f}%")

    # Save detailed results
    output = {
        'study_name': STUDY_NAME,
        'top_trials': top_trials,
        'bottom_trials': bottom_trials,
        'parameter_analysis': results
    }

    with open('analysis_hyperparameters.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n\nDetailed results saved to: analysis_hyperparameters.json")

    conn.close()

if __name__ == "__main__":
    main()
