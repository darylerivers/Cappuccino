#!/usr/bin/env python3
"""Comprehensive statistical analysis of training results."""

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.study_config import get_current_study


def analyze_training_results(db_path: str = "databases/optuna_cappuccino.db"):
    """Analyze all training trials and provide comprehensive statistics."""

    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)

    # Get study info
    studies_df = pd.read_sql_query("SELECT * FROM studies", conn)
    print("=" * 100)
    print("CAPPUCCINO TRAINING ANALYSIS")
    print("=" * 100)
    print()

    if studies_df.empty:
        print("No studies found in database.")
        return

    study_name = get_current_study()
    print(f"Study: {study_name}")
    print()

    # Get all trials with their values
    query = """
    SELECT
        t.trial_id,
        t.number,
        t.state,
        t.datetime_start,
        t.datetime_complete,
        tv.value
    FROM trials t
    LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
    JOIN studies s ON t.study_id = s.study_id
    WHERE s.study_name = ?
    ORDER BY t.number
    """
    trials_df = pd.read_sql_query(query, conn, params=(study_name,))

    # Basic statistics
    print("━" * 100)
    print("OVERALL STATISTICS")
    print("━" * 100)

    total_trials = len(trials_df)
    completed = len(trials_df[trials_df['state'] == 'COMPLETE'])
    running = len(trials_df[trials_df['state'] == 'RUNNING'])
    failed = len(trials_df[trials_df['state'] == 'FAIL'])

    print(f"Total Trials:       {total_trials:,}")
    print(f"  Completed:        {completed:,} ({completed/total_trials*100:.1f}%)")
    print(f"  Running:          {running:,} ({running/total_trials*100:.1f}%)")
    print(f"  Failed:           {failed:,} ({failed/total_trials*100:.1f}%)")
    print()

    # Completed trials statistics
    completed_trials = trials_df[trials_df['state'] == 'COMPLETE'].copy()

    if completed_trials.empty:
        print("No completed trials to analyze.")
        return

    values = completed_trials['value'].dropna().values

    print("━" * 100)
    print("PERFORMANCE METRICS (Completed Trials)")
    print("━" * 100)

    print(f"Best Value:         {values.max():.6f}")
    print(f"Worst Value:        {values.min():.6f}")
    print(f"Mean:               {values.mean():.6f}")
    print(f"Median:             {np.median(values):.6f}")
    print(f"Std Dev:            {values.std():.6f}")
    print()

    # Percentiles
    percentiles = [99, 95, 90, 75, 50, 25, 10, 5, 1]
    print("Percentile Distribution:")
    for p in percentiles:
        val = np.percentile(values, p)
        print(f"  {p:>3}th percentile:  {val:.6f}")
    print()

    # Top performers
    print("━" * 100)
    print("TOP 10 TRIALS")
    print("━" * 100)
    top_10 = completed_trials.nlargest(10, 'value')[['number', 'value', 'datetime_complete']]
    print(f"{'Trial #':<10} {'Value':<15} {'Completed':<25}")
    print("-" * 100)
    for _, row in top_10.iterrows():
        trial_num = int(row['number'])
        value = row['value']
        completed_time = row['datetime_complete'] if pd.notna(row['datetime_complete']) else 'N/A'
        print(f"{trial_num:<10} {value:<15.6f} {completed_time:<25}")
    print()

    # Top 10% threshold
    top_10_pct_threshold = np.percentile(values, 90)
    top_10_pct_count = np.sum(values >= top_10_pct_threshold)
    print(f"Top 10% Threshold:  {top_10_pct_threshold:.6f}")
    print(f"Trials in Top 10%:  {top_10_pct_count} trials (≥ {top_10_pct_threshold:.6f})")
    print()

    # Time analysis
    print("━" * 100)
    print("TIME ANALYSIS")
    print("━" * 100)

    # Parse timestamps
    completed_trials['datetime_start'] = pd.to_datetime(completed_trials['datetime_start'])
    completed_trials['datetime_complete'] = pd.to_datetime(completed_trials['datetime_complete'])
    completed_trials['duration'] = (completed_trials['datetime_complete'] - completed_trials['datetime_start']).dt.total_seconds()

    durations = completed_trials['duration'].dropna().values
    if len(durations) > 0:
        print(f"Average Trial Duration:  {durations.mean()/60:.1f} minutes")
        print(f"Median Trial Duration:   {np.median(durations)/60:.1f} minutes")
        print(f"Min Trial Duration:      {durations.min()/60:.1f} minutes")
        print(f"Max Trial Duration:      {durations.max()/60:.1f} minutes")
        print()

    # First and last trial times
    first_trial = completed_trials['datetime_start'].min()
    last_trial = completed_trials['datetime_complete'].max()

    if pd.notna(first_trial) and pd.notna(last_trial):
        total_time = last_trial - first_trial
        print(f"First Trial Started:     {first_trial}")
        print(f"Last Trial Completed:    {last_trial}")
        print(f"Total Training Time:     {total_time.days} days, {total_time.seconds//3600} hours")
        print()

    # Recent performance (last 24 hours, last week)
    print("━" * 100)
    print("RECENT PERFORMANCE")
    print("━" * 100)

    now = datetime.now()

    # Last 24 hours
    last_24h = completed_trials[completed_trials['datetime_complete'] > (now - timedelta(hours=24))]
    if not last_24h.empty:
        print(f"Last 24 Hours:")
        print(f"  Completed:         {len(last_24h)} trials")
        print(f"  Best Value:        {last_24h['value'].max():.6f}")
        print(f"  Mean Value:        {last_24h['value'].mean():.6f}")
        print(f"  Trials/Hour:       {len(last_24h)/24:.1f}")
        print()

    # Last 7 days
    last_7d = completed_trials[completed_trials['datetime_complete'] > (now - timedelta(days=7))]
    if not last_7d.empty:
        print(f"Last 7 Days:")
        print(f"  Completed:         {len(last_7d)} trials")
        print(f"  Best Value:        {last_7d['value'].max():.6f}")
        print(f"  Mean Value:        {last_7d['value'].mean():.6f}")
        print(f"  Trials/Day:        {len(last_7d)/7:.1f}")
        print()

    # Value distribution analysis
    print("━" * 100)
    print("VALUE DISTRIBUTION")
    print("━" * 100)

    # Binning
    bins = [-float('inf'), 0.02, 0.04, 0.05, 0.06, 0.065, 0.07, 0.075, float('inf')]
    bin_labels = ['< 0.02', '0.02-0.04', '0.04-0.05', '0.05-0.06', '0.06-0.065', '0.065-0.07', '0.07-0.075', '≥ 0.075']

    value_dist = pd.cut(values, bins=bins, labels=bin_labels)
    dist_counts = value_dist.value_counts().sort_index()

    print(f"{'Range':<15} {'Count':<10} {'Percentage':<12} {'Bar':<50}")
    print("-" * 100)
    max_count = dist_counts.max()
    for label, count in dist_counts.items():
        pct = count / len(values) * 100
        bar_width = int((count / max_count) * 40)
        bar = '█' * bar_width
        print(f"{label:<15} {count:<10} {pct:>6.2f}%      {bar}")
    print()

    # Trend analysis (improving over time?)
    print("━" * 100)
    print("TREND ANALYSIS")
    print("━" * 100)

    # Split into chunks
    chunk_size = max(len(completed_trials) // 10, 1)
    chunks = []
    for i in range(0, len(completed_trials), chunk_size):
        chunk = completed_trials.iloc[i:i+chunk_size]
        if not chunk.empty and chunk['value'].notna().any():
            chunks.append({
                'trials': f"{i+1}-{min(i+chunk_size, len(completed_trials))}",
                'mean': chunk['value'].mean(),
                'max': chunk['value'].max(),
                'count': len(chunk)
            })

    if chunks:
        print(f"{'Trials':<15} {'Mean':<15} {'Max':<15} {'Count':<10}")
        print("-" * 100)
        for chunk in chunks:
            print(f"{chunk['trials']:<15} {chunk['mean']:<15.6f} {chunk['max']:<15.6f} {chunk['count']:<10}")
        print()

        # Check if improving
        if len(chunks) >= 2:
            early_mean = np.mean([c['mean'] for c in chunks[:len(chunks)//3]])
            late_mean = np.mean([c['mean'] for c in chunks[2*len(chunks)//3:]])
            improvement = ((late_mean - early_mean) / early_mean * 100)

            if improvement > 0:
                print(f"✓ Training is IMPROVING: {improvement:+.2f}% improvement in mean performance")
            else:
                print(f"✗ Training may be plateauing: {improvement:+.2f}% change in mean performance")
            print()

    # Get hyperparameters of best trials
    print("━" * 100)
    print("BEST TRIAL HYPERPARAMETERS")
    print("━" * 100)

    best_trial_id = completed_trials.nlargest(1, 'value')['trial_id'].iloc[0]

    params_query = """
    SELECT param_name, param_value
    FROM trial_params
    WHERE trial_id = ?
    """
    best_params = pd.read_sql_query(params_query, conn, params=(best_trial_id,))

    if not best_params.empty:
        print(f"Best Trial #{int(completed_trials[completed_trials['trial_id'] == best_trial_id]['number'].iloc[0])} "
              f"(Value: {completed_trials[completed_trials['trial_id'] == best_trial_id]['value'].iloc[0]:.6f})")
        print()
        for _, row in best_params.iterrows():
            print(f"  {row['param_name']:<30} = {row['param_value']}")
        print()

    conn.close()

    print("=" * 100)
    print(f"Analysis complete. Total trials analyzed: {total_trials:,}")
    print("=" * 100)


if __name__ == "__main__":
    analyze_training_results()
