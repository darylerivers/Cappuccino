#!/usr/bin/env python3
"""Analyze Optuna training results from database."""

import argparse
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np


def analyze_study(db_path: str, study_name: str = None):
    """Analyze Optuna study results."""

    conn = sqlite3.connect(db_path)

    # List available studies if no study name provided
    if study_name is None:
        studies = pd.read_sql_query("SELECT study_name FROM studies", conn)
        print(f"\nAvailable studies in {Path(db_path).name}:")
        for name in studies['study_name']:
            print(f"  - {name}")
        conn.close()
        return

    # Get study ID
    study_query = "SELECT study_id FROM studies WHERE study_name = ?"
    study_df = pd.read_sql_query(study_query, conn, params=(study_name,))

    if study_df.empty:
        print(f"Study '{study_name}' not found in database")
        conn.close()
        return

    study_id = study_df['study_id'].iloc[0]

    # Get trial data
    trials_query = """
    SELECT
        trial_id,
        number,
        value,
        state,
        datetime_start,
        datetime_complete
    FROM trials
    WHERE study_id = ?
    ORDER BY number
    """
    trials_df = pd.read_sql_query(trials_query, conn, params=(study_id,))

    # Get trial parameters
    params_query = """
    SELECT
        t.number,
        tp.param_name,
        tp.param_value
    FROM trial_params tp
    JOIN trials t ON tp.trial_id = t.trial_id
    WHERE t.study_id = ?
    """
    params_df = pd.read_sql_query(params_query, conn, params=(study_id,))

    # Get trial user attributes (contains model_name, etc.)
    attrs_query = """
    SELECT
        t.number,
        tua.key,
        tua.value_json
    FROM trial_user_attributes tua
    JOIN trials t ON tua.trial_id = t.trial_id
    WHERE t.study_id = ?
    """
    attrs_df = pd.read_sql_query(attrs_query, conn, params=(study_id,))

    conn.close()

    # Filter completed trials
    completed = trials_df[trials_df['state'] == 'COMPLETE'].copy()

    if completed.empty:
        print(f"\nNo completed trials found for study '{study_name}'")
        return

    # Calculate duration
    completed['datetime_start'] = pd.to_datetime(completed['datetime_start'])
    completed['datetime_complete'] = pd.to_datetime(completed['datetime_complete'])
    completed['duration_minutes'] = (completed['datetime_complete'] - completed['datetime_start']).dt.total_seconds() / 60

    # Extract model names from attributes
    model_map = {}
    for _, row in attrs_df.iterrows():
        if row['key'] == 'model_name':
            model_map[row['number']] = row['value_json'].strip('"')

    completed['model'] = completed['number'].map(model_map)

    # Print overall statistics
    print(f"\n{'='*80}")
    print(f"ANALYSIS: {study_name}")
    print(f"Database: {Path(db_path).name}")
    print(f"{'='*80}")

    print(f"\nTotal trials: {len(trials_df)}")
    print(f"Completed: {len(completed)}")
    print(f"Failed: {len(trials_df[trials_df['state'] == 'FAIL'])}")
    print(f"Pruned: {len(trials_df[trials_df['state'] == 'PRUNED'])}")
    print(f"Running: {len(trials_df[trials_df['state'] == 'RUNNING'])}")

    # Performance statistics
    print(f"\n{'='*80}")
    print("PERFORMANCE STATISTICS (Objective Value)")
    print(f"{'='*80}")

    values = completed['value'].dropna()
    print(f"\nCount:      {len(values)}")
    print(f"Mean:       {values.mean():.6f}")
    print(f"Median:     {values.median():.6f}")
    print(f"Std Dev:    {values.std():.6f}")
    print(f"Min:        {values.min():.6f}")
    print(f"Max:        {values.max():.6f}")
    print(f"Q1 (25%):   {values.quantile(0.25):.6f}")
    print(f"Q3 (75%):   {values.quantile(0.75):.6f}")

    # Best trials
    print(f"\n{'='*80}")
    print("TOP 10 TRIALS")
    print(f"{'='*80}")

    top_trials = completed.nlargest(10, 'value')[['number', 'value', 'model', 'duration_minutes']]
    print(f"\n{'Trial':<8} {'Value':>12} {'Model':<10} {'Duration (min)':>15}")
    print("-" * 80)
    for _, row in top_trials.iterrows():
        model = row['model'] if pd.notna(row['model']) else 'unknown'
        duration = row['duration_minutes'] if pd.notna(row['duration_minutes']) else 0
        print(f"{int(row['number']):<8} {row['value']:>12.6f} {model:<10} {duration:>15.1f}")

    # Worst trials
    print(f"\n{'='*80}")
    print("BOTTOM 10 TRIALS")
    print(f"{'='*80}")

    bottom_trials = completed.nsmallest(10, 'value')[['number', 'value', 'model', 'duration_minutes']]
    print(f"\n{'Trial':<8} {'Value':>12} {'Model':<10} {'Duration (min)':>15}")
    print("-" * 80)
    for _, row in bottom_trials.iterrows():
        model = row['model'] if pd.notna(row['model']) else 'unknown'
        duration = row['duration_minutes'] if pd.notna(row['duration_minutes']) else 0
        print(f"{int(row['number']):<8} {row['value']:>12.6f} {model:<10} {duration:>15.1f}")

    # Model comparison
    if completed['model'].notna().any():
        print(f"\n{'='*80}")
        print("PERFORMANCE BY MODEL")
        print(f"{'='*80}")

        model_stats = completed.groupby('model')['value'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
        model_stats = model_stats.sort_values('mean', ascending=False)

        print(f"\n{'Model':<10} {'Count':>7} {'Mean':>12} {'Median':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
        print("-" * 80)
        for model, row in model_stats.iterrows():
            print(f"{model:<10} {int(row['count']):>7} {row['mean']:>12.6f} {row['median']:>12.6f} "
                  f"{row['std']:>12.6f} {row['min']:>12.6f} {row['max']:>12.6f}")

    # Duration statistics
    print(f"\n{'='*80}")
    print("TRAINING DURATION STATISTICS")
    print(f"{'='*80}")

    durations = completed['duration_minutes'].dropna()
    if not durations.empty:
        print(f"\nMean duration:   {durations.mean():.1f} minutes")
        print(f"Median duration: {durations.median():.1f} minutes")
        print(f"Total time:      {durations.sum():.1f} minutes ({durations.sum()/60:.1f} hours)")
        print(f"Min duration:    {durations.min():.1f} minutes")
        print(f"Max duration:    {durations.max():.1f} minutes")

    # Parameter analysis for top trials
    print(f"\n{'='*80}")
    print("TOP TRIAL HYPERPARAMETERS")
    print(f"{'='*80}")

    best_trial = completed.nlargest(1, 'value').iloc[0]
    best_params = params_df[params_df['number'] == best_trial['number']]

    print(f"\nBest Trial #{int(best_trial['number'])} - Value: {best_trial['value']:.6f}")
    print("-" * 80)
    for _, row in best_params.iterrows():
        print(f"  {row['param_name']:<30} {row['param_value']}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Optuna training results")
    parser.add_argument("--db", type=str, required=True, help="Path to Optuna database")
    parser.add_argument("--study", type=str, help="Study name (omit to list available studies)")

    args = parser.parse_args()

    analyze_study(args.db, args.study)
