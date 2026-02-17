#!/usr/bin/env python3
"""Create simple ensemble from existing trial models."""

import json
import sqlite3
from pathlib import Path


def find_best_trials_with_models(
    db_path="databases/optuna_cappuccino.db",
    study_name="cappuccino_1year_20251121",
    top_n=10
):
    """Find best trials that have saved models."""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all completed trials with values, ordered by value
    query = """
    SELECT t.number, tv.value
    FROM trials t
    JOIN trial_values tv ON t.trial_id = tv.trial_id
    JOIN studies s ON t.study_id = s.study_id
    WHERE s.study_name = ?
        AND t.state = 'COMPLETE'
    ORDER BY tv.value DESC
    """

    cursor.execute(query, (study_name,))
    all_trials = cursor.fetchall()
    conn.close()

    # Filter to trials that actually have model files
    trials_with_models = []
    base_path = Path("train_results/cwd_tests")

    for trial_num, value in all_trials:
        trial_folders = list(base_path.glob(f"trial_{trial_num}_*"))
        if trial_folders:
            # Check for actor.pth directly in trial folder
            actor_path = trial_folders[0] / "actor.pth"

            # Also check in stored_agent subfolder
            if not actor_path.exists():
                actor_path = trial_folders[0] / "stored_agent" / "actor.pth"

            if actor_path.exists():
                trials_with_models.append({
                    'trial_number': trial_num,
                    'value': value,
                    'path': str(trial_folders[0]),
                    'actor_path': str(actor_path),
                })

                if len(trials_with_models) >= top_n:
                    break

    return trials_with_models


def create_ensemble_manifest(trials, output_dir="train_results/ensemble"):
    """Create ensemble manifest file."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ensemble_info = {
        'model_count': len(trials),
        'trial_numbers': [t['trial_number'] for t in trials],
        'trial_values': [t['value'] for t in trials],
        'trial_paths': [t['path'] for t in trials],
        'actor_paths': [t['actor_path'] for t in trials],
        'mean_value': sum(t['value'] for t in trials) / len(trials) if trials else 0,
        'best_value': max(t['value'] for t in trials) if trials else 0,
        'worst_value': min(t['value'] for t in trials) if trials else 0,
    }

    # Save as JSON
    with (output_path / "ensemble_manifest.json").open("w") as f:
        json.dump(ensemble_info, f, indent=2)

    return ensemble_info


def main():
    print("=" * 80)
    print("SIMPLE ENSEMBLE CREATION")
    print("=" * 80)
    print()

    print("🔍 Finding best trials with saved models...")
    trials = find_best_trials_with_models(top_n=10)

    if not trials:
        print("❌ No trials with saved models found!")
        return

    print(f"✓ Found {len(trials)} trials with saved models")
    print()
    print("Top trials:")
    for i, trial in enumerate(trials[:5], 1):
        print(f"  {i}. Trial #{trial['trial_number']}: {trial['value']:.6f}")
    print()

    print("📝 Creating ensemble manifest...")
    info = create_ensemble_manifest(trials)

    print(f"✓ Ensemble created!")
    print()
    print("Ensemble Statistics:")
    print(f"  Models: {info['model_count']}")
    print(f"  Mean Value: {info['mean_value']:.6f}")
    print(f"  Best Value: {info['best_value']:.6f}")
    print(f"  Worst Value: {info['worst_value']:.6f}")
    print()
    print("=" * 80)
    print("ENSEMBLE READY!")
    print("=" * 80)
    print()
    print("Manifest saved to: train_results/ensemble/ensemble_manifest.json")


if __name__ == "__main__":
    main()
