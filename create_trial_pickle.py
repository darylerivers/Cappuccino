#!/usr/bin/env python3
"""
Create best_trial pickle file from database for paper trading.

This script reconstructs the trial object needed by paper_trader_alpaca_polling.py
from the Optuna database.
"""

import argparse
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, Any

import optuna


def create_trial_pickle(trial_id: int, db_path: str, output_dir: Path) -> None:
    """Create best_trial pickle file for a specific trial."""

    # Load study
    storage = optuna.storages.RDBStorage(f"sqlite:///{db_path}")

    # Find the study that contains this trial
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT s.study_name
        FROM studies s
        JOIN trials t ON s.study_id = t.study_id
        WHERE t.trial_id = ?
    """, (trial_id,))

    result = cursor.fetchone()
    if not result:
        raise ValueError(f"Trial {trial_id} not found in database")

    study_name = result[0]
    conn.close()

    print(f"Loading study: {study_name}")
    study = optuna.load_study(study_name=study_name, storage=storage)

    # Find the trial
    trial = None
    for t in study.trials:
        if t.number == trial_id or t._trial_id == trial_id:
            trial = t
            break

    if trial is None:
        raise ValueError(f"Trial {trial_id} not found in study {study_name}")

    print(f"Found trial #{trial.number}")
    print(f"  Value: {trial.value}")
    print(f"  Params: {len(trial.params)} parameters")
    print(f"  State: {trial.state}")

    # Add necessary user attributes if missing
    if "name_folder" not in trial.user_attrs:
        # Infer from output_dir
        folder_name = output_dir.name.replace("_1h", "").replace("trial_", "")
        trial.set_user_attr("name_folder", f"cwd_tests/{output_dir.name}")
        print(f"  Added name_folder: cwd_tests/{output_dir.name}")

    if "model_name" not in trial.user_attrs:
        trial.set_user_attr("model_name", "ppo")
        print(f"  Added model_name: ppo")

    # Save to pickle
    output_path = output_dir / "best_trial"
    output_dir.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        pickle.dump(trial, f)

    print(f"✓ Created: {output_path}")

    # Verify it can be loaded
    with output_path.open("rb") as f:
        loaded_trial = pickle.load(f)

    print(f"✓ Verified: Trial #{loaded_trial.number} with value {loaded_trial.value}")


def main():
    parser = argparse.ArgumentParser(
        description="Create best_trial pickle file from Optuna database"
    )
    parser.add_argument(
        "--trial-id",
        type=int,
        required=True,
        help="Trial ID to export"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="databases/optuna_cappuccino.db",
        help="Path to Optuna database"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory (e.g., train_results/cwd_tests/trial_3358_1h)"
    )

    args = parser.parse_args()

    create_trial_pickle(
        trial_id=args.trial_id,
        db_path=args.db,
        output_dir=Path(args.output_dir)
    )


if __name__ == "__main__":
    main()
