#!/usr/bin/env python3
"""Export an Optuna trial to a format usable by paper_trader_alpaca_polling.py"""
import argparse
import pickle
import optuna
from pathlib import Path

def export_trial(study_name: str, trial_number: int, output_dir: Path):
    """Load trial from database and export to paper trader format"""

    # Load study
    storage = f"sqlite:///databases/optuna_cappuccino.db"
    study = optuna.load_study(study_name=study_name, storage=storage)

    # Find trial
    trials = [t for t in study.trials if t.number == trial_number]
    if not trials:
        raise ValueError(f"Trial #{trial_number} not found in study {study_name}")

    trial = trials[0]

    # Set name_folder if not already set
    if "name_folder" not in trial.user_attrs:
        # Derive from output_dir
        folder_name = output_dir.name if output_dir.name != "." else "trial_temp"
        trial.set_user_attr("name_folder", f"cwd_tests/{folder_name}")
        print(f"✓ Set name_folder to: cwd_tests/{folder_name}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save trial as best_trial
    with open(output_dir / "best_trial", "wb") as f:
        pickle.dump(trial, f)

    print(f"✓ Exported trial #{trial_number} to {output_dir}/best_trial")
    print(f"  Sharpe: {trial.value:.6f}")
    print(f"  State: {trial.state}")

    # Print hyperparameters
    print("\nHyperparameters:")
    for key, value in sorted(trial.params.items()):
        print(f"  {key}: {value}")

    # Check if stored_agent exists
    name_folder = trial.user_attrs.get("name_folder")
    if name_folder:
        stored_agent_path = Path("train_results") / name_folder / "stored_agent"
        actor_path = Path("train_results") / name_folder / "actor.pth"

        if stored_agent_path.exists():
            print(f"\n✓ Model weights found at: {stored_agent_path}")
        elif actor_path.exists():
            # Create stored_agent directory and symlink
            stored_agent_path.mkdir(parents=True, exist_ok=True)
            (stored_agent_path / "actor.pth").symlink_to("../actor.pth")
            print(f"\n✓ Created stored_agent directory and symlinked actor.pth")
        else:
            print(f"\n⚠️  Warning: Model weights not found at {stored_agent_path} or {actor_path}")
    else:
        print(f"\n⚠️  Warning: Trial doesn't have name_folder attribute")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Optuna trial for paper trading")
    parser.add_argument("--study", required=True, help="Study name")
    parser.add_argument("--trial", type=int, required=True, help="Trial number")
    parser.add_argument("--output-dir", required=True, help="Output directory")

    args = parser.parse_args()

    export_trial(args.study, args.trial, Path(args.output_dir))
