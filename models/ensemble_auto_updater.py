#!/usr/bin/env python3
"""
Ensemble Auto-Updater

Automatically updates the adaptive ensemble with new best trials from training.
Runs in daemon mode, periodically checking for new top performers.

Features:
- Syncs top N trials from database to ensemble
- Adds new models that outperform current ensemble
- Removes underperforming models
- Hot-reloads the paper trader without full restart
"""

import argparse
import json
import pickle
import shutil
import signal
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set


class EnsembleAutoUpdater:
    def __init__(
        self,
        study_name: str,
        ensemble_dir: str = "train_results/ensemble_adaptive",
        db_path: str = "databases/optuna_cappuccino.db",
        top_n: int = 10,
        check_interval: int = 300,  # 5 minutes
        min_improvement: float = 0.001,  # Minimum value improvement to swap
    ):
        self.study_name = study_name
        self.ensemble_dir = Path(ensemble_dir)
        self.db_path = db_path
        self.top_n = top_n
        self.check_interval = check_interval
        self.min_improvement = min_improvement
        self.running = True

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        self.log(f"Received signal {signum}, shutting down...")
        self.running = False

    def log(self, msg: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {msg}")

    def get_current_ensemble(self) -> Dict[int, float]:
        """Get current ensemble trial numbers and their values."""
        manifest_path = self.ensemble_dir / "ensemble_manifest.json"
        if not manifest_path.exists():
            return {}

        with open(manifest_path) as f:
            manifest = json.load(f)

        trial_numbers = manifest.get("trial_numbers", [])

        # Get values from individual model directories
        trials = {}
        for i, model_dir in enumerate(sorted(self.ensemble_dir.glob("model_*"))):
            best_trial = model_dir / "best_trial"
            if best_trial.exists():
                try:
                    with open(best_trial, "rb") as f:
                        trial = pickle.load(f)
                    trials[trial.number] = trial.values[0] if trial.values else 0.0
                except Exception:
                    pass

        return trials

    def get_top_trials_from_db(self) -> List[Dict]:
        """Get top N trials from the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            query = """
            SELECT t.trial_id, t.number, tv.value
            FROM trials t
            JOIN studies s ON t.study_id = s.study_id
            JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE s.study_name = ?
            AND t.state = 'COMPLETE'
            ORDER BY tv.value DESC
            LIMIT ?
            """
            cursor.execute(query, (self.study_name, self.top_n * 50))  # Get many extras to filter for trials with model files
            results = cursor.fetchall()
            conn.close()

            return [
                {"trial_id": r[0], "number": r[1], "value": r[2]}
                for r in results
            ]
        except Exception as e:
            self.log(f"Error querying database: {e}")
            return []

    def find_model_source(self, trial_number: int) -> Optional[Path]:
        """Find the source directory for a trial's model files."""
        # Check various locations
        patterns = [
            f"train_results/trial_{trial_number}_1h",
            f"train_results/cwd_tests/trial_{trial_number}_1h",
        ]

        for pattern in patterns:
            path = Path(pattern)
            if path.exists():
                # Check for stored_agent subdirectory first
                stored_agent = path / "stored_agent"
                if stored_agent.exists() and (stored_agent / "actor.pth").exists():
                    return stored_agent
                # Then check directly in the trial directory
                if (path / "actor.pth").exists():
                    return path

        return None

    def get_frozen_trial(self, trial_number: int) -> Optional[object]:
        """Get FrozenTrial object from Optuna database."""
        try:
            import optuna
            study = optuna.load_study(
                study_name=self.study_name,
                storage=f"sqlite:///{self.db_path}",
            )
            for trial in study.trials:
                if trial.number == trial_number:
                    return trial
        except Exception as e:
            self.log(f"Error getting FrozenTrial: {e}")
        return None

    def copy_model_to_ensemble(self, trial_number: int, slot: int) -> bool:
        """Copy a trial's model files to the ensemble directory."""
        source = self.find_model_source(trial_number)
        if source is None:
            self.log(f"Cannot find model files for trial #{trial_number}")
            return False

        dest = self.ensemble_dir / f"model_{slot}"
        dest.mkdir(parents=True, exist_ok=True)

        try:
            # Copy actor weights
            shutil.copy(source / "actor.pth", dest / "actor.pth")

            # Copy best_trial pickle if it exists, otherwise create from database
            if (source / "best_trial").exists():
                shutil.copy(source / "best_trial", dest / "best_trial")
            else:
                # Get trial from Optuna and save
                trial = self.get_frozen_trial(trial_number)
                if trial is not None:
                    with open(dest / "best_trial", "wb") as f:
                        pickle.dump(trial, f)
                    self.log(f"  Created best_trial from database for #{trial_number}")
                else:
                    self.log(f"  Warning: Could not create best_trial for #{trial_number}")
                    return False

            self.log(f"Copied trial #{trial_number} to slot {slot}")
            return True
        except Exception as e:
            self.log(f"Error copying trial #{trial_number}: {e}")
            return False

    def update_manifest(self, trial_numbers: List[int], values: Dict[int, float]):
        """Update the ensemble manifest file."""
        # Sort trials by value descending
        sorted_trials = sorted(trial_numbers, key=lambda t: values.get(t, 0), reverse=True)

        # Build trial_values, trial_paths, actor_paths lists using actual model locations
        trial_values = []
        trial_paths = []
        actor_paths = []

        for t in sorted_trials:
            trial_values.append(values.get(t, 0))
            # Find actual model source path
            source = self.find_model_source(t)
            if source:
                # Use actual path found
                trial_paths.append(str(source.parent if source.name == "stored_agent" else source))
                actor_paths.append(str(source / "actor.pth"))
            else:
                # Fallback to default path (shouldn't happen if ensemble is properly synced)
                trial_paths.append(f"train_results/trial_{t}_1h")
                actor_paths.append(f"train_results/trial_{t}_1h/actor.pth")

        manifest = {
            "model_count": len(sorted_trials),
            "trial_numbers": sorted_trials,
            "trial_values": trial_values,
            "trial_paths": trial_paths,
            "actor_paths": actor_paths,
            "mean_value": sum(trial_values) / len(trial_values) if trial_values else 0,
            "best_value": max(trial_values) if trial_values else 0,
            "worst_value": min(trial_values) if trial_values else 0,
            "study_name": self.study_name,
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "adaptive",
        }

        with open(self.ensemble_dir / "ensemble_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    def signal_paper_trader_reload(self):
        """Signal the paper trader to reload models (via flag file)."""
        reload_flag = self.ensemble_dir / ".reload_models"
        reload_flag.touch()
        self.log("Signaled paper trader to reload models")

    def check_and_update(self) -> bool:
        """Check for new top trials and sync ensemble to match top N."""
        current = self.get_current_ensemble()
        current_trials = set(current.keys())

        self.log(f"Current ensemble: {len(current_trials)} models")

        # Get top trials from database
        top_trials = self.get_top_trials_from_db()
        if not top_trials:
            self.log("No trials found in database")
            return False

        # DEBUG: Log first few trials being checked
        self.log(f"Top {min(5, len(top_trials))} trials from DB: {[t['number'] for t in top_trials[:5]]}")

        # CRITICAL FIX: Filter to only trials with existing model files
        trials_with_models = []
        missing_count = 0
        for trial in top_trials:
            source = self.find_model_source(trial["number"])
            if source is not None:
                trials_with_models.append(trial)
                if len(trials_with_models) <= 3:  # Log first 3 found
                    self.log(f"  ✓ Trial #{trial['number']} (value: {trial['value']:.6f}) - found at {source}")
            else:
                missing_count += 1
                if missing_count <= 3:  # Log first 3 missing
                    self.log(f"  ✗ Trial #{trial['number']} (value: {trial['value']:.6f}) - model files not found")

        if not trials_with_models:
            self.log("ERROR: No trials have model files! Check train_results/cwd_tests/")
            return False

        self.log(f"Found {len(trials_with_models)} trials with model files (out of {len(top_trials)} in database)")

        # Target: top N trials from filtered list
        target_trials = {t["number"]: t["value"] for t in trials_with_models[:self.top_n]}
        target_set = set(target_trials.keys())

        # Find trials to add and remove
        to_add = target_set - current_trials
        to_remove = current_trials - target_set

        if not to_add and not to_remove:
            self.log("No changes needed - ensemble matches top trials")
            return False

        self.log(f"Syncing: +{len(to_add)} trials, -{len(to_remove)} trials")
        if to_add:
            self.log(f"  Adding: {sorted(to_add)}")
        if to_remove:
            self.log(f"  Removing: {sorted(to_remove)}")

        # Build slot mapping
        slot_to_trial = {}
        trial_to_slot = {}
        for model_dir in self.ensemble_dir.glob("model_*"):
            best_trial_path = model_dir / "best_trial"
            if best_trial_path.exists():
                try:
                    slot = int(model_dir.name.split("_")[1])
                    with open(best_trial_path, "rb") as f:
                        trial = pickle.load(f)
                    slot_to_trial[slot] = trial.number
                    trial_to_slot[trial.number] = slot
                except Exception:
                    pass

        # Remove old trials and free their slots
        freed_slots = []
        for trial_num in to_remove:
            if trial_num in trial_to_slot:
                slot = trial_to_slot[trial_num]
                model_dir = self.ensemble_dir / f"model_{slot}"
                if model_dir.exists():
                    shutil.rmtree(model_dir)
                    self.log(f"  Removed model #{trial_num} from slot {slot}")
                    freed_slots.append(slot)
                del current[trial_num]
                current_trials.remove(trial_num)

        # Add new trials using freed slots first
        changes_made = len(to_remove) > 0
        trial_values = {t["number"]: t["value"] for t in top_trials}

        used_slots = set(trial_to_slot.values()) - set(freed_slots)
        for trial_num in sorted(to_add, key=lambda t: target_trials.get(t, 0), reverse=True):
            # Use a freed slot or find new one
            if freed_slots:
                slot = freed_slots.pop(0)
            else:
                # Find next available slot
                slot = 0
                while slot in used_slots:
                    slot += 1

            if self.copy_model_to_ensemble(trial_num, slot):
                current_trials.add(trial_num)
                current[trial_num] = target_trials.get(trial_num, 0)
                used_slots.add(slot)
                trial_to_slot[trial_num] = slot
                changes_made = True

        if changes_made:
            # Update manifest
            self.update_manifest(list(current_trials), trial_values)
            # Signal reload
            self.signal_paper_trader_reload()
            self.log(f"Ensemble updated: {len(current_trials)} models")

        return changes_made

    def run(self):
        """Run the auto-updater daemon."""
        self.log("=" * 60)
        self.log("ENSEMBLE AUTO-UPDATER STARTED")
        self.log("=" * 60)
        self.log(f"Study: {self.study_name}")
        self.log(f"Ensemble: {self.ensemble_dir}")
        self.log(f"Top N: {self.top_n}")
        self.log(f"Check interval: {self.check_interval}s")
        self.log(f"Min improvement: {self.min_improvement}")
        self.log("=" * 60)

        while self.running:
            try:
                self.check_and_update()
            except Exception as e:
                self.log(f"Error in check cycle: {e}")

            # Sleep in small increments to allow signal handling
            for _ in range(self.check_interval):
                if not self.running:
                    break
                time.sleep(1)

        self.log("Auto-updater stopped")


def main():
    # Load configuration from .env.training
    from dotenv import load_dotenv
    import os

    load_dotenv('.env.training')
    default_study = os.getenv('ACTIVE_STUDY_NAME', 'cappuccino_1year_20251121')
    default_top_n = int(os.getenv('ENSEMBLE_TOP_N', '10'))
    default_interval = int(os.getenv('ENSEMBLE_UPDATE_INTERVAL', '300'))

    parser = argparse.ArgumentParser(description="Ensemble Auto-Updater")
    parser.add_argument("--study", default=default_study, help="Study name (default from .env.training)")
    parser.add_argument("--ensemble-dir", default="train_results/ensemble", help="Ensemble directory")
    parser.add_argument("--top-n", type=int, default=default_top_n, help="Number of models to keep")
    parser.add_argument("--interval", type=int, default=default_interval, help="Check interval in seconds")
    parser.add_argument("--min-improvement", type=float, default=0.0005, help="Minimum improvement to swap models")
    parser.add_argument("--once", action="store_true", help="Run once and exit")

    args = parser.parse_args()

    updater = EnsembleAutoUpdater(
        study_name=args.study,
        ensemble_dir=args.ensemble_dir,
        top_n=args.top_n,
        check_interval=args.interval,
        min_improvement=args.min_improvement,
    )

    if args.once:
        updater.check_and_update()
    else:
        updater.run()


if __name__ == "__main__":
    main()
