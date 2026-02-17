#!/usr/bin/env python3
"""
Complete Arena Setup - Clean Implementation

This script:
1. Identifies top N models from best study
2. Validates all model files exist
3. Creates/validates best_trial pickle files
4. Starts Arena with these models
5. Provides clear status reporting

NO CONFUSION - Everything validated step by step.
"""

import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import subprocess

# Import our validation module
from validate_models import ModelValidator, ModelInfo


class ArenaSetup:
    """Clean Arena setup with validation at every step."""

    def __init__(self, db_path: str = "databases/optuna_cappuccino.db"):
        self.db_path = db_path
        self.validator = ModelValidator(db_path)
        self.arena_dir = Path("arena_state")
        self.deployments_dir = Path("deployments")

    def find_best_study(self) -> Tuple[str, float, int]:
        """Find the study with the best performing models."""
        print("\n" + "="*70)
        print("STEP 1: Finding Best Study")
        print("="*70)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Find study with best max Sharpe
        cursor.execute("""
            SELECT
                s.study_name,
                COUNT(*) as trial_count,
                MAX(tv.value) as best_sharpe,
                AVG(tv.value) as avg_sharpe
            FROM trials t
            JOIN studies s ON t.study_id = s.study_id
            JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.state = 'COMPLETE'
            GROUP BY s.study_name
            ORDER BY best_sharpe DESC
            LIMIT 5
        """)

        results = cursor.fetchall()
        conn.close()

        print("\nTop 5 Studies by Best Sharpe:")
        print(f"{'Study':<40} {'Trials':<10} {'Best Sharpe':<15} {'Avg Sharpe':<15}")
        print("-" * 80)
        for study, count, best, avg in results:
            print(f"{study:<40} {count:<10} {best:<15.4f} {avg:<15.4f}")

        if not results:
            print("❌ ERROR: No completed studies found!")
            sys.exit(1)

        best_study, trial_count, best_sharpe, avg_sharpe = results[0]
        print(f"\n✓ Selected: {best_study}")
        print(f"  • {trial_count} completed trials")
        print(f"  • Best Sharpe: {best_sharpe:.4f}")
        print(f"  • Avg Sharpe: {avg_sharpe:.4f}")

        return best_study, best_sharpe, trial_count

    def get_top_models(self, study_name: str, top_n: int = 10) -> List[Tuple[int, int, float]]:
        """Get top N models from study."""
        print("\n" + "="*70)
        print(f"STEP 2: Getting Top {top_n} Models from {study_name}")
        print("="*70)

        trials = self.validator.get_top_trials(study_name, top_n)

        if not trials:
            print(f"❌ ERROR: No trials found in study {study_name}")
            sys.exit(1)

        print(f"\nTop {len(trials)} Trials:")
        print(f"{'Rank':<6} {'Trial #':<10} {'Trial ID':<12} {'Sharpe':<15}")
        print("-" * 50)
        # get_top_trials returns: (trial_id, trial_number, state, sharpe)
        for i, (trial_id, trial_num, state, sharpe) in enumerate(trials, 1):
            print(f"{i:<6} {trial_num:<10} {trial_id:<12} {sharpe:<15.4f}")

        # Reformat to (trial_num, trial_id, sharpe) for consistency
        return [(trial_num, trial_id, sharpe) for trial_id, trial_num, state, sharpe in trials]

    def validate_all_models(self, study_name: str, trials: List[Tuple[int, int, float]]) -> List[ModelInfo]:
        """Validate all models, creating missing best_trial files."""
        print("\n" + "="*70)
        print(f"STEP 3: Validating All Models")
        print("="*70)

        validated_models = []

        for i, (trial_num, trial_id, expected_sharpe) in enumerate(trials, 1):
            print(f"\n[{i}/{len(trials)}] Validating Trial #{trial_num} (ID={trial_id}, Sharpe={expected_sharpe:.4f})")

            # Use complete validation
            # validate_model_complete returns: (success, ModelInfo, List[ValidationResult])
            success, model_info, issues = self.validator.validate_model_complete(
                study_name=study_name,
                trial_number=trial_num,
                trial_id=trial_id,
                expected_sharpe=expected_sharpe,
                auto_fix=True  # Auto-create missing best_trial files
            )

            if not success or not model_info:
                print(f"  ❌ FAILED validation with {len(issues)} issue(s):")
                for issue in issues:
                    print(f"     • {issue.message}")
                continue

            # Show validation results
            print(f"  ✓ Study: {model_info.study_name}")
            print(f"  ✓ Files exist: {model_info.files_exist}")
            print(f"  ✓ best_trial: {model_info.best_trial_valid}")

            if model_info.best_trial_valid:
                print(f"    → Sharpe in model: {model_info.sharpe_value:.4f}")

            # Check for any failed validations
            has_issues = not (model_info.files_exist and model_info.best_trial_valid and model_info.sharpe_matches)
            if has_issues:
                print(f"  ⚠ Model has issues:")
                for issue in issues:
                    if not issue.passed:
                        print(f"     • {issue.message}")

            validated_models.append(model_info)

        # Summary
        fully_valid = sum(1 for m in validated_models if (m.files_exist and m.best_trial_valid and m.sharpe_matches))
        print(f"\n{'='*70}")
        print(f"Validation Summary: {fully_valid}/{len(validated_models)} models fully valid")
        print(f"{'='*70}")

        return validated_models

    def create_arena_config(self, study_name: str, models: List[ModelInfo]) -> Path:
        """Create Arena configuration file."""
        print("\n" + "="*70)
        print("STEP 4: Creating Arena Configuration")
        print("="*70)

        config = {
            "arena_name": f"arena_{study_name}",
            "study_name": study_name,
            "model_count": len(models),
            "models": []
        }

        for i, model in enumerate(models):
            all_valid = model.files_exist and model.best_trial_valid and model.sharpe_matches
            model_config = {
                "model_id": i,
                "trial_number": model.trial_number,
                "trial_id": model.trial_id,
                "sharpe_value": model.sharpe_value,
                "model_path": str(model.model_dir),
                "validation_status": {
                    "all_valid": all_valid,
                    "files_exist": model.files_exist,
                    "best_trial_valid": model.best_trial_valid,
                    "sharpe_matches": model.sharpe_matches
                }
            }
            config["models"].append(model_config)

        # Save config
        config_path = self.arena_dir / "arena_config.json"
        self.arena_dir.mkdir(exist_ok=True)

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n✓ Configuration saved to: {config_path}")
        print(f"  • {len(models)} models configured")
        print(f"  • Study: {study_name}")

        return config_path

    def deploy_to_arena(self, models: List[ModelInfo]) -> List[Path]:
        """Deploy models to Arena deployments directory."""
        print("\n" + "="*70)
        print("STEP 5: Deploying Models to Arena")
        print("="*70)

        # Clean existing deployments
        if self.deployments_dir.exists():
            print(f"\n🗑️  Cleaning existing deployments...")
            shutil.rmtree(self.deployments_dir)

        self.deployments_dir.mkdir(exist_ok=True)

        deployed_paths = []

        for i, model in enumerate(models):
            deployment_dir = self.deployments_dir / f"model_{i}"
            deployment_dir.mkdir(exist_ok=True)

            print(f"\n[{i+1}/{len(models)}] Deploying Trial #{model.trial_number} to {deployment_dir}")

            # Copy actor.pth
            if model.model_dir.exists():
                actor_src = model.model_dir / "actor.pth"
                if actor_src.exists():
                    shutil.copy(actor_src, deployment_dir / "actor.pth")
                    print(f"  ✓ Copied actor.pth")
                else:
                    # Check stored_agent subdirectory
                    actor_alt = model.model_dir / "stored_agent" / "actor.pth"
                    if actor_alt.exists():
                        shutil.copy(actor_alt, deployment_dir / "actor.pth")
                        print(f"  ✓ Copied actor.pth (from stored_agent)")
                    else:
                        print(f"  ❌ WARNING: No actor.pth found!")

            # Copy best_trial
            best_trial_src = model.model_dir / "best_trial"
            if best_trial_src.exists():
                shutil.copy(best_trial_src, deployment_dir / "best_trial")
                print(f"  ✓ Copied best_trial")
            else:
                print(f"  ⚠ No best_trial file (will create from DB)")
                # Create best_trial from database
                result = self.validator.create_best_trial_file(
                    model.trial_id,
                    deployment_dir / "best_trial"
                )
                if result.passed:
                    print(f"  ✓ Created best_trial from database")
                else:
                    print(f"  ❌ Failed to create best_trial: {result.message}")

            # Create metadata file
            all_valid = model.files_exist and model.best_trial_valid and model.sharpe_matches
            metadata = {
                "trial_number": model.trial_number,
                "trial_id": model.trial_id,
                "sharpe_value": model.sharpe_value,
                "source_dir": str(model.model_dir),
                "validation_passed": all_valid
            }

            with open(deployment_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"  ✓ Created metadata.json")

            deployed_paths.append(deployment_dir)

        print(f"\n{'='*70}")
        print(f"✓ Deployed {len(deployed_paths)} models to {self.deployments_dir}")
        print(f"{'='*70}")

        return deployed_paths

    def start_arena(self, config_path: Path, background: bool = True) -> bool:
        """Start Arena with deployed models."""
        print("\n" + "="*70)
        print("STEP 6: Starting Arena")
        print("="*70)

        # Check if arena_runner.py exists
        arena_script = Path("arena_runner.py")
        if not arena_script.exists():
            print(f"❌ ERROR: {arena_script} not found!")
            return False

        print(f"\n✓ Found arena runner script: {arena_script}")
        print(f"✓ Using config: {config_path}")

        # Build command
        cmd = ["python", str(arena_script), "--config", str(config_path)]

        if background:
            print("\n🚀 Starting Arena in background...")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
            print(f"✓ Arena started (PID: {process.pid})")
            print(f"\nMonitor with:")
            print(f"  • tail -f logs/arena.log")
            print(f"  • ./status_arena.sh")
            print(f"  • dashboard.py (Page 3)")
        else:
            print("\n🚀 Starting Arena in foreground...")
            print("(Press Ctrl+C to stop)")
            result = subprocess.run(cmd)
            return result.returncode == 0

        return True

    def print_summary(self, study_name: str, models: List[ModelInfo], deployed: List[Path]):
        """Print final summary."""
        print("\n" + "="*70)
        print("ARENA SETUP COMPLETE")
        print("="*70)

        print(f"\n📊 Study: {study_name}")
        print(f"🎯 Models Deployed: {len(deployed)}")
        fully_valid_count = sum(1 for m in models if (m.files_exist and m.best_trial_valid and m.sharpe_matches))
        print(f"✓ Validation: {fully_valid_count}/{len(models)} fully valid")

        print("\n📈 Top 5 Models by Sharpe:")
        sorted_models = sorted(models, key=lambda m: m.sharpe_value, reverse=True)[:5]
        for i, model in enumerate(sorted_models, 1):
            all_valid = model.files_exist and model.best_trial_valid and model.sharpe_matches
            status = "✓" if all_valid else "⚠"
            print(f"  {i}. Trial #{model.trial_number} - Sharpe {model.sharpe_value:.4f} {status}")

        print("\n📁 Files Created:")
        print(f"  • Config: arena_state/arena_config.json")
        print(f"  • Deployments: deployments/model_0 ... model_{len(deployed)-1}")
        print(f"  • Each with: actor.pth, best_trial, metadata.json")

        print("\n📊 Monitoring:")
        print(f"  • Dashboard: python dashboard.py (Page 3)")
        print(f"  • Status: ./status_arena.sh")
        print(f"  • Logs: tail -f logs/arena.log")

        print("\n🎮 Control:")
        print(f"  • Stop: ./stop_arena.sh")
        print(f"  • Restart: ./start_arena.sh")

        print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Setup Arena with validated models")
    parser.add_argument("--study", type=str, help="Study name (auto-detect if not specified)")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top models (default: 10)")
    parser.add_argument("--db", type=str, default="databases/optuna_cappuccino.db", help="Database path")
    parser.add_argument("--no-start", action="store_true", help="Don't start Arena after setup")
    parser.add_argument("--foreground", action="store_true", help="Run Arena in foreground")

    args = parser.parse_args()

    print("="*70)
    print("ARENA SETUP - CLEAN IMPLEMENTATION")
    print("="*70)
    print("\nThis script will:")
    print("1. Find the best study (or use specified)")
    print("2. Get top N models")
    print("3. Validate all models")
    print("4. Create Arena configuration")
    print("5. Deploy models to Arena")
    print("6. Start Arena")
    print("\nAll steps are validated. No confusion.")

    setup = ArenaSetup(args.db)

    # Step 1: Find best study
    if args.study:
        study_name = args.study
        print(f"\n✓ Using specified study: {study_name}")
    else:
        study_name, best_sharpe, trial_count = setup.find_best_study()

    # Step 2: Get top models
    trials = setup.get_top_models(study_name, args.top_n)

    if not trials:
        print("\n❌ ERROR: No trials found. Cannot proceed.")
        sys.exit(1)

    # Step 3: Validate all models
    models = setup.validate_all_models(study_name, trials)

    if not models:
        print("\n❌ ERROR: No models validated successfully. Cannot proceed.")
        sys.exit(1)

    # Check if we have enough valid models
    fully_valid = [m for m in models if (m.files_exist and m.best_trial_valid and m.sharpe_matches)]
    if len(fully_valid) < 3:
        print(f"\n⚠ WARNING: Only {len(fully_valid)} fully valid models!")
        response = input("Continue anyway? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            sys.exit(1)

    # Step 4: Create configuration
    config_path = setup.create_arena_config(study_name, models)

    # Step 5: Deploy models
    deployed = setup.deploy_to_arena(models)

    # Step 6: Start Arena
    if not args.no_start:
        success = setup.start_arena(config_path, background=not args.foreground)
        if not success:
            print("\n❌ ERROR: Failed to start Arena")
            sys.exit(1)
    else:
        print("\n⏸️  Skipping Arena start (--no-start specified)")
        print("\nTo start manually:")
        print(f"  python arena_runner.py --config {config_path}")

    # Final summary
    setup.print_summary(study_name, models, deployed)


if __name__ == "__main__":
    main()
