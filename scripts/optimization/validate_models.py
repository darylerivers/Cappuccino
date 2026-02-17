#!/usr/bin/env python3
"""
Model Validation Script

Validates models before deployment to catch issues early.
Checks:
1. Database integrity (study exists, trials complete)
2. File existence (actor.pth, best_trial, etc.)
3. Sharpe value consistency (DB vs manifest vs file)
4. Trial uniqueness (no ambiguous trial numbers)
5. Study consistency (all references match)

Usage:
    python validate_models.py --study cappuccino_alpaca_v2 --top-n 10
    python validate_models.py --trial-ids 686,687,521 --study cappuccino_alpaca_v2
    python validate_models.py --manifest train_results/ensemble_best/ensemble_manifest.json
"""

import argparse
import json
import pickle
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class ValidationResult:
    """Result of validation check."""
    passed: bool
    message: str
    details: Dict = field(default_factory=dict)


@dataclass
class ModelInfo:
    """Complete information about a model."""
    # Database info
    study_name: str
    trial_number: int
    trial_id: int
    sharpe_value: float
    state: str

    # File paths
    model_dir: Path
    actor_path: Path
    critic_path: Optional[Path] = None
    best_trial_path: Optional[Path] = None

    # Validation status
    files_exist: bool = False
    best_trial_valid: bool = False
    sharpe_matches: bool = False

    # Loaded data
    params: Dict = field(default_factory=dict)
    trial_object: Optional[object] = None


class ModelValidator:
    """Validates models for deployment."""

    def __init__(self, db_path: str = "databases/optuna_cappuccino.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()

    def validate_study_exists(self, study_name: str) -> ValidationResult:
        """Check if study exists in database."""
        self.cursor.execute(
            "SELECT study_id, study_name FROM studies WHERE study_name = ?",
            (study_name,)
        )
        result = self.cursor.fetchone()

        if result:
            return ValidationResult(
                passed=True,
                message=f"✓ Study '{study_name}' exists",
                details={"study_id": result[0]}
            )
        else:
            # List available studies
            self.cursor.execute("SELECT study_name FROM studies ORDER BY study_name")
            available = [row[0] for row in self.cursor.fetchall()]
            return ValidationResult(
                passed=False,
                message=f"✗ Study '{study_name}' not found in database",
                details={"available_studies": available}
            )

    def get_top_trials(self, study_name: str, top_n: int) -> List[Tuple]:
        """Get top N trials from study."""
        self.cursor.execute("""
            SELECT t.trial_id, t.number, t.state, tv.value
            FROM trials t
            JOIN studies s ON t.study_id = s.study_id
            LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE s.study_name = ? AND t.state = 'COMPLETE'
            ORDER BY tv.value DESC
            LIMIT ?
        """, (study_name, top_n))

        return self.cursor.fetchall()

    def get_trial_by_id(self, trial_id: int) -> Optional[Tuple]:
        """Get trial by trial_id."""
        self.cursor.execute("""
            SELECT t.trial_id, t.number, t.state, tv.value, s.study_name
            FROM trials t
            JOIN studies s ON t.study_id = s.study_id
            LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.trial_id = ?
        """, (trial_id,))

        return self.cursor.fetchone()

    def get_trial_params(self, trial_id: int) -> Dict:
        """Get trial parameters from database."""
        self.cursor.execute("""
            SELECT param_name, param_value
            FROM trial_params
            WHERE trial_id = ?
        """, (trial_id,))

        return {row[0]: row[1] for row in self.cursor.fetchall()}

    def check_trial_uniqueness(self, trial_number: int) -> ValidationResult:
        """Check if trial number appears in multiple studies."""
        self.cursor.execute("""
            SELECT s.study_name, t.trial_id, tv.value
            FROM trials t
            JOIN studies s ON t.study_id = s.study_id
            LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.number = ?
        """, (trial_number,))

        results = self.cursor.fetchall()

        if len(results) == 1:
            return ValidationResult(
                passed=True,
                message=f"✓ Trial {trial_number} is unique",
                details={"count": 1}
            )
        else:
            studies = [{"study": r[0], "trial_id": r[1], "sharpe": r[2]} for r in results]
            return ValidationResult(
                passed=False,
                message=f"⚠ Trial {trial_number} appears in {len(results)} studies",
                details={"occurrences": studies}
            )

    def find_model_directory(self, trial_number: int) -> Optional[Path]:
        """Find model directory for trial."""
        # Common patterns
        patterns = [
            f"train_results/cwd_tests/trial_{trial_number}_1h",
            f"train_results/cwd_tests/trial_{trial_number}",
            f"train_results/trial_{trial_number}_1h",
            f"train_results/trial_{trial_number}",
        ]

        for pattern in patterns:
            path = Path(pattern)
            if path.exists():
                return path

        return None

    def validate_model_files(self, model_dir: Path) -> ValidationResult:
        """Check all required files exist."""
        required_files = {
            "actor.pth": model_dir / "actor.pth",
            "critic.pth": model_dir / "critic.pth",
        }

        optional_files = {
            "best_trial": model_dir / "best_trial",
            "recorder.npy": model_dir / "recorder.npy",
        }

        missing_required = []
        missing_optional = []

        for name, path in required_files.items():
            if not path.exists():
                missing_required.append(name)

        for name, path in optional_files.items():
            if not path.exists():
                missing_optional.append(name)

        if missing_required:
            return ValidationResult(
                passed=False,
                message=f"✗ Missing required files: {', '.join(missing_required)}",
                details={
                    "missing_required": missing_required,
                    "missing_optional": missing_optional,
                    "model_dir": str(model_dir)
                }
            )
        else:
            msg = f"✓ All required files exist"
            if missing_optional:
                msg += f" (optional missing: {', '.join(missing_optional)})"
            return ValidationResult(
                passed=True,
                message=msg,
                details={"missing_optional": missing_optional}
            )

    def validate_best_trial_file(self, best_trial_path: Path, expected_trial_number: int, expected_sharpe: float) -> ValidationResult:
        """Validate best_trial pickle file."""
        if not best_trial_path.exists():
            return ValidationResult(
                passed=False,
                message="✗ best_trial file does not exist",
                details={"can_create": True}
            )

        try:
            with open(best_trial_path, 'rb') as f:
                trial = pickle.load(f)

            # Check trial number matches
            if trial.number != expected_trial_number:
                return ValidationResult(
                    passed=False,
                    message=f"✗ best_trial has wrong trial number (expected {expected_trial_number}, got {trial.number})",
                    details={"expected": expected_trial_number, "actual": trial.number}
                )

            # Check Sharpe value matches (allow small tolerance)
            if trial.value is not None and abs(trial.value - expected_sharpe) > 0.001:
                return ValidationResult(
                    passed=False,
                    message=f"✗ best_trial Sharpe mismatch (expected {expected_sharpe:.4f}, got {trial.value:.4f})",
                    details={"expected": expected_sharpe, "actual": trial.value}
                )

            return ValidationResult(
                passed=True,
                message=f"✓ best_trial valid (trial {trial.number}, Sharpe {trial.value:.4f})",
                details={"trial_number": trial.number, "sharpe": trial.value}
            )

        except Exception as e:
            return ValidationResult(
                passed=False,
                message=f"✗ Failed to load best_trial: {e}",
                details={"error": str(e)}
            )

    def validate_actor_file(self, actor_path: Path) -> ValidationResult:
        """Validate actor.pth file."""
        if not actor_path.exists():
            return ValidationResult(
                passed=False,
                message="✗ actor.pth does not exist"
            )

        try:
            state_dict = torch.load(actor_path, map_location='cpu', weights_only=True)

            # Check it's a state dict
            if not isinstance(state_dict, dict):
                return ValidationResult(
                    passed=False,
                    message="✗ actor.pth is not a valid state dict"
                )

            # Get input dimension from first layer
            input_dim = None
            if 'net.0.weight' in state_dict:
                input_dim = state_dict['net.0.weight'].shape[1]

            return ValidationResult(
                passed=True,
                message=f"✓ actor.pth valid (input_dim={input_dim})",
                details={"input_dim": input_dim, "num_params": len(state_dict)}
            )

        except Exception as e:
            return ValidationResult(
                passed=False,
                message=f"✗ Failed to load actor.pth: {e}",
                details={"error": str(e)}
            )

    def create_best_trial_file(self, trial_id: int, output_path: Path) -> ValidationResult:
        """Create best_trial file from database."""
        try:
            import optuna

            # Get study name for this trial
            result = self.get_trial_by_id(trial_id)
            if not result:
                return ValidationResult(
                    passed=False,
                    message=f"✗ Trial {trial_id} not found in database"
                )

            trial_id_db, trial_number, state, sharpe, study_name = result

            # Load study
            study = optuna.load_study(
                study_name=study_name,
                storage=f"sqlite:///{self.db_path}"
            )

            # Find trial in study
            trial_obj = None
            for trial in study.trials:
                if trial.number == trial_number:
                    trial_obj = trial
                    break

            if not trial_obj:
                return ValidationResult(
                    passed=False,
                    message=f"✗ Could not find trial {trial_number} in study {study_name}"
                )

            # Save to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump(trial_obj, f)

            return ValidationResult(
                passed=True,
                message=f"✓ Created best_trial file (trial {trial_number}, Sharpe {sharpe:.4f})",
                details={"trial_number": trial_number, "sharpe": sharpe, "path": str(output_path)}
            )

        except Exception as e:
            return ValidationResult(
                passed=False,
                message=f"✗ Failed to create best_trial: {e}",
                details={"error": str(e)}
            )

    def validate_model_complete(self, study_name: str, trial_number: int, trial_id: int, expected_sharpe: float, auto_fix: bool = False) -> Tuple[bool, ModelInfo, List[ValidationResult]]:
        """Complete validation of a single model."""
        results = []

        # Find model directory
        model_dir = self.find_model_directory(trial_number)
        if not model_dir:
            results.append(ValidationResult(
                passed=False,
                message=f"✗ Model directory not found for trial {trial_number}"
            ))
            return False, None, results

        results.append(ValidationResult(
            passed=True,
            message=f"✓ Found model directory: {model_dir}"
        ))

        # Create ModelInfo
        model_info = ModelInfo(
            study_name=study_name,
            trial_number=trial_number,
            trial_id=trial_id,
            sharpe_value=expected_sharpe,
            state="COMPLETE",
            model_dir=model_dir,
            actor_path=model_dir / "actor.pth",
            critic_path=model_dir / "critic.pth",
            best_trial_path=model_dir / "best_trial"
        )

        # Validate files
        file_result = self.validate_model_files(model_dir)
        results.append(file_result)
        model_info.files_exist = file_result.passed

        # Validate actor
        actor_result = self.validate_actor_file(model_info.actor_path)
        results.append(actor_result)

        # Validate best_trial
        best_trial_result = self.validate_best_trial_file(
            model_info.best_trial_path,
            trial_number,
            expected_sharpe
        )
        results.append(best_trial_result)
        model_info.best_trial_valid = best_trial_result.passed

        # Check if Sharpe values match
        if best_trial_result.passed and best_trial_result.details:
            loaded_sharpe = best_trial_result.details.get("sharpe", 0)
            sharpe_diff = abs(loaded_sharpe - expected_sharpe)
            model_info.sharpe_matches = (sharpe_diff < 0.01)  # Within 1% tolerance
        else:
            model_info.sharpe_matches = False

        # Auto-fix if requested and needed
        if auto_fix and not best_trial_result.passed and best_trial_result.details.get("can_create"):
            fix_result = self.create_best_trial_file(trial_id, model_info.best_trial_path)
            results.append(fix_result)
            model_info.best_trial_valid = fix_result.passed

            # Re-check sharpe match after auto-fix
            if fix_result.passed and fix_result.details:
                loaded_sharpe = fix_result.details.get("sharpe", 0)
                sharpe_diff = abs(loaded_sharpe - expected_sharpe)
                model_info.sharpe_matches = (sharpe_diff < 0.01)

        # Get params
        model_info.params = self.get_trial_params(trial_id)

        # Overall pass/fail
        all_passed = all(r.passed for r in results)

        return all_passed, model_info, results

    def validate_manifest(self, manifest_path: Path) -> Tuple[bool, List[ValidationResult]]:
        """Validate ensemble manifest file."""
        results = []

        if not manifest_path.exists():
            results.append(ValidationResult(
                passed=False,
                message=f"✗ Manifest not found: {manifest_path}"
            ))
            return False, results

        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except Exception as e:
            results.append(ValidationResult(
                passed=False,
                message=f"✗ Failed to parse manifest: {e}"
            ))
            return False, results

        # Check required fields
        required_fields = ["model_count", "trial_numbers", "trial_values", "actor_paths"]
        missing = [f for f in required_fields if f not in manifest]
        if missing:
            results.append(ValidationResult(
                passed=False,
                message=f"✗ Manifest missing fields: {', '.join(missing)}"
            ))
            return False, results

        # Check study_name exists
        if "study_name" not in manifest:
            results.append(ValidationResult(
                passed=False,
                message="⚠ Manifest missing 'study_name' field (CRITICAL BUG!)"
            ))
            return False, results

        results.append(ValidationResult(
            passed=True,
            message=f"✓ Manifest structure valid ({manifest['model_count']} models)"
        ))

        # Validate study exists
        study_result = self.validate_study_exists(manifest['study_name'])
        results.append(study_result)

        # Check counts match
        counts = [
            len(manifest['trial_numbers']),
            len(manifest['trial_values']),
            len(manifest['actor_paths'])
        ]
        if len(set(counts)) > 1:
            results.append(ValidationResult(
                passed=False,
                message=f"✗ Array length mismatch: {counts}"
            ))
            return False, results

        # Validate each model
        for i, (trial_num, sharpe, actor_path) in enumerate(
            zip(manifest['trial_numbers'], manifest['trial_values'], manifest['actor_paths'])
        ):
            actor_path_obj = Path(actor_path)
            if not actor_path_obj.exists():
                results.append(ValidationResult(
                    passed=False,
                    message=f"✗ Model {i+1}: actor.pth not found at {actor_path}"
                ))
                return False, results

        results.append(ValidationResult(
            passed=True,
            message=f"✓ All {manifest['model_count']} model files exist"
        ))

        return True, results


def print_results(results: List[ValidationResult], verbose: bool = False):
    """Print validation results."""
    for result in results:
        print(result.message)
        if verbose and result.details:
            for key, value in result.details.items():
                print(f"    {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Validate models before deployment")
    parser.add_argument("--study", help="Study name to validate")
    parser.add_argument("--top-n", type=int, help="Validate top N models from study")
    parser.add_argument("--trial-ids", help="Comma-separated trial IDs to validate")
    parser.add_argument("--manifest", help="Path to manifest file to validate")
    parser.add_argument("--auto-fix", action="store_true", help="Auto-fix issues (create missing best_trial files)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    validator = ModelValidator()

    print("=" * 80)
    print("MODEL VALIDATION")
    print("=" * 80)
    print()

    # Mode 1: Validate manifest
    if args.manifest:
        print(f"Validating manifest: {args.manifest}")
        print("-" * 80)
        passed, results = validator.validate_manifest(Path(args.manifest))
        print_results(results, args.verbose)
        print()
        if passed:
            print("✓ Manifest validation PASSED")
            return 0
        else:
            print("✗ Manifest validation FAILED")
            return 1

    # Mode 2: Validate top N from study
    if args.study and args.top_n:
        print(f"Validating top {args.top_n} models from study: {args.study}")
        print("-" * 80)

        # Check study exists
        study_result = validator.validate_study_exists(args.study)
        print_results([study_result], args.verbose)
        if not study_result.passed:
            return 1
        print()

        # Get top trials
        trials = validator.get_top_trials(args.study, args.top_n)
        print(f"Found {len(trials)} trials")
        print()

        all_passed = True
        validated_models = []

        for trial_id, trial_number, state, sharpe in trials:
            print(f"Trial {trial_number} (ID {trial_id}, Sharpe {sharpe:.4f})")
            print("-" * 80)

            passed, model_info, results = validator.validate_model_complete(
                args.study, trial_number, trial_id, sharpe, auto_fix=args.auto_fix
            )
            print_results(results, args.verbose)
            print()

            if passed:
                validated_models.append(model_info)
            else:
                all_passed = False

        print("=" * 80)
        if all_passed:
            print(f"✓ ALL {len(validated_models)} MODELS VALIDATED")
            print()
            print("Models ready for deployment:")
            for m in validated_models:
                print(f"  Trial {m.trial_number}: Sharpe {m.sharpe_value:.4f} ({m.model_dir})")
            return 0
        else:
            print(f"✗ VALIDATION FAILED (passed: {len(validated_models)}/{len(trials)})")
            return 1

    # Mode 3: Validate specific trial IDs
    if args.trial_ids:
        trial_ids = [int(x.strip()) for x in args.trial_ids.split(',')]
        print(f"Validating {len(trial_ids)} specific trials")
        print("-" * 80)

        all_passed = True
        for trial_id in trial_ids:
            result = validator.get_trial_by_id(trial_id)
            if not result:
                print(f"✗ Trial ID {trial_id} not found")
                all_passed = False
                continue

            trial_id_db, trial_number, state, sharpe, study_name = result
            print(f"Trial {trial_number} from {study_name} (ID {trial_id}, Sharpe {sharpe:.4f})")
            print("-" * 80)

            passed, model_info, results = validator.validate_model_complete(
                study_name, trial_number, trial_id, sharpe, auto_fix=args.auto_fix
            )
            print_results(results, args.verbose)
            print()

            if not passed:
                all_passed = False

        if all_passed:
            print("✓ ALL TRIALS VALIDATED")
            return 0
        else:
            print("✗ SOME TRIALS FAILED VALIDATION")
            return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
