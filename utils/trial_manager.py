#!/usr/bin/env python3
"""
Trial Manager - Archive Top Performers and Clean Old Trials

Manages the lifecycle of training trials:
- Archives top 10% of trials with their models
- Cleans up old logs and trial data
- Maintains a registry of archived trials
- Provides easy deployment of archived models
"""

import os
import sys
import json
import shutil
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import pickle

# Add parent to path for imports
PARENT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PARENT_DIR))

from utils.trial_naming import generate_trial_vin, is_top_percentile, grade_to_numeric


class TrialManager:
    """Manage trial archival and cleanup."""

    def __init__(self, base_dir: Path = None, archive_dir: Path = None):
        """
        Initialize trial manager.

        Args:
            base_dir: Base directory for the project (defaults to /opt/user-data/experiment/cappuccino)
            archive_dir: Directory for archived trials (defaults to base_dir/trial_archive)
        """
        self.base_dir = Path(base_dir) if base_dir else Path("/opt/user-data/experiment/cappuccino")
        self.archive_dir = Path(archive_dir) if archive_dir else self.base_dir / "trial_archive"
        self.registry_file = self.archive_dir / "trial_registry.json"
        # Try optuna database first, fallback to pipeline_v2.db
        optuna_db = self.base_dir / "databases/optuna_cappuccino.db"
        self.db_path = optuna_db if optuna_db.exists() else self.base_dir / "pipeline_v2.db"

        # Create archive directory
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        # Load or create registry
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        """Load the trial registry or create a new one."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {
            'trials': [],
            'last_updated': datetime.now().isoformat(),
            'stats': {
                'total_archived': 0,
                'by_grade': {},
                'by_model': {}
            }
        }

    def _save_registry(self):
        """Save the trial registry."""
        self.registry['last_updated'] = datetime.now().isoformat()
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def get_study_trials(self, study_name: str = None) -> List[Dict]:
        """
        Get trials from Optuna database.

        Args:
            study_name: Specific study name, or None for latest

        Returns:
            List of trial dictionaries with metadata
        """
        if not self.db_path.exists():
            print(f"⚠️  Database not found: {self.db_path}")
            return []

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Get study info
        if study_name:
            cursor.execute(
                "SELECT study_id, study_name FROM studies WHERE study_name = ?",
                (study_name,)
            )
        else:
            # Get most recent study
            cursor.execute(
                "SELECT study_id, study_name FROM studies ORDER BY study_id DESC LIMIT 1"
            )

        study_row = cursor.fetchone()
        if not study_row:
            print("⚠️  No studies found in database")
            conn.close()
            return []

        study_id, study_name = study_row
        print(f"📊 Loading trials from study: {study_name} (ID: {study_id})")

        # Get all completed trials with values (Optuna schema)
        cursor.execute("""
            SELECT t.trial_id, t.number, tv.value, t.datetime_start, t.datetime_complete, t.state
            FROM trials t
            LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.study_id = ? AND t.state = 'COMPLETE'
            ORDER BY tv.value DESC
        """, (study_id,))

        trials = []
        for row in cursor.fetchall():
            trial_id, number, sharpe, start_time, end_time, state = row

            # Get trial parameters
            cursor.execute("""
                SELECT param_name, param_value
                FROM trial_params
                WHERE trial_id = ?
            """, (trial_id,))

            params = {}
            for param_name, param_value in cursor.fetchall():
                # Try to convert to appropriate type
                if isinstance(param_value, (int, float)):
                    params[param_name] = param_value
                else:
                    try:
                        params[param_name] = float(param_value) if '.' in str(param_value) or 'e' in str(param_value).lower() else int(param_value)
                    except (ValueError, AttributeError, TypeError):
                        params[param_name] = param_value

            # Get trial user attributes (metadata) - Optuna schema uses 'value_json'
            cursor.execute("""
                SELECT key, value_json
                FROM trial_user_attributes
                WHERE trial_id = ?
            """, (trial_id,))

            user_attrs = {}
            for key, value_json in cursor.fetchall():
                try:
                    user_attrs[key] = json.loads(value_json) if value_json else None
                except (json.JSONDecodeError, TypeError):
                    user_attrs[key] = value_json

            trials.append({
                'trial_id': trial_id,
                'number': number,
                'sharpe': sharpe if sharpe is not None else 0.0,
                'params': params,
                'user_attrs': user_attrs,
                'start_time': start_time,
                'end_time': end_time,
                'study_name': study_name
            })

        conn.close()
        print(f"✅ Loaded {len(trials)} completed trials")
        return trials

    def archive_top_trials(
        self,
        study_name: str = None,
        top_percentile: int = 10,
        min_sharpe: float = 0.10
    ) -> List[Dict]:
        """
        Archive the top performing trials.

        Args:
            study_name: Study name to archive from (None = latest)
            top_percentile: Keep top X% of trials (default 10%)
            min_sharpe: Minimum Sharpe ratio to consider (default 0.10)

        Returns:
            List of archived trial metadata
        """
        print(f"\n{'='*70}")
        print(f"ARCHIVING TOP {top_percentile}% OF TRIALS")
        print(f"{'='*70}\n")

        # Get all trials
        trials = self.get_study_trials(study_name)
        if not trials:
            print("⚠️  No trials to archive")
            return []

        # Filter by minimum Sharpe
        trials = [t for t in trials if t['sharpe'] >= min_sharpe]
        print(f"📊 {len(trials)} trials meet minimum Sharpe threshold ({min_sharpe:.2f})")

        if not trials:
            print("⚠️  No trials meet minimum threshold")
            return []

        # Sort by Sharpe ratio
        trials.sort(key=lambda x: x['sharpe'], reverse=True)

        # Calculate top N trials
        n_top = max(1, int(len(trials) * top_percentile / 100))
        top_trials = trials[:n_top]

        print(f"🎯 Archiving top {n_top} trials (Sharpe range: {top_trials[-1]['sharpe']:.4f} to {top_trials[0]['sharpe']:.4f})")

        archived = []
        for i, trial in enumerate(top_trials, 1):
            try:
                metadata = self._archive_single_trial(trial, i, n_top)
                archived.append(metadata)
                print(f"  ✅ [{i}/{n_top}] {metadata['vin']}")
            except Exception as e:
                print(f"  ❌ [{i}/{n_top}] Trial {trial['number']}: {e}")

        # Update registry
        self._update_registry_stats(archived)
        self._save_registry()

        print(f"\n✅ Archived {len(archived)} trials to {self.archive_dir}")
        return archived

    def _archive_single_trial(self, trial: Dict, rank: int, total: int) -> Dict:
        """Archive a single trial with its model and metadata."""
        # Generate VIN code
        vin, grade, metadata = generate_trial_vin(
            model_type='ppo',  # Default, could be extracted from user_attrs
            sharpe=trial['sharpe'],
            hyperparams=trial['params'],
            timestamp=datetime.fromisoformat(trial['end_time']) if trial['end_time'] else datetime.now()
        )

        # Create trial directory
        trial_dir = self.archive_dir / vin
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Enhanced metadata
        metadata.update({
            'trial_number': trial['number'],
            'trial_id': trial['trial_id'],
            'study_name': trial['study_name'],
            'rank': rank,
            'percentile': (rank / total) * 100,
            'archived_at': datetime.now().isoformat(),
        })

        # Save metadata
        with open(trial_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Try to find and copy the model
        # Models are typically in: experiments/trial_{number}/actor.pth
        model_sources = [
            self.base_dir / "experiments" / f"trial_{trial['number']}" / "actor.pth",
            self.base_dir / "experiments" / f"trial_{trial['number']}" / "agent.pkl",
        ]

        model_copied = False
        for source in model_sources:
            if source.exists():
                shutil.copy2(source, trial_dir / source.name)
                model_copied = True
                metadata['model_file'] = source.name
                break

        if not model_copied:
            print(f"    ⚠️  Model not found for trial {trial['number']}")
            metadata['model_file'] = None

        # Save hyperparameters as separate JSON for easy access
        with open(trial_dir / 'hyperparams.json', 'w') as f:
            json.dump(trial['params'], f, indent=2)

        # Add to registry
        self.registry['trials'].append(metadata)

        return metadata

    def _update_registry_stats(self, archived_trials: List[Dict]):
        """Update registry statistics."""
        self.registry['stats']['total_archived'] = len(self.registry['trials'])

        # Count by grade
        grade_counts = {}
        model_counts = {}
        for trial in self.registry['trials']:
            grade = trial['grade']
            model = trial['model_type']
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
            model_counts[model] = model_counts.get(model, 0) + 1

        self.registry['stats']['by_grade'] = grade_counts
        self.registry['stats']['by_model'] = model_counts

    def list_archived_trials(
        self,
        min_grade: str = 'F',
        limit: int = None
    ) -> List[Dict]:
        """
        List archived trials.

        Args:
            min_grade: Minimum grade to show (S, A, B, C, D, F)
            limit: Maximum number of trials to return

        Returns:
            List of trial metadata
        """
        min_grade_num = grade_to_numeric(min_grade)
        filtered = [
            t for t in self.registry['trials']
            if grade_to_numeric(t['grade']) >= min_grade_num
        ]

        # Sort by grade then sharpe
        filtered.sort(key=lambda x: (grade_to_numeric(x['grade']), x['sharpe']), reverse=True)

        if limit:
            filtered = filtered[:limit]

        return filtered

    def get_best_trial_for_deployment(self) -> Optional[Dict]:
        """Get the best archived trial for deployment."""
        if not self.registry['trials']:
            return None

        # Sort by grade numeric then sharpe
        trials = sorted(
            self.registry['trials'],
            key=lambda x: (grade_to_numeric(x['grade']), x['sharpe']),
            reverse=True
        )

        return trials[0] if trials else None

    def clean_old_trials(
        self,
        keep_top_n: int = None,
        keep_days: int = 30,
        dry_run: bool = False
    ):
        """
        Clean up old trial data from the experiments directory.

        Args:
            keep_top_n: Keep top N trials by Sharpe (None = use days)
            keep_days: Keep trials from last N days (ignored if keep_top_n set)
            dry_run: Don't actually delete, just show what would be deleted
        """
        print(f"\n{'='*70}")
        print(f"CLEANING OLD TRIALS" + (" (DRY RUN)" if dry_run else ""))
        print(f"{'='*70}\n")

        experiments_dir = self.base_dir / "experiments"
        if not experiments_dir.exists():
            print("⚠️  No experiments directory found")
            return

        # Get all trial directories
        trial_dirs = sorted(experiments_dir.glob("trial_*"))
        print(f"📁 Found {len(trial_dirs)} trial directories")

        if not trial_dirs:
            return

        # Determine which to keep
        trials = self.get_study_trials()
        if keep_top_n:
            trials.sort(key=lambda x: x['sharpe'], reverse=True)
            keep_numbers = set(t['number'] for t in trials[:keep_top_n])
            print(f"🎯 Keeping top {keep_top_n} trials by Sharpe ratio")
        else:
            cutoff_date = datetime.now().timestamp() - (keep_days * 24 * 3600)
            keep_numbers = set(
                t['number'] for t in trials
                if datetime.fromisoformat(t['end_time']).timestamp() > cutoff_date
            ) if trials else set()
            print(f"🎯 Keeping trials from last {keep_days} days")

        # Delete old trials
        deleted = 0
        for trial_dir in trial_dirs:
            trial_num = int(trial_dir.name.split('_')[1])
            if trial_num not in keep_numbers:
                if dry_run:
                    print(f"  Would delete: {trial_dir.name}")
                else:
                    shutil.rmtree(trial_dir)
                    print(f"  ❌ Deleted: {trial_dir.name}")
                deleted += 1

        print(f"\n{'✅' if not dry_run else '📋'} {'Deleted' if not dry_run else 'Would delete'} {deleted} trial directories")

    def clean_old_logs(self, keep_days: int = 7, dry_run: bool = False):
        """
        Clean up old log files.

        Args:
            keep_days: Keep logs from last N days
            dry_run: Don't actually delete, just show what would be deleted
        """
        print(f"\n{'='*70}")
        print(f"CLEANING OLD LOGS" + (" (DRY RUN)" if dry_run else ""))
        print(f"{'='*70}\n")

        logs_dir = self.base_dir / "logs"
        if not logs_dir.exists():
            print("⚠️  No logs directory found")
            return

        cutoff_time = datetime.now().timestamp() - (keep_days * 24 * 3600)

        deleted = 0
        total_size = 0

        for log_file in logs_dir.glob("*.log"):
            # Skip broken symlinks
            if not log_file.exists():
                print(f"  ⚠️  Skipping broken symlink: {log_file.name}")
                continue

            try:
                file_mtime = log_file.stat().st_mtime
                if file_mtime < cutoff_time:
                    size = log_file.stat().st_size
                    total_size += size

                    if dry_run:
                        print(f"  Would delete: {log_file.name} ({size / 1024 / 1024:.1f} MB)")
                    else:
                        log_file.unlink()
                        print(f"  ❌ Deleted: {log_file.name} ({size / 1024 / 1024:.1f} MB)")
                    deleted += 1
            except (OSError, FileNotFoundError) as e:
                print(f"  ⚠️  Skipping {log_file.name}: {e}")
                continue

        print(f"\n{'✅' if not dry_run else '📋'} {'Deleted' if not dry_run else 'Would delete'} {deleted} log files ({total_size / 1024 / 1024:.1f} MB)")

    def print_summary(self):
        """Print summary of archived trials."""
        print(f"\n{'='*70}")
        print(f"TRIAL ARCHIVE SUMMARY")
        print(f"{'='*70}\n")

        stats = self.registry['stats']
        print(f"Total Archived: {stats['total_archived']}")
        print(f"Last Updated: {self.registry['last_updated']}")

        if stats['by_grade']:
            print(f"\nBy Grade:")
            for grade in ['S', 'A', 'B', 'C', 'D', 'F']:
                count = stats['by_grade'].get(grade, 0)
                if count > 0:
                    print(f"  {grade}: {count}")

        # Show top 5 trials
        top_trials = self.list_archived_trials(limit=5)
        if top_trials:
            print(f"\nTop 5 Trials:")
            for i, trial in enumerate(top_trials, 1):
                print(f"  {i}. {trial['vin']} (Sharpe: {trial['sharpe']:.4f})")


def main():
    """CLI interface for trial management."""
    import argparse

    parser = argparse.ArgumentParser(description="Manage training trials")
    parser.add_argument('--archive', action='store_true', help='Archive top trials')
    parser.add_argument('--clean-trials', action='store_true', help='Clean old trial directories')
    parser.add_argument('--clean-logs', action='store_true', help='Clean old log files')
    parser.add_argument('--list', action='store_true', help='List archived trials')
    parser.add_argument('--study-name', type=str, help='Specific study to process')
    parser.add_argument('--top-percent', type=int, default=10, help='Top percentage to archive (default: 10)')
    parser.add_argument('--keep-days', type=int, default=7, help='Days to keep logs/trials (default: 7)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted without deleting')

    args = parser.parse_args()

    manager = TrialManager()

    if args.archive:
        manager.archive_top_trials(
            study_name=args.study_name,
            top_percentile=args.top_percent
        )

    if args.clean_trials:
        manager.clean_old_trials(
            keep_days=args.keep_days,
            dry_run=args.dry_run
        )

    if args.clean_logs:
        manager.clean_old_logs(
            keep_days=args.keep_days,
            dry_run=args.dry_run
        )

    if args.list or not any([args.archive, args.clean_trials, args.clean_logs]):
        manager.print_summary()


if __name__ == "__main__":
    main()
