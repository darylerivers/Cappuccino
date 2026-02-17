#!/usr/bin/env python3
"""
Archive Manager - Compress and archive old training trials to free disk space.

Features:
- Identifies trials not in current ensemble
- Compresses trials into tar.gz archives (batched)
- Moves to archive location on different filesystem
- Maintains manifest for restoration
- Safe deletion with verification

Usage:
    # Analyze what can be archived
    python archive_manager.py --analyze

    # Archive old trials (dry run)
    python archive_manager.py --archive --dry-run

    # Archive old trials (actual)
    python archive_manager.py --archive --archive-dir /data/cappuccino_archive

    # Restore specific trial
    python archive_manager.py --restore --trial 123
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class ArchiveManager:
    """Manages archival of old training trials."""

    def __init__(
        self,
        train_results_dir: str = "train_results/cwd_tests",
        ensemble_manifest: str = "train_results/ensemble/ensemble_manifest.json",
        archive_dir: str = "/data/cappuccino_archive",
        batch_size: int = 50,  # Trials per archive
    ):
        self.train_results_dir = Path(train_results_dir)
        self.ensemble_manifest_path = Path(ensemble_manifest)
        self.archive_dir = Path(archive_dir)
        self.batch_size = batch_size
        self.manifest_path = self.archive_dir / "archive_manifest.json"

    def get_ensemble_trials(self) -> Set[int]:
        """Get set of trial numbers currently in ensemble."""
        if not self.ensemble_manifest_path.exists():
            logger.warning(f"Ensemble manifest not found: {self.ensemble_manifest_path}")
            return set()

        with open(self.ensemble_manifest_path) as f:
            manifest = json.load(f)
        return set(manifest.get('trial_numbers', []))

    def get_all_trials(self) -> List[Tuple[int, Path]]:
        """Get all trial directories with their trial numbers."""
        trials = []
        for trial_dir in self.train_results_dir.glob("trial_*_1h"):
            try:
                trial_num = int(trial_dir.name.split('_')[1])
                trials.append((trial_num, trial_dir))
            except (ValueError, IndexError):
                logger.warning(f"Could not parse trial number: {trial_dir.name}")
        return sorted(trials, key=lambda x: x[0])

    def get_archivable_trials(self, min_age_days: int = 1) -> List[Tuple[int, Path]]:
        """Get trials that can be archived (not in ensemble, old enough)."""
        ensemble_trials = self.get_ensemble_trials()
        all_trials = self.get_all_trials()

        from datetime import timedelta
        min_age = datetime.now().timestamp() - (min_age_days * 86400)

        archivable = []
        for trial_num, trial_dir in all_trials:
            if trial_num in ensemble_trials:
                continue  # Keep ensemble trials

            # Check age
            try:
                mtime = trial_dir.stat().st_mtime
                if mtime < min_age:
                    archivable.append((trial_num, trial_dir))
            except OSError:
                continue

        return archivable

    def analyze(self) -> Dict:
        """Analyze what can be archived."""
        ensemble_trials = self.get_ensemble_trials()
        all_trials = self.get_all_trials()
        archivable = self.get_archivable_trials()

        # Calculate sizes
        total_size = 0
        archivable_size = 0
        ensemble_size = 0

        for trial_num, trial_dir in all_trials:
            try:
                size = sum(f.stat().st_size for f in trial_dir.rglob('*') if f.is_file())
                total_size += size
                if trial_num in ensemble_trials:
                    ensemble_size += size
                elif (trial_num, trial_dir) in archivable:
                    archivable_size += size
            except OSError:
                continue

        # For archivable, just sum them
        archivable_size = 0
        for trial_num, trial_dir in archivable:
            try:
                size = sum(f.stat().st_size for f in trial_dir.rglob('*') if f.is_file())
                archivable_size += size
            except OSError:
                continue

        return {
            "total_trials": len(all_trials),
            "ensemble_trials": len(ensemble_trials),
            "archivable_trials": len(archivable),
            "total_size_gb": round(total_size / (1024**3), 2),
            "ensemble_size_gb": round(ensemble_size / (1024**3), 2),
            "archivable_size_gb": round(archivable_size / (1024**3), 2),
            "estimated_compressed_gb": round(archivable_size * 0.7 / (1024**3), 2),
            "ensemble_trial_numbers": sorted(ensemble_trials),
        }

    def archive_trials(
        self,
        dry_run: bool = False,
        max_batches: int = None,
    ) -> Dict:
        """Archive trials in batches."""
        archivable = self.get_archivable_trials()

        if not archivable:
            logger.info("No trials to archive")
            return {"archived": 0, "batches": 0}

        # Create archive directory
        if not dry_run:
            self.archive_dir.mkdir(parents=True, exist_ok=True)

        # Load or create manifest
        manifest = self._load_manifest()

        # Group into batches
        batches = [
            archivable[i:i + self.batch_size]
            for i in range(0, len(archivable), self.batch_size)
        ]

        if max_batches:
            batches = batches[:max_batches]

        archived_count = 0
        space_freed = 0

        for batch_idx, batch in enumerate(batches):
            batch_name = f"trials_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{batch_idx:04d}"
            archive_path = self.archive_dir / f"{batch_name}.tar.gz"

            trial_nums = [t[0] for t in batch]
            trial_dirs = [t[1] for t in batch]

            logger.info(f"Batch {batch_idx + 1}/{len(batches)}: {len(batch)} trials ({min(trial_nums)}-{max(trial_nums)})")

            if dry_run:
                batch_size = sum(
                    sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
                    for _, d in batch
                )
                logger.info(f"  [DRY RUN] Would archive {batch_size / (1024**3):.2f} GB to {archive_path}")
                archived_count += len(batch)
                space_freed += batch_size
                continue

            # Create tar archive
            try:
                batch_size = self._create_archive(archive_path, trial_dirs)

                if batch_size > 0:
                    # Verify archive
                    if self._verify_archive(archive_path, trial_dirs):
                        # Record in manifest
                        manifest["batches"][batch_name] = {
                            "archive_path": str(archive_path),
                            "trial_numbers": trial_nums,
                            "trial_dirs": [str(d) for d in trial_dirs],
                            "created": datetime.now().isoformat(),
                            "original_size": batch_size,
                            "compressed_size": archive_path.stat().st_size,
                        }
                        self._save_manifest(manifest)

                        # Delete original directories
                        for trial_dir in trial_dirs:
                            shutil.rmtree(trial_dir)
                            logger.info(f"  Removed: {trial_dir.name}")

                        archived_count += len(batch)
                        space_freed += batch_size
                        logger.info(f"  Archived: {batch_size / (1024**3):.2f} GB -> {archive_path.stat().st_size / (1024**3):.2f} GB")
                    else:
                        logger.error(f"  Archive verification failed, keeping originals")
                        archive_path.unlink()
                else:
                    logger.error(f"  Archive creation failed")

            except Exception as e:
                logger.error(f"  Batch failed: {e}")
                if archive_path.exists():
                    archive_path.unlink()

        return {
            "archived": archived_count,
            "batches": len(batches),
            "space_freed_gb": round(space_freed / (1024**3), 2),
        }

    def _create_archive(self, archive_path: Path, trial_dirs: List[Path]) -> int:
        """Create tar.gz archive of trial directories."""
        total_size = 0

        try:
            with tarfile.open(archive_path, "w:gz", compresslevel=6) as tar:
                for trial_dir in trial_dirs:
                    if trial_dir.exists():
                        # Calculate size before adding
                        dir_size = sum(f.stat().st_size for f in trial_dir.rglob('*') if f.is_file())
                        total_size += dir_size

                        # Add to archive
                        tar.add(trial_dir, arcname=trial_dir.name)

            return total_size
        except Exception as e:
            logger.error(f"Archive creation error: {e}")
            return 0

    def _verify_archive(self, archive_path: Path, trial_dirs: List[Path]) -> bool:
        """Verify archive contains all expected directories."""
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                archived_names = set(m.name.split('/')[0] for m in tar.getmembers())
                expected_names = set(d.name for d in trial_dirs)

                if archived_names >= expected_names:
                    return True
                else:
                    missing = expected_names - archived_names
                    logger.error(f"Archive missing: {missing}")
                    return False
        except Exception as e:
            logger.error(f"Archive verification error: {e}")
            return False

    def restore_trial(self, trial_num: int) -> bool:
        """Restore a specific trial from archive."""
        manifest = self._load_manifest()

        # Find which batch contains this trial
        for batch_name, batch_info in manifest.get("batches", {}).items():
            if trial_num in batch_info.get("trial_numbers", []):
                archive_path = Path(batch_info["archive_path"])

                if not archive_path.exists():
                    logger.error(f"Archive not found: {archive_path}")
                    return False

                # Extract just this trial
                trial_dir_name = f"trial_{trial_num}_1h"

                try:
                    with tarfile.open(archive_path, "r:gz") as tar:
                        members = [m for m in tar.getmembers() if m.name.startswith(trial_dir_name)]
                        if not members:
                            logger.error(f"Trial {trial_num} not found in archive")
                            return False

                        tar.extractall(self.train_results_dir, members=members)
                        logger.info(f"Restored trial {trial_num} to {self.train_results_dir / trial_dir_name}")
                        return True

                except Exception as e:
                    logger.error(f"Restore failed: {e}")
                    return False

        logger.error(f"Trial {trial_num} not found in any archive")
        return False

    def _load_manifest(self) -> Dict:
        """Load archive manifest."""
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {"batches": {}, "created": datetime.now().isoformat()}

    def _save_manifest(self, manifest: Dict):
        """Save archive manifest."""
        manifest["updated"] = datetime.now().isoformat()
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Archive old training trials")
    parser.add_argument("--analyze", action="store_true", help="Analyze archivable trials")
    parser.add_argument("--archive", action="store_true", help="Archive old trials")
    parser.add_argument("--restore", action="store_true", help="Restore a trial")
    parser.add_argument("--trial", type=int, help="Trial number to restore")
    parser.add_argument("--archive-dir", type=str, default="/data/cappuccino_archive",
                        help="Archive directory (default: /data/cappuccino_archive)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be archived")
    parser.add_argument("--max-batches", type=int, help="Maximum batches to process")
    parser.add_argument("--batch-size", type=int, default=50, help="Trials per archive batch")

    args = parser.parse_args()

    manager = ArchiveManager(
        archive_dir=args.archive_dir,
        batch_size=args.batch_size,
    )

    if args.analyze:
        print("\n" + "=" * 60)
        print("ARCHIVE ANALYSIS")
        print("=" * 60)

        analysis = manager.analyze()

        print(f"\nTotal trials:        {analysis['total_trials']}")
        print(f"Ensemble trials:     {analysis['ensemble_trials']} (protected)")
        print(f"Archivable trials:   {analysis['archivable_trials']}")
        print(f"\nTotal size:          {analysis['total_size_gb']} GB")
        print(f"Ensemble size:       {analysis['ensemble_size_gb']} GB")
        print(f"Archivable size:     {analysis['archivable_size_gb']} GB")
        print(f"Est. compressed:     {analysis['estimated_compressed_gb']} GB")
        print(f"\nSpace savings:       ~{analysis['archivable_size_gb'] - analysis['estimated_compressed_gb']:.1f} GB")
        print(f"\nEnsemble trials: {analysis['ensemble_trial_numbers']}")
        print("=" * 60 + "\n")

    elif args.archive:
        print("\n" + "=" * 60)
        print("ARCHIVING TRIALS")
        print("=" * 60)
        print(f"Archive directory: {args.archive_dir}")
        print(f"Dry run: {args.dry_run}")
        print("=" * 60 + "\n")

        result = manager.archive_trials(
            dry_run=args.dry_run,
            max_batches=args.max_batches,
        )

        print(f"\nArchived: {result['archived']} trials in {result['batches']} batches")
        print(f"Space freed: {result.get('space_freed_gb', 'N/A')} GB")

    elif args.restore:
        if not args.trial:
            print("Error: --trial required for restore")
            return

        print(f"\nRestoring trial {args.trial}...")
        if manager.restore_trial(args.trial):
            print("Restore successful")
        else:
            print("Restore failed")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
