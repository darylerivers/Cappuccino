#!/usr/bin/env python3
"""
Redundancy Manager - Handles backups, atomic writes, and failover for critical system components.

Features:
- Database backups with SQLite backup API
- Atomic state file writes with automatic backups
- Ensemble multi-location redundancy
- Health monitoring utilities
"""

import json
import logging
import os
import shutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DatabaseBackupManager:
    """Manages SQLite database backups with rotation."""

    def __init__(
        self,
        db_path: str = "databases/optuna_cappuccino.db",
        backup_dir: str = "databases/backups",
        max_backups: int = 24,  # Keep 24 hourly backups (1 day)
    ):
        self.db_path = Path(db_path)
        self.backup_dir = Path(backup_dir)
        self.max_backups = max_backups
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def create_backup(self) -> Optional[Path]:
        """Create a timestamped backup using SQLite's backup API.

        Returns:
            Path to backup file, or None if failed
        """
        if not self.db_path.exists():
            logger.warning(f"Database not found: {self.db_path}")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"optuna_cappuccino_{timestamp}.db"

        try:
            # Use SQLite backup API for consistency (handles active connections)
            src_conn = sqlite3.connect(str(self.db_path))
            dst_conn = sqlite3.connect(str(backup_path))

            src_conn.backup(dst_conn)

            src_conn.close()
            dst_conn.close()

            # Verify backup integrity
            if self._verify_backup(backup_path):
                logger.info(f"Database backup created: {backup_path}")
                self._cleanup_old_backups()
                return backup_path
            else:
                backup_path.unlink()
                logger.error("Backup verification failed, removed corrupt backup")
                return None

        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            if backup_path.exists():
                backup_path.unlink()
            return None

    def _verify_backup(self, backup_path: Path) -> bool:
        """Verify backup integrity with PRAGMA integrity_check."""
        try:
            conn = sqlite3.connect(str(backup_path))
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check;")
            result = cursor.fetchone()
            conn.close()
            return result[0] == "ok"
        except Exception as e:
            logger.error(f"Backup verification error: {e}")
            return False

    def _cleanup_old_backups(self):
        """Remove old backups, keeping only max_backups most recent."""
        backups = sorted(
            self.backup_dir.glob("optuna_cappuccino_*.db"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        for old_backup in backups[self.max_backups:]:
            try:
                old_backup.unlink()
                logger.info(f"Removed old backup: {old_backup.name}")
            except Exception as e:
                logger.warning(f"Could not remove old backup {old_backup}: {e}")

    def restore_latest(self) -> bool:
        """Restore from the most recent valid backup.

        Returns:
            True if restoration successful
        """
        backups = sorted(
            self.backup_dir.glob("optuna_cappuccino_*.db"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        for backup in backups:
            if self._verify_backup(backup):
                try:
                    # Create backup of current (possibly corrupt) database
                    if self.db_path.exists():
                        corrupt_path = self.db_path.with_suffix('.db.corrupt')
                        shutil.move(str(self.db_path), str(corrupt_path))

                    # Restore from backup
                    shutil.copy2(str(backup), str(self.db_path))
                    logger.info(f"Database restored from: {backup.name}")
                    return True
                except Exception as e:
                    logger.error(f"Restoration failed: {e}")
                    continue

        logger.error("No valid backups found for restoration")
        return False

    def get_backup_stats(self) -> Dict[str, Any]:
        """Get statistics about current backups."""
        backups = list(self.backup_dir.glob("optuna_cappuccino_*.db"))

        if not backups:
            return {"count": 0, "oldest": None, "newest": None, "total_size_mb": 0}

        sorted_backups = sorted(backups, key=lambda p: p.stat().st_mtime)
        total_size = sum(b.stat().st_size for b in backups)

        return {
            "count": len(backups),
            "oldest": sorted_backups[0].name,
            "newest": sorted_backups[-1].name,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }


class AtomicStateManager:
    """Manages atomic writes for JSON state files with automatic backups."""

    def __init__(self, backup_count: int = 3):
        self.backup_count = backup_count

    def save(self, filepath: str, data: Dict[str, Any]) -> bool:
        """Atomically save state with automatic backup.

        Process:
        1. Write to temp file
        2. Verify temp file is valid JSON
        3. Backup current file (if exists)
        4. Atomic rename temp -> target

        Returns:
            True if save successful
        """
        path = Path(filepath)
        temp_path = path.with_suffix('.json.tmp')
        backup_path = path.with_suffix('.json.bak')

        try:
            # Write to temp file
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            # Verify temp file
            with open(temp_path, 'r') as f:
                json.load(f)  # Will raise if invalid

            # Rotate backups if file exists
            if path.exists():
                self._rotate_backups(path)

            # Atomic rename (on same filesystem)
            temp_path.rename(path)
            return True

        except Exception as e:
            logger.error(f"Failed to save state {filepath}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            return False

    def load(self, filepath: str, default: Dict = None) -> Dict[str, Any]:
        """Load state with automatic fallback to backup.

        Returns:
            State dict, or default if no valid state found
        """
        path = Path(filepath)
        backup_path = path.with_suffix('.json.bak')

        # Try primary file
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Primary state file corrupt: {e}")

        # Try backup
        if backup_path.exists():
            try:
                with open(backup_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"Restored state from backup: {backup_path}")
                # Restore primary from backup
                shutil.copy2(str(backup_path), str(path))
                return data
            except Exception as e:
                logger.warning(f"Backup state file also corrupt: {e}")

        # Try numbered backups
        for i in range(1, self.backup_count + 1):
            numbered_backup = path.with_suffix(f'.json.bak.{i}')
            if numbered_backup.exists():
                try:
                    with open(numbered_backup, 'r') as f:
                        data = json.load(f)
                    logger.info(f"Restored state from backup: {numbered_backup}")
                    return data
                except:
                    continue

        logger.warning(f"No valid state found for {filepath}, using default")
        return default if default is not None else {}

    def _rotate_backups(self, path: Path):
        """Rotate backup files: .bak -> .bak.1 -> .bak.2 -> ..."""
        # Remove oldest backup
        oldest = path.with_suffix(f'.json.bak.{self.backup_count}')
        if oldest.exists():
            oldest.unlink()

        # Rotate existing backups
        for i in range(self.backup_count - 1, 0, -1):
            current = path.with_suffix(f'.json.bak.{i}')
            next_backup = path.with_suffix(f'.json.bak.{i + 1}')
            if current.exists():
                current.rename(next_backup)

        # Move .bak to .bak.1
        backup = path.with_suffix('.json.bak')
        if backup.exists():
            backup.rename(path.with_suffix('.json.bak.1'))

        # Current file becomes .bak
        shutil.copy2(str(path), str(backup))


class EnsembleRedundancyManager:
    """Manages redundant storage of ensemble models."""

    PRIMARY_PATH = Path("train_results/ensemble")
    BACKUP_PATHS = [
        Path("train_results/ensemble_backup"),
        Path("deployments/ensemble_snapshot"),
    ]

    def __init__(self):
        self.all_paths = [self.PRIMARY_PATH] + self.BACKUP_PATHS

    def save_redundant(self, manifest: Dict, actor_paths: List[str]) -> int:
        """Save ensemble to multiple locations.

        Returns:
            Number of successful saves
        """
        success_count = 0

        for save_path in self.all_paths:
            try:
                save_path.mkdir(parents=True, exist_ok=True)

                # Save manifest
                manifest_path = save_path / "ensemble_manifest.json"
                with open(manifest_path, 'w') as f:
                    json.dump(manifest, f, indent=2)

                # Copy actor files
                for actor_path in actor_paths:
                    src = Path(actor_path)
                    if src.exists():
                        # Preserve directory structure
                        rel_path = src.name
                        dst = save_path / rel_path
                        shutil.copy2(str(src), str(dst))

                success_count += 1
                logger.info(f"Ensemble saved to: {save_path}")

            except Exception as e:
                logger.warning(f"Failed to save ensemble to {save_path}: {e}")

        return success_count

    def load_best_available(self) -> Optional[Dict]:
        """Load ensemble from first available valid location.

        Returns:
            Manifest dict, or None if no valid ensemble found
        """
        for load_path in self.all_paths:
            manifest_path = load_path / "ensemble_manifest.json"
            if manifest_path.exists():
                try:
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)

                    # Verify at least some models exist
                    if 'actor_paths' in manifest:
                        valid_models = sum(
                            1 for p in manifest['actor_paths']
                            if Path(p).exists()
                        )
                        if valid_models > 0:
                            logger.info(f"Loaded ensemble from: {load_path}")
                            return manifest

                except Exception as e:
                    logger.warning(f"Failed to load ensemble from {load_path}: {e}")
                    continue

        logger.error("No valid ensemble found in any location")
        return None

    def sync_backups(self) -> bool:
        """Sync primary ensemble to all backup locations.

        Returns:
            True if at least one backup successful
        """
        if not (self.PRIMARY_PATH / "ensemble_manifest.json").exists():
            logger.warning("No primary ensemble to sync")
            return False

        try:
            with open(self.PRIMARY_PATH / "ensemble_manifest.json", 'r') as f:
                manifest = json.load(f)

            actor_paths = manifest.get('actor_paths', [])

            success = False
            for backup_path in self.BACKUP_PATHS:
                try:
                    backup_path.mkdir(parents=True, exist_ok=True)

                    # Copy manifest
                    shutil.copy2(
                        str(self.PRIMARY_PATH / "ensemble_manifest.json"),
                        str(backup_path / "ensemble_manifest.json")
                    )

                    logger.info(f"Ensemble synced to: {backup_path}")
                    success = True
                except Exception as e:
                    logger.warning(f"Failed to sync to {backup_path}: {e}")

            return success

        except Exception as e:
            logger.error(f"Ensemble sync failed: {e}")
            return False


class HealthMonitor:
    """Monitors system health and component status."""

    def __init__(self):
        self.db_backup = DatabaseBackupManager()
        self.state_manager = AtomicStateManager()

    def get_full_status(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        import subprocess

        def check_process(pattern: str) -> bool:
            try:
                result = subprocess.run(
                    ["pgrep", "-f", pattern],
                    capture_output=True, text=True
                )
                return result.returncode == 0
            except:
                return False

        def get_process_count(pattern: str) -> int:
            try:
                result = subprocess.run(
                    ["pgrep", "-cf", pattern],
                    capture_output=True, text=True
                )
                return int(result.stdout.strip()) if result.returncode == 0 else 0
            except:
                return 0

        # Check all components
        status = {
            "timestamp": datetime.now().isoformat(),
            "components": {
                "paper_trader": check_process("paper_trader_alpaca"),
                "training_workers": get_process_count("1_optimize_unified"),
                "watchdog": check_process("system_watchdog"),
                "ai_advisor": check_process("ollama_autonomous_advisor"),
            },
            "database": {
                "exists": Path("databases/optuna_cappuccino.db").exists(),
                "size_mb": round(
                    Path("databases/optuna_cappuccino.db").stat().st_size / (1024 * 1024), 2
                ) if Path("databases/optuna_cappuccino.db").exists() else 0,
                "backups": self.db_backup.get_backup_stats(),
            },
            "ensemble": {
                "primary_exists": Path("train_results/ensemble/ensemble_manifest.json").exists(),
                "backup_exists": Path("train_results/ensemble_backup/ensemble_manifest.json").exists(),
            },
        }

        # Overall health
        status["healthy"] = (
            status["components"]["paper_trader"] and
            status["components"]["training_workers"] >= 1 and
            status["components"]["watchdog"] and
            status["database"]["exists"]
        )

        return status


# Convenience functions for direct use
_db_backup = None
_state_manager = None
_ensemble_manager = None


def get_db_backup_manager() -> DatabaseBackupManager:
    global _db_backup
    if _db_backup is None:
        _db_backup = DatabaseBackupManager()
    return _db_backup


def get_state_manager() -> AtomicStateManager:
    global _state_manager
    if _state_manager is None:
        _state_manager = AtomicStateManager()
    return _state_manager


def get_ensemble_manager() -> EnsembleRedundancyManager:
    global _ensemble_manager
    if _ensemble_manager is None:
        _ensemble_manager = EnsembleRedundancyManager()
    return _ensemble_manager


def backup_database() -> Optional[Path]:
    """Create database backup."""
    return get_db_backup_manager().create_backup()


def save_state(filepath: str, data: Dict) -> bool:
    """Atomically save state file."""
    return get_state_manager().save(filepath, data)


def load_state(filepath: str, default: Dict = None) -> Dict:
    """Load state file with fallback."""
    return get_state_manager().load(filepath, default)


def sync_ensemble_backups() -> bool:
    """Sync ensemble to backup locations."""
    return get_ensemble_manager().sync_backups()


def get_system_health() -> Dict:
    """Get full system health status."""
    return HealthMonitor().get_full_status()


if __name__ == "__main__":
    # Test/demo functionality
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("REDUNDANCY MANAGER TEST")
    print("=" * 60)

    # Test database backup
    print("\n1. Testing database backup...")
    backup_path = backup_database()
    if backup_path:
        print(f"   Backup created: {backup_path}")

    # Test state manager
    print("\n2. Testing atomic state manager...")
    test_state = {"test": "data", "timestamp": datetime.now().isoformat()}
    if save_state("deployments/test_state.json", test_state):
        print("   State saved successfully")
        loaded = load_state("deployments/test_state.json")
        print(f"   State loaded: {loaded}")

    # Test ensemble sync
    print("\n3. Testing ensemble sync...")
    if sync_ensemble_backups():
        print("   Ensemble synced to backups")

    # Get health status
    print("\n4. System health status:")
    health = get_system_health()
    print(f"   Overall healthy: {health['healthy']}")
    print(f"   Components: {health['components']}")
    print(f"   Database backups: {health['database']['backups']}")

    print("\n" + "=" * 60)
