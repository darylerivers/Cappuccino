#!/usr/bin/env python3
"""
Path Auto-Detection Utility
Automatically discovers database files, training directories, and model paths
instead of hardcoding them in individual scripts.

Usage:
    from path_detector import PathDetector

    detector = PathDetector()
    optuna_db = detector.find_optuna_db()
    pipeline_db = detector.find_pipeline_db()
    data_dir = detector.find_data_dir()
"""

import os
import glob
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class PathDetector:
    """Automatically detect common file paths in the project"""

    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize path detector

        Args:
            project_root: Root directory of project (defaults to script location)
        """
        if project_root is None:
            # Use current working directory as default
            self.project_root = Path.cwd()
        else:
            self.project_root = Path(project_root)

    def find_optuna_db(self, default: str = "/tmp/optuna_working.db") -> str:
        """
        Find Optuna database file

        Search order:
        1. /tmp/optuna_working.db (common location)
        2. databases/optuna*.db
        3. *.db in project root
        4. Return default if nothing found

        Returns:
            Path to Optuna database
        """
        # Check databases directory FIRST (preferred location with actual data)
        db_dir = self.project_root / "databases"
        if db_dir.exists():
            db_files = list(db_dir.glob("optuna*.db"))
            # Return first non-empty database file
            for db_file in sorted(db_files, key=lambda x: x.stat().st_size, reverse=True):
                if db_file.stat().st_size > 0:
                    logger.debug(f"Found Optuna DB in databases/: {db_file}")
                    return str(db_file)

        # Check common location (only if non-empty)
        common_path = "/tmp/optuna_working.db"
        if os.path.exists(common_path) and os.path.getsize(common_path) > 0:
            logger.debug(f"Found Optuna DB at common location: {common_path}")
            return common_path

        # Continue checking other locations
        db_dir = self.project_root / "databases"
        if db_dir.exists():
            optuna_dbs = list(db_dir.glob("optuna*.db"))
            if optuna_dbs:
                # Use most recently modified
                latest = max(optuna_dbs, key=lambda p: p.stat().st_mtime)
                logger.debug(f"Found Optuna DB in databases/: {latest}")
                return str(latest)

        # Check project root
        root_dbs = list(self.project_root.glob("optuna*.db"))
        if root_dbs:
            latest = max(root_dbs, key=lambda p: p.stat().st_mtime)
            logger.debug(f"Found Optuna DB in project root: {latest}")
            return str(latest)

        logger.warning(f"No Optuna DB found, using default: {default}")
        return default

    def find_pipeline_db(self, default: str = "pipeline_v2.db") -> str:
        """
        Find pipeline database file

        Search order:
        1. pipeline_v2.db in project root
        2. databases/pipeline*.db
        3. Return default if nothing found

        Returns:
            Path to pipeline database
        """
        # Check project root first
        root_path = self.project_root / "pipeline_v2.db"
        if root_path.exists():
            logger.debug(f"Found pipeline DB in root: {root_path}")
            return str(root_path)

        # Check databases directory
        db_dir = self.project_root / "databases"
        if db_dir.exists():
            pipeline_dbs = list(db_dir.glob("pipeline*.db"))
            if pipeline_dbs:
                latest = max(pipeline_dbs, key=lambda p: p.stat().st_mtime)
                logger.debug(f"Found pipeline DB in databases/: {latest}")
                return str(latest)

        # Check for any pipeline*.db in root
        root_dbs = list(self.project_root.glob("pipeline*.db"))
        if root_dbs:
            latest = max(root_dbs, key=lambda p: p.stat().st_mtime)
            logger.debug(f"Found pipeline DB in project root: {latest}")
            return str(latest)

        logger.warning(f"No pipeline DB found, using default: {default}")
        return default

    def find_data_dir(self, default: str = "databases") -> str:
        """
        Find training data directory

        Search order:
        1. databases/ directory
        2. data/ directory
        3. Return default if nothing found

        Returns:
            Path to data directory
        """
        # Check common locations
        for dirname in ["databases", "data", "training_data"]:
            dir_path = self.project_root / dirname
            if dir_path.exists() and dir_path.is_dir():
                logger.debug(f"Found data directory: {dir_path}")
                return str(dir_path)

        logger.warning(f"No data directory found, using default: {default}")
        return default

    def find_model_dirs(self) -> List[str]:
        """
        Find all model directories

        Returns:
            List of paths containing trained models
        """
        model_dirs = []

        # Common patterns for model directories
        patterns = [
            "models_*",
            "paper_trading_trial*",
            "trained_models",
            "checkpoints"
        ]

        for pattern in patterns:
            matches = list(self.project_root.glob(pattern))
            model_dirs.extend([str(p) for p in matches if p.is_dir()])

        # Also check databases directory
        db_dir = self.project_root / "databases"
        if db_dir.exists():
            for pattern in patterns:
                matches = list(db_dir.glob(pattern))
                model_dirs.extend([str(p) for p in matches if p.is_dir()])

        logger.debug(f"Found {len(model_dirs)} model directories")
        return model_dirs

    def find_log_dir(self, default: str = "logs") -> str:
        """
        Find logs directory

        Returns:
            Path to logs directory
        """
        log_path = self.project_root / "logs"
        if log_path.exists():
            return str(log_path)

        return default

    def find_config_file(self, config_name: str) -> Optional[str]:
        """
        Find configuration file

        Args:
            config_name: Name of config file (e.g., "pipeline_v2_config.json")

        Returns:
            Path to config file or None if not found
        """
        # Check config directory
        config_dir = self.project_root / "config"
        if config_dir.exists():
            config_path = config_dir / config_name
            if config_path.exists():
                logger.debug(f"Found config: {config_path}")
                return str(config_path)

        # Check project root
        root_config = self.project_root / config_name
        if root_config.exists():
            logger.debug(f"Found config in root: {root_config}")
            return str(root_config)

        logger.warning(f"Config file not found: {config_name}")
        return None

    def find_paper_trading_logs(self) -> List[str]:
        """
        Find all paper trading log files

        Returns:
            List of paper trading log file paths
        """
        log_dir = Path(self.find_log_dir())

        if not log_dir.exists():
            return []

        # Find all paper trading logs
        logs = list(log_dir.glob("paper_trader_trial_*.log"))
        return [str(p) for p in sorted(logs)]

    def get_all_paths(self) -> dict:
        """
        Get all detected paths as a dictionary

        Returns:
            Dictionary of detected paths
        """
        return {
            "project_root": str(self.project_root),
            "optuna_db": self.find_optuna_db(),
            "pipeline_db": self.find_pipeline_db(),
            "data_dir": self.find_data_dir(),
            "log_dir": self.find_log_dir(),
            "model_dirs": self.find_model_dirs(),
            "paper_trading_logs": self.find_paper_trading_logs()
        }


def main():
    """Demo/test the path detector"""
    import json

    detector = PathDetector()
    paths = detector.get_all_paths()

    print("=" * 60)
    print("Auto-Detected Paths")
    print("=" * 60)
    print(json.dumps(paths, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
