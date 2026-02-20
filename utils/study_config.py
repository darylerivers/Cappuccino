"""Single source of truth for the active Optuna study name.

Priority order:
  1. ACTIVE_STUDY_NAME environment variable
  2. .current_study file in the project root
  3. Auto-detect from running worker process args

Usage (from any script in the project):
    from utils.study_config import get_current_study
    STUDY = get_current_study()
"""

import os
import subprocess
from pathlib import Path

# Project root is two levels up from utils/
_ROOT = Path(__file__).parent.parent


def get_current_study() -> str:
    """Return the active study name, checking multiple sources in order."""

    # 1. Environment variable — highest priority, easy to override per-shell
    env_study = os.getenv('ACTIVE_STUDY_NAME', '').strip()
    if env_study:
        return env_study

    # 2. .current_study file written by the automation pipeline
    study_file = _ROOT / '.current_study'
    if study_file.exists():
        name = study_file.read_text().strip()
        if name:
            return name

    # 3. Detect from running worker command lines
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=3)
        for line in result.stdout.splitlines():
            if '1_optimize_unified.py' in line and '--study-name' in line:
                parts = line.split('--study-name')
                if len(parts) > 1:
                    return parts[1].strip().split()[0]
    except Exception:
        pass

    raise RuntimeError(
        "Cannot determine the active study name.\n"
        "Fix one of:\n"
        "  • Set env var:  export ACTIVE_STUDY_NAME=<study>\n"
        f"  • Write file:   echo '<study>' > {study_file}\n"
        "  • Start workers with --study-name so auto-detect works."
    )


def set_current_study(name: str) -> None:
    """Write name to .current_study so all scripts pick it up automatically."""
    study_file = _ROOT / '.current_study'
    study_file.write_text(name.strip())
