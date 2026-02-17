#!/usr/bin/env python3
"""
Check if automation is ready to start (100+ trials completed).
Can be run manually or via cron.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Load configuration
from dotenv import load_dotenv
load_dotenv('.env.training')

ACTIVE_STUDY = os.getenv('ACTIVE_STUDY_NAME')
DB_PATH = "databases/optuna_cappuccino.db"
READY_FLAG = "deployments/.automation_ready"
MIN_TRIALS = 100


def get_trial_count(study_name: str) -> int:
    """Get completed trial count for study."""
    import sqlite3

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        query = """
        SELECT COUNT(*)
        FROM trials t
        JOIN studies s ON t.study_id = s.study_id
        WHERE s.study_name = ?
          AND t.state = 'COMPLETE'
        """

        cursor.execute(query, (study_name,))
        count = cursor.fetchone()[0]
        conn.close()

        return count
    except Exception as e:
        print(f"Error querying database: {e}")
        return 0


def is_automation_running() -> bool:
    """Check if automation is already running."""
    pid_files = [
        "deployments/auto_deployer.pid",
        "deployments/ensemble_updater.pid",
        "deployments/watchdog.pid"
    ]

    for pid_file in pid_files:
        if Path(pid_file).exists():
            try:
                with open(pid_file) as f:
                    pid = int(f.read().strip())

                # Check if process is running
                import subprocess
                result = subprocess.run(
                    ["ps", "-p", str(pid)],
                    capture_output=True
                )
                if result.returncode == 0:
                    return True
            except:
                pass

    return False


def main():
    print("=" * 60)
    print("AUTOMATION READINESS CHECK")
    print("=" * 60)
    print(f"Study: {ACTIVE_STUDY}")
    print(f"Minimum trials required: {MIN_TRIALS}")
    print("")

    # Get trial count
    trial_count = get_trial_count(ACTIVE_STUDY)
    print(f"Completed trials: {trial_count}")

    # Check if automation is already running
    automation_running = is_automation_running()

    if automation_running:
        print("Status: ✓ Automation is ALREADY RUNNING")
        print("")
        print("No action needed.")
        return 0

    # Check if ready
    if trial_count >= MIN_TRIALS:
        print(f"Status: ✅ READY for automation!")
        print("")

        # Create flag file
        flag_path = Path(READY_FLAG)
        if not flag_path.exists():
            flag_path.parent.mkdir(exist_ok=True)
            with open(flag_path, 'w') as f:
                f.write(f"Ready at: {datetime.now()}\n")
                f.write(f"Trial count: {trial_count}\n")
                f.write(f"Study: {ACTIVE_STUDY}\n")

            print("✓ Created ready flag: deployments/.automation_ready")

        print("")
        print("=" * 60)
        print("ACTION REQUIRED:")
        print("=" * 60)
        print("Run the following command to start automation:")
        print("")
        print("  ./start_automation.sh")
        print("")
        print("This will start:")
        print("  - Auto-model deployer (deploys best models)")
        print("  - Ensemble updater (syncs top 20 models)")
        print("  - System watchdog (monitors processes)")
        print("  - Performance monitor (tracks metrics)")
        print("")

        return 1  # Indicates ready but not started
    else:
        remaining = MIN_TRIALS - trial_count
        print(f"Status: ⏳ Not ready yet")
        print(f"Need {remaining} more trials")
        print("")

        # Estimate time
        # Assume 3 trials/hour with 3 workers
        hours_remaining = remaining / 9
        print(f"Estimated time: ~{hours_remaining:.1f} hours")
        print("")
        print("Check back later, or monitor via:")
        print("  python3 dashboard.py")

        return 2  # Indicates not ready yet


if __name__ == "__main__":
    sys.exit(main())
