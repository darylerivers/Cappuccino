#!/usr/bin/env python3
"""
Auto-Fix Crashed Trials - Monitors for crashed paper traders and redeploys them.

This script runs in the background and:
1. Checks for trials marked as 'deployed' in pipeline DB
2. Checks if their processes are actually running
3. Redeploys any crashed trials automatically
"""

import time
import sqlite3
import psutil
import logging
from pathlib import Path
from deploy_v2 import PaperTradingDeployer

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/auto_repair.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def get_deployed_trials():
    """Get list of deployed trials from pipeline DB."""
    try:
        conn = sqlite3.connect('pipeline_v2.db', timeout=5)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT trial_number
            FROM trials
            WHERE status = 'deployed'
            ORDER BY trial_number
        """)

        trials = [{'trial_number': row[0], 'pid': None} for row in cursor.fetchall()]
        conn.close()

        # Find PIDs by checking running processes
        for trial in trials:
            trial_num = trial['trial_number']
            trial['pid'] = find_trial_pid(trial_num)

        return trials
    except Exception as e:
        logger.error(f"Failed to query pipeline DB: {e}")
        return []


def find_trial_pid(trial_num):
    """Find the PID of a running trial by checking processes."""
    try:
        for proc in psutil.process_iter(['pid', 'cmdline']):
            cmdline = proc.info.get('cmdline', [])
            if cmdline and 'paper_trader_alpaca_polling.py' in ' '.join(cmdline):
                if f'trial_{trial_num}_' in ' '.join(cmdline):
                    return proc.info['pid']
    except Exception:
        pass
    return None


def is_process_running(pid):
    """Check if a process is running."""
    if not pid:
        return False
    try:
        process = psutil.Process(pid)
        return process.is_running() and 'paper_trader' in ' '.join(process.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def check_and_repair():
    """Check for crashed trials and redeploy them."""
    deployed = get_deployed_trials()
    deployer = PaperTradingDeployer()

    for trial in deployed:
        trial_num = trial['trial_number']
        pid = trial['pid']

        if not pid or not is_process_running(pid):
            logger.warning(f"Trial {trial_num} is deployed but not running (PID: {pid}) - redeploying...")

            try:
                result = deployer.deploy(trial_num)
                if result['success']:
                    new_pid = result['process_id']
                    logger.info(f"✓ Trial {trial_num} redeployed successfully with PID {new_pid}")
                else:
                    logger.error(f"✗ Failed to redeploy trial {trial_num}: {result.get('error')}")

            except Exception as e:
                logger.error(f"Error redeploying trial {trial_num}: {e}", exc_info=True)


def main():
    logger.info("Starting auto-repair service...")
    logger.info("Checking for crashed trials every 60 seconds")

    while True:
        try:
            check_and_repair()
        except Exception as e:
            logger.error(f"Error in check cycle: {e}", exc_info=True)

        time.sleep(60)


if __name__ == '__main__':
    main()
