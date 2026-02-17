#!/usr/bin/env python3
"""
Paper Trading Viewer - Monitor deployed trials in real-time
"""
import os
import sys
import time
import sqlite3
import subprocess
import re
from datetime import datetime
from pathlib import Path
from path_detector import PathDetector

def get_running_traders():
    """Get list of running paper trading processes."""
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True,
            timeout=2
        )

        traders = []
        for line in result.stdout.split('\n'):
            if 'paper_trader_alpaca' in line and 'grep' not in line:
                # Extract trial number from model-dir argument
                match = re.search(r'trial_(\d+)_', line)
                if match:
                    trial_num = int(match.group(1))
                    # Extract PID
                    pid = int(line.split()[1])
                    traders.append({'trial': trial_num, 'pid': pid})

        return traders
    except Exception as e:
        return []

def get_deployed_trials():
    """Get list of deployed trials from pipeline DB."""
    try:
        # Auto-detect pipeline database
        detector = PathDetector()
        db_path = detector.find_pipeline_db()

        conn = sqlite3.connect(db_path, timeout=2)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT trial_number, status, updated_at
            FROM trials
            WHERE status = 'deployed'
            ORDER BY trial_number
        """)

        trials = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return trials
    except:
        return []

def parse_log_for_metrics(log_file):
    """Extract trading metrics from log file."""
    metrics = {
        'status': 'Unknown',
        'last_update': 'N/A',
        'positions': 'N/A',
        'portfolio_value': 'N/A',
        'error': None
    }

    try:
        # Read last 100 lines
        result = subprocess.run(
            ['tail', '-100', log_file],
            capture_output=True,
            text=True,
            timeout=2
        )

        lines = result.stdout.split('\n')

        # Check for errors
        for line in reversed(lines):
            if 'Error' in line and 'fetching' in line:
                metrics['error'] = line[:80]
                metrics['status'] = 'Network Error'
                break

        # Look for last polling timestamp
        for line in reversed(lines):
            match = re.search(r'\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})', line)
            if match:
                metrics['last_update'] = match.group(1)
                break

        # Look for trading decisions (cash= lines)
        for line in reversed(lines):
            if 'cash=' in line and 'total=' in line:
                cash_match = re.search(r'cash=(\d+\.\d+)', line)
                total_match = re.search(r'total=(\d+\.\d+)', line)
                if cash_match and total_match:
                    cash = float(cash_match.group(1))
                    total = float(total_match.group(1))
                    metrics['portfolio_value'] = f"cash=${cash:.2f} total=${total:.2f}"
                    metrics['status'] = 'Trading' if cash != 1000.0 else 'Active'
                break

        # Check if waiting for bars
        for line in reversed(lines[-10:]):
            if 'No new bars' in line or 'waiting for next' in line:
                if metrics['status'] not in ['Network Error', 'Trading']:
                    metrics['status'] = 'Waiting'
                break

    except:
        pass

    return metrics

def main():
    print("\033[?25l", end="")  # Hide cursor

    # Auto-detect paths once at startup
    detector = PathDetector()
    log_dir = detector.find_log_dir()

    try:
        while True:
            os.system('clear')

            # Header
            print("=" * 120)
            print("PAPER TRADING MONITOR - Live Deployed Trials")
            print("=" * 120)
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 120)

            # Get data
            deployed = get_deployed_trials()
            running = get_running_traders()
            running_trials = {t['trial']: t['pid'] for t in running}

            # Summary
            print(f"\nDeployed Trials: {len(deployed)}")
            print(f"Active Traders:  {len(running)}")
            print(f"Crashed/Stopped: {len(deployed) - len(running)}")

            # Detailed status
            print("\n" + "=" * 120)
            print(f"{'Trial':<8} {'PID':<10} {'Status':<12} {'Last Update':<18} {'Info':<60}")
            print("=" * 120)

            for trial in deployed:
                trial_num = trial['trial_number']
                log_file = f"{log_dir}/paper_trading_trial{trial_num}.log"

                # Check if running
                pid = running_trials.get(trial_num, None)
                pid_str = str(pid) if pid else "stopped"

                # Parse log
                if Path(log_file).exists():
                    metrics = parse_log_for_metrics(log_file)
                    status = metrics['status']
                    last_update = metrics['last_update']
                    info = metrics['error'] if metrics['error'] else metrics['portfolio_value']

                    # Color coding
                    if status == 'Active' and pid:
                        status_display = f"✓ {status}"
                    elif status == 'Error':
                        status_display = f"✗ {status}"
                    elif status == 'Waiting':
                        status_display = f"⏳ {status}"
                    else:
                        status_display = status
                else:
                    status_display = "No log"
                    last_update = "N/A"
                    info = "Log file not found"

                print(f"{trial_num:<8} {pid_str:<10} {status_display:<12} {last_update:<18} {info[:60]:<60}")

            print("=" * 120)

            # Active trader details
            if running:
                print("\nActive Traders Detail:")
                for trader in running:
                    trial_num = trader['trial']
                    log_file = f"{log_dir}/paper_trading_trial{trial_num}.log"

                    print(f"\nTrial {trial_num} (PID {trader['pid']}):")

                    # Show last few log lines
                    try:
                        result = subprocess.run(
                            ['tail', '-5', log_file],
                            capture_output=True,
                            text=True,
                            timeout=1
                        )
                        for line in result.stdout.split('\n')[-3:]:
                            if line.strip():
                                print(f"  {line[:115]}")
                    except:
                        print("  Unable to read log")
            else:
                print("\n⚠️  No active paper traders running")
                print("   Deployed trials may have crashed on startup")
                print("   Check logs/paper_trading_trial*.log for errors")

            print("\n" + "=" * 120)
            print("Press Ctrl+C to exit | Refreshing every 10 seconds...")
            print("=" * 120)

            time.sleep(10)

    except KeyboardInterrupt:
        print("\033[?25h")  # Show cursor
        print("\n\nStopped monitoring.")
        sys.exit(0)
    except Exception as e:
        print("\033[?25h")
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
