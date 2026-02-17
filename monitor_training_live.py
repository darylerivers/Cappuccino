#!/usr/bin/env python3
"""
Real-time Training Monitor
Displays live training progress with trial completion rate and best performance
"""

import sqlite3
import time
import os
from datetime import datetime, timedelta
from pathlib import Path

# Configuration
DB_PATH = "databases/optuna_cappuccino.db"
STUDY_NAME = os.getenv('ACTIVE_STUDY_NAME', 'cappuccino_auto_20260214_2059')
REFRESH_INTERVAL = 10  # seconds

def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name == 'posix' else 'cls')

def get_training_stats():
    """Get current training statistics"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Total trials
    cursor.execute('''
        SELECT COUNT(*), MAX(t.number)
        FROM trials t
        JOIN studies s ON t.study_id = s.study_id
        WHERE s.study_name = ?
    ''', (STUDY_NAME,))
    total_trials, max_number = cursor.fetchone()

    # Completed trials
    cursor.execute('''
        SELECT COUNT(*)
        FROM trials t
        JOIN studies s ON t.study_id = s.study_id
        WHERE s.study_name = ? AND t.state = 'COMPLETE'
    ''', (STUDY_NAME,))
    completed = cursor.fetchone()[0]

    # Failed trials
    cursor.execute('''
        SELECT COUNT(*)
        FROM trials t
        JOIN studies s ON t.study_id = s.study_id
        WHERE s.study_name = ? AND t.state = 'FAIL'
    ''', (STUDY_NAME,))
    failed = cursor.fetchone()[0]

    # Running trials
    cursor.execute('''
        SELECT COUNT(*)
        FROM trials t
        JOIN studies s ON t.study_id = s.study_id
        WHERE s.study_name = ? AND t.state = 'RUNNING'
    ''', (STUDY_NAME,))
    running = cursor.fetchone()[0]

    # Best trial
    cursor.execute('''
        SELECT t.number, tv.value, t.datetime_complete
        FROM trials t
        LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
        JOIN studies s ON t.study_id = s.study_id
        WHERE s.study_name = ? AND t.state = 'COMPLETE'
        ORDER BY tv.value DESC
        LIMIT 1
    ''', (STUDY_NAME,))
    best = cursor.fetchone()

    # Recent activity (last hour)
    one_hour_ago = (datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('''
        SELECT COUNT(*)
        FROM trials t
        JOIN studies s ON t.study_id = s.study_id
        WHERE s.study_name = ?
          AND t.state = 'COMPLETE'
          AND t.datetime_complete > ?
    ''', (STUDY_NAME, one_hour_ago))
    last_hour = cursor.fetchone()[0]

    # Recent activity (last 10 minutes)
    ten_min_ago = (datetime.now() - timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('''
        SELECT COUNT(*)
        FROM trials t
        JOIN studies s ON t.study_id = s.study_id
        WHERE s.study_name = ?
          AND t.state = 'COMPLETE'
          AND t.datetime_complete > ?
    ''', (STUDY_NAME, ten_min_ago))
    last_10min = cursor.fetchone()[0]

    # Latest trial
    cursor.execute('''
        SELECT t.number, t.datetime_complete, t.state
        FROM trials t
        JOIN studies s ON t.study_id = s.study_id
        WHERE s.study_name = ?
        ORDER BY t.datetime_start DESC
        LIMIT 1
    ''', (STUDY_NAME,))
    latest = cursor.fetchone()

    # Top 10 trials
    cursor.execute('''
        SELECT t.number, tv.value
        FROM trials t
        LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
        JOIN studies s ON t.study_id = s.study_id
        WHERE s.study_name = ? AND t.state = 'COMPLETE'
        ORDER BY tv.value DESC
        LIMIT 10
    ''', (STUDY_NAME,))
    top_10 = cursor.fetchall()

    conn.close()

    return {
        'total': total_trials,
        'max_number': max_number,
        'completed': completed,
        'failed': failed,
        'running': running,
        'best': best,
        'last_hour': last_hour,
        'last_10min': last_10min,
        'latest': latest,
        'top_10': top_10,
    }

def display_stats(stats):
    """Display formatted training statistics"""
    clear_screen()

    print("=" * 80)
    print(f"  CAPPUCCINO TRAINING MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    print(f"Study: {STUDY_NAME}")
    print()

    # Progress
    print("PROGRESS:")
    print(f"  Total Trials:     {stats['total']:,}")
    print(f"  Completed:        {stats['completed']:,} ({stats['completed']/max(stats['total'], 1)*100:.1f}%)")
    print(f"  Failed:           {stats['failed']:,}")
    print(f"  Running:          {stats['running']:,}")
    print()

    # Activity
    print("ACTIVITY:")
    print(f"  Last 10 minutes:  {stats['last_10min']} trials ({stats['last_10min'] * 6:.0f}/hour)")
    print(f"  Last hour:        {stats['last_hour']} trials")
    print()

    # Best trial
    if stats['best']:
        trial_num, sharpe, completed_at = stats['best']
        print("BEST TRIAL:")
        print(f"  Trial #{trial_num}")
        print(f"  Sharpe Ratio:     {sharpe:.4f}")
        print(f"  Completed:        {completed_at}")
    print()

    # Latest trial
    if stats['latest']:
        latest_num, latest_time, latest_state = stats['latest']
        print("LATEST TRIAL:")
        print(f"  Trial #{latest_num}")
        print(f"  State:            {latest_state}")
        if latest_time:
            print(f"  Completed:        {latest_time}")
    print()

    # Top 10
    if stats['top_10']:
        print("TOP 10 TRIALS:")
        for i, (trial_num, sharpe) in enumerate(stats['top_10'], 1):
            print(f"  {i:2d}. Trial #{trial_num:4d}: Sharpe = {sharpe:.4f}")

    print()
    print("=" * 80)
    print(f"Refreshing every {REFRESH_INTERVAL} seconds... (Ctrl+C to exit)")
    print("=" * 80)

def main():
    """Main monitoring loop"""
    print("Starting training monitor...")
    print(f"Study: {STUDY_NAME}")
    print(f"Database: {DB_PATH}")
    print()

    # Check if database exists
    if not Path(DB_PATH).exists():
        print(f"ERROR: Database not found: {DB_PATH}")
        return

    try:
        while True:
            stats = get_training_stats()
            display_stats(stats)
            time.sleep(REFRESH_INTERVAL)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
