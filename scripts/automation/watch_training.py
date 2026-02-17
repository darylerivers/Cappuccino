#!/usr/bin/env python3
"""
Live Training Viewer - Real-time monitoring of training progress
"""
import os
import sys
import time
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path
from path_detector import PathDetector

def get_gpu_info():
    """Get GPU memory and utilization."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            return {
                'index': parts[0],
                'name': parts[1],
                'memory_used': int(parts[2]),
                'memory_total': int(parts[3]),
                'utilization': int(parts[4]),
                'temperature': int(parts[5])
            }
    except:
        pass
    return None

def get_training_status():
    """Check if training process is running."""
    try:
        result = subprocess.run(
            ['pgrep', '-f', '1_optimize_unified.py'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            return [pid for pid in pids if pid]
    except:
        pass
    return []

def get_trial_stats(db_path):
    """Get trial statistics from Optuna database."""
    try:
        conn = sqlite3.connect(db_path, timeout=2)
        cursor = conn.cursor()

        # Count by state
        cursor.execute("""
            SELECT state, COUNT(*) as count
            FROM trials
            GROUP BY state
        """)
        state_counts = {row[0]: row[1] for row in cursor.fetchall()}

        # Get current running trial
        cursor.execute("""
            SELECT number, datetime_start
            FROM trials
            WHERE state = 'RUNNING'
            ORDER BY number DESC
            LIMIT 1
        """)
        running = cursor.fetchone()

        # Get recent completed trials
        cursor.execute("""
            SELECT t.number, tv.value, t.datetime_complete
            FROM trials t
            LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.state = 'COMPLETE'
            ORDER BY t.number DESC
            LIMIT 5
        """)
        recent = cursor.fetchall()

        conn.close()

        return {
            'state_counts': state_counts,
            'running': running,
            'recent': recent
        }
    except Exception as e:
        return None

def get_log_tail(log_file, lines=25):
    """Get last N lines from log file."""
    try:
        result = subprocess.run(
            ['tail', f'-{lines}', log_file],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            return result.stdout
    except:
        pass
    return ""

def format_duration(seconds):
    """Format seconds to human readable duration."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds/60)}m {int(seconds%60)}s"
    else:
        hours = int(seconds / 3600)
        mins = int((seconds % 3600) / 60)
        return f"{hours}h {mins}m"

def main():
    # Auto-detect paths
    detector = PathDetector()
    db_path = detector.find_optuna_db()
    log_dir = detector.find_log_dir()
    log_file = f"{log_dir}/training_working.log"

    print("\033[?25l", end="")  # Hide cursor

    try:
        while True:
            # Clear screen
            os.system('clear')

            # Header
            print("=" * 120)
            print("LIVE TRAINING VIEWER - DRL Agent Optimization")
            print("=" * 120)
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 120)

            # GPU Status
            gpu = get_gpu_info()
            if gpu:
                mem_pct = (gpu['memory_used'] / gpu['memory_total']) * 100
                print(f"\nGPU Status: {gpu['name']}")
                print(f"  VRAM:        {gpu['memory_used']:>5} MB / {gpu['memory_total']:>5} MB  ({mem_pct:>5.1f}%)")
                print(f"  Utilization: {gpu['utilization']:>3}%")
                print(f"  Temperature: {gpu['temperature']:>3}°C")

                # Visual bar for VRAM
                bar_width = 50
                filled = int(bar_width * mem_pct / 100)
                bar = "█" * filled + "░" * (bar_width - filled)
                print(f"  VRAM: [{bar}]")
            else:
                print("\nGPU Status: Not available")

            # Training Process
            pids = get_training_status()
            if pids:
                print(f"\nTraining Process: RUNNING (PIDs: {', '.join(pids)})")
            else:
                print("\n⚠️  Training Process: NOT RUNNING")

            # Trial Statistics
            stats = get_trial_stats(db_path)
            if stats:
                print("\nTrial Statistics:")
                counts = stats['state_counts']
                total = sum(counts.values())
                print(f"  Total Trials:    {total}")
                print(f"  ✓ Complete:      {counts.get('COMPLETE', 0)}")
                print(f"  ⏳ Running:       {counts.get('RUNNING', 0)}")
                print(f"  ✗ Failed:        {counts.get('FAIL', 0)}")
                print(f"  ⊗ Pruned:        {counts.get('PRUNED', 0)}")

                # Current running trial
                if stats['running']:
                    trial_num, start_time = stats['running']
                    start_dt = datetime.fromisoformat(start_time)
                    duration = (datetime.now() - start_dt).total_seconds()
                    print(f"\nCurrent Trial: #{trial_num}")
                    print(f"  Started:  {start_dt.strftime('%H:%M:%S')}")
                    print(f"  Duration: {format_duration(duration)}")

                # Recent completed trials
                if stats['recent']:
                    print("\nRecent Completed Trials:")
                    for trial_num, value, completed in stats['recent'][:3]:
                        if completed:
                            comp_time = datetime.fromisoformat(completed).strftime('%H:%M:%S')
                            val_str = f"{value:.6f}" if value is not None else "N/A"
                            print(f"  Trial #{trial_num:<3} Value: {val_str}  Completed: {comp_time}")
            else:
                print("\nTrial Statistics: Database not available")

            # Log output
            print("\n" + "=" * 120)
            print("Recent Log Output (last 25 lines):")
            print("=" * 120)
            log_output = get_log_tail(log_file)
            if log_output:
                # Print log with line wrapping
                for line in log_output.split('\n')[-25:]:
                    if len(line) > 118:
                        print(line[:118] + "…")
                    else:
                        print(line)
            else:
                print("No log output available")

            print("=" * 120)
            print("Press Ctrl+C to exit | Refreshing every 5 seconds...")
            print("=" * 120)

            # Sleep
            time.sleep(5)

    except KeyboardInterrupt:
        print("\033[?25h")  # Show cursor
        print("\n\nStopped watching training.")
        sys.exit(0)
    except Exception as e:
        print("\033[?25h")  # Show cursor
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
