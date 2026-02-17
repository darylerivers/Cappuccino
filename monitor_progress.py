#!/usr/bin/env python3
"""Live progress monitor for running trials."""
import sqlite3
import time
from datetime import datetime
import sys

try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm", "-q"])
    from tqdm import tqdm

DB = "databases/optuna_cappuccino.db"
STUDY = "cappuccino_auto_20260214_2059"

def get_trials():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    
    c.execute("SELECT study_id FROM studies WHERE study_name = ?", (STUDY,))
    sid = c.fetchone()
    if not sid:
        return [], {}
    
    # Get running trials
    c.execute("""
        SELECT t.number, t.datetime_start
        FROM trials t
        WHERE t.study_id = ? AND t.state = 'RUNNING'
    """, (sid[0],))
    running = c.fetchall()
    
    # Get stats
    c.execute("""
        SELECT 
            COUNT(CASE WHEN state='COMPLETE' THEN 1 END),
            COUNT(CASE WHEN state='RUNNING' THEN 1 END),
            AVG(CASE WHEN state='COMPLETE' THEN tv.value END),
            MAX(tv.value)
        FROM trials t
        LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE t.study_id = ?
    """, (sid[0],))
    
    stats = c.fetchone()
    conn.close()
    
    return running, {
        'completed': stats[0],
        'running': stats[1],
        'avg': stats[2] or 0,
        'max': stats[3] or 0
    }

print("\n" + "="*70)
print("  CAPPUCCINO TRAINING PROGRESS")
print("="*70 + "\n")

running, stats = get_trials()

print(f"📊 Completed: {stats['completed']} | Running: {stats['running']}")
print(f"   Avg Sharpe: {stats['avg']:.4f} | Max: {stats['max']:.4f}\n")

if not running:
    print("No trials currently running.\n")
else:
    print(f"🔄 Running Trials ({len(running)}):\n")
    
    for number, start in running:
        start_time = datetime.fromisoformat(start)
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        
        # Estimate 30-60 min per trial
        est_total = 45
        progress = min(100, (elapsed / est_total) * 100)
        
        # Create bar
        bar = tqdm(
            total=100,
            desc=f"  Trial #{number:3d}",
            initial=int(progress),
            bar_format='{desc} |{bar}| {percentage:3.0f}% [{postfix}]',
            leave=True,
            ncols=65
        )
        bar.set_postfix_str(f"{int(elapsed)}m / ~{est_total}m")
        bar.close()

print("\n" + "="*70)
print("Note: Progress based on estimated 45min per trial")
print("Refresh: python monitor_progress.py")
print("="*70 + "\n")
