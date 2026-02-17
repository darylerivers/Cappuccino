#!/usr/bin/env python3
"""
Training Campaign Monitor

Shows real-time progress of all training studies.
"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

def monitor_training(db_path='databases/ensemble_ft_campaign.db'):
    """Monitor all training studies in the campaign."""

    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        print("Training hasn't started yet or database path is incorrect.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all studies
    cursor.execute("SELECT study_id, study_name FROM studies")
    studies = cursor.fetchall()

    if not studies:
        print("❌ No studies found in database")
        return

    print("\n" + "="*80)
    print("🎯 ENSEMBLE + FT-TRANSFORMER TRAINING CAMPAIGN")
    print("="*80 + "\n")

    total_trials = 0
    completed_trials = 0

    for study_id, study_name in studies:
        # Get study statistics
        cursor.execute("""
            SELECT COUNT(*),
                   SUM(CASE WHEN state = 'COMPLETE' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN state = 'RUNNING' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN state = 'FAIL' THEN 1 ELSE 0 END)
            FROM trials
            WHERE study_id = ?
        """, (study_id,))

        stats = cursor.fetchone()
        total, complete, running, failed = stats

        # Get best trial
        cursor.execute("""
            SELECT t.trial_id, tv.value
            FROM trials t
            JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.study_id = ? AND t.state = 'COMPLETE'
            ORDER BY tv.value DESC
            LIMIT 1
        """, (study_id,))

        best = cursor.fetchone()
        best_sharpe = best[1] if best else 0.0

        # Progress
        total_trials += total
        completed_trials += complete
        progress = (complete / 100.0 * 100) if complete > 0 else 0

        # Study type
        study_type = "🤖 FT" if "ft_transformer" in study_name.lower() else "📊 Ensemble"

        # Status color
        if running > 0:
            status = "🟢 RUNNING"
        elif complete >= 100:
            status = "✅ COMPLETE"
        elif failed > 0:
            status = "🟡 IN PROGRESS"
        else:
            status = "⚪ IDLE"

        print(f"{study_type} {study_name}")
        print(f"  Status:       {status}")
        print(f"  Progress:     {complete}/100 trials ({progress:.1f}%)")
        print(f"  Running:      {running}")
        print(f"  Failed:       {failed}")
        print(f"  Best Sharpe:  {best_sharpe:.4f}")

        # Progress bar
        bar_length = 40
        filled = int(bar_length * progress / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"  [{bar}] {progress:.1f}%")
        print()

    # Overall progress
    print("="*80)
    print(f"📈 OVERALL PROGRESS: {completed_trials}/500 trials ({completed_trials/500*100:.1f}%)")

    if completed_trials > 0:
        # Estimate time remaining (rough estimate)
        # Assume ~2 minutes per trial on average
        remaining_trials = 500 - completed_trials
        estimated_hours = remaining_trials * 2 / 60
        print(f"⏱️  Estimated remaining: ~{estimated_hours:.1f} hours")

    print("="*80 + "\n")

    conn.close()

if __name__ == '__main__':
    db_path = sys.argv[1] if len(sys.argv) > 1 else 'databases/ensemble_ft_campaign.db'
    monitor_training(db_path)
