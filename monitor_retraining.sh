#!/bin/bash
# Monitor the retraining progress

cd /opt/user-data/experiment/cappuccino

echo "========================================================================"
echo "RETRAINING PROGRESS MONITOR"
echo "========================================================================"
echo ""

python3 << 'EOF'
import sqlite3
from datetime import datetime

conn = sqlite3.connect('databases/optuna_cappuccino.db')
cursor = conn.cursor()

# Get study info
cursor.execute("""
    SELECT COUNT(*),
           SUM(CASE WHEN state = 'COMPLETE' THEN 1 ELSE 0 END) as completed,
           SUM(CASE WHEN state = 'RUNNING' THEN 1 ELSE 0 END) as running,
           SUM(CASE WHEN state = 'FAIL' THEN 1 ELSE 0 END) as failed
    FROM trials t
    JOIN studies s ON t.study_id = s.study_id
    WHERE s.study_name = 'paper_traders_retrain'
""")

total, completed, running, failed = cursor.fetchone()

print(f"Study: paper_traders_retrain")
print(f"Target: 10 trials")
print(f"Progress: {completed}/10 complete")
print(f"Running: {running or 0}")
print(f"Failed: {failed or 0}")
print()

# Get completed trials with scores
cursor.execute("""
    SELECT t.number, tv.value, t.datetime_complete
    FROM trials t
    JOIN studies s ON t.study_id = s.study_id
    LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
    WHERE s.study_name = 'paper_traders_retrain'
    AND t.state = 'COMPLETE'
    ORDER BY tv.value DESC
""")

results = cursor.fetchall()

if results:
    print("Completed Trials:")
    print(f"{'Trial':<8} {'Sharpe':<10} {'Completed At'}")
    print("-" * 50)
    for trial_num, sharpe, completed_at in results:
        print(f"#{trial_num:<7} {sharpe:<10.4f} {completed_at}")

    print()
    print(f"Best so far: Trial #{results[0][0]} (Sharpe: {results[0][1]:.4f})")
else:
    print("No trials completed yet. Training in progress...")

conn.close()

# Check if process is still running
import subprocess
result = subprocess.run(['pgrep', '-f', 'paper_traders_retrain'],
                       capture_output=True, text=True)
if result.returncode == 0:
    print()
    print("✓ Training is RUNNING")
    print()
    print("Estimated time remaining: ~{} minutes".format((10 - completed) * 5))
else:
    print()
    print("✗ Training process not found")
    if completed >= 10:
        print("✓ Training may have completed!")
        print("  Run: ./deploy_retrained_models.sh")

EOF

echo ""
echo "========================================================================"
echo "To watch in real-time, you can also check GPU usage:"
echo "  watch -n 1 'rocm-smi --showuse'"
echo "========================================================================"
