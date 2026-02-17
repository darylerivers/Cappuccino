#!/bin/bash
# Monitor parallel training progress

while true; do
    clear
    echo "========================================"
    echo "PARALLEL TRAINING MONITOR - $(date +%H:%M:%S)"
    echo "========================================"
    echo ""

    # System resources
    echo "=== System Resources ==="
    free -h | head -2
    echo ""
    rocm-smi --showuse --showmeminfo vram | grep -E "GPU|busy|VRAM"
    echo ""

    # Training processes
    echo "=== Training Processes ==="
    ps aux | grep "1_optimize_unified" | grep -v grep | awk '{printf "PID %s: CPU %s%%, MEM %s%%\n", $2, $3, $4}'
    echo ""

    # Study progress
    echo "=== Study Progress ==="
    python3 << 'EOF'
import optuna
from datetime import datetime

for i in [1, 2, 3]:
    try:
        study = optuna.load_study(f"cappuccino_5m_parallel_{i}", storage="sqlite:///databases/optuna_cappuccino.db")
        trials = study.get_trials(deepcopy=False)
        complete = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        running = [t for t in trials if t.state == optuna.trial.TrialState.RUNNING]

        print(f"Study {i}: {len(complete)}/167 complete, {len(running)} running")

        if complete:
            latest = complete[-1]
            dur = (latest.datetime_complete - latest.datetime_start).total_seconds() / 60
            print(f"  Last: #{latest.number} ({dur:.1f}min, Sharpe:{latest.value:.4f})")

        if running:
            for t in running:
                elapsed = (datetime.now() - t.datetime_start.replace(tzinfo=None)).total_seconds() / 60
                print(f"  Running: #{t.number} ({elapsed:.1f}min)")
    except:
        print(f"Study {i}: Loading...")
EOF

    echo ""
    echo "Press Ctrl+C to exit | Refreshing in 10s..."
    sleep 10
done
