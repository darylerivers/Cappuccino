#!/bin/bash
# Quick monitoring script for 1000-trial training

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║        CAPPUCCINO 1000-TRIAL TRAINING - STATUS CHECK              ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

# Check workers
WORKERS=$(ps aux | grep 1_optimize_unified.py | grep -v grep | wc -l)
echo "🔧 Workers: $WORKERS running"
echo ""

# Check GPU
echo "🎮 GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | \
  awk -F', ' '{printf "   Utilization: %s%%\n   VRAM: %s MB / %s MB (%.0f%%)\n   Temperature: %s°C\n   Power: %s W\n", $1, $2, $3, ($2/$3)*100, $4, $5}'
echo ""

# Check progress
echo "📊 Training Progress:"
python3 << 'EOF'
import optuna
try:
    study = optuna.load_study(
        study_name='cappuccino_cge_1000trials',
        storage='sqlite:///databases/optuna_cappuccino.db'
    )
    completed = len([t for t in study.trials if t.state.name == 'COMPLETE'])
    running = len([t for t in study.trials if t.state.name == 'RUNNING'])

    pct = (completed / 1000) * 100
    remaining = 1000 - completed
    est_time = (remaining / 15) * 6  # 15 workers, 6 min avg per batch

    print(f"   Completed: {completed}/1000 ({pct:.1f}%)")
    print(f"   Running: {running}")

    if completed > 0:
        print(f"   Best Sharpe: {study.best_value:.6f}")
        print(f"   Best trial: #{study.best_trial.number}")

    if remaining > 0:
        print(f"   Estimated time remaining: {est_time/60:.1f} hours")
except Exception as e:
    print(f"   Study initializing...")
EOF

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "Commands:"
echo "  Watch logs:      tail -f training_worker_0.log"
echo "  Watch GPU:       watch -n 2 nvidia-smi"
echo "  Stop training:   kill \$(cat training_workers.pids)"
echo "═══════════════════════════════════════════════════════════════════"
