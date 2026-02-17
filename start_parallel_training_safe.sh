#!/bin/bash
# Safe restart of parallel training after crash
# Uses 2 parallel studies instead of 3 to reduce RAM pressure

set -e

echo "=================================================="
echo "PARALLEL TRAINING - SAFE MODE (Post-Crash)"
echo "=================================================="
echo ""

# Stop any zombie processes
echo "1. Cleaning up any remaining processes..."
pkill -9 -f "1_optimize_unified.py" 2>/dev/null || echo "   (No processes to kill)"
sleep 3

# Check system resources
echo ""
echo "2. System Resources:"
free -h | grep -E "Mem|Swap"
echo ""

# AGGRESSIVE - Push GPU to max (3 parallel studies was working at 80-84%)
N_PARALLEL=3        # 3 parallel studies for best GPU utilization
TRIALS_EACH=167     # 501 total trials
N_ENVS=8            # 24 total parallel envs (3 processes × 8 envs each)

echo "3. Configuration (GPU-optimized):"
echo "   • Parallel studies: $N_PARALLEL (reduced for stability)"
echo "   • Trials per study: $TRIALS_EACH"
echo "   • Vectorization: ENABLED (n_envs=$N_ENVS)"
echo "   • Total parallel envs: $((N_PARALLEL * N_ENVS))"
echo "   • Total trials: $((N_PARALLEL * TRIALS_EACH))"
echo "   • Expected RAM: ~20-24GB (vs 32GB total)"
echo ""

# Function to start a single study
start_study() {
    local study_num=$1
    local study_name="cappuccino_5m_aggressive_${study_num}"  # New name to avoid batch_size distribution conflict
    local log_file="logs/worker_aggressive_${study_num}.log"

    nohup /home/mrc/.pyenv/versions/cappuccino-rocm/bin/python -u \
        scripts/training/1_optimize_unified.py \
        --n-trials $TRIALS_EACH \
        --gpu 0 \
        --study-name "$study_name" \
        --timeframe 5m \
        --data-dir data/5m \
        --n-envs $N_ENVS \
        > "$log_file" 2>&1 &

    echo $!
}

# Start parallel studies
echo "4. Starting $N_PARALLEL parallel training studies..."
PIDS=()
for i in $(seq 1 $N_PARALLEL); do
    PID=$(start_study $i)
    PIDS+=($PID)
    echo "   ✅ Study $i started (PID: $PID, Log: logs/worker_aggressive_${i}.log)"
    sleep 3  # Stagger starts
done

echo ""
echo "5. Waiting for initialization..."
sleep 20

# Verify processes are running
RUNNING=0
for PID in "${PIDS[@]}"; do
    if ps -p $PID > /dev/null 2>&1; then
        ((RUNNING++))
    fi
done

echo ""
if [ $RUNNING -eq $N_PARALLEL ]; then
    echo "=================================================="
    echo "✅ TRAINING ACTIVE - SAFE MODE"
    echo "=================================================="
    echo ""
    echo "Studies running: $RUNNING/$N_PARALLEL"
    echo "PIDs: ${PIDS[@]}"
    echo ""
    echo "Expected Performance:"
    echo "  • RAM usage: ~20-24GB / 32GB (65-75% - safe)"
    echo "  • GPU usage: 80-95% (vectorized, GPU-optimized)"
    echo "  • Trials/hour: ~4-6 (2 studies × 2-3 trials/hour each)"
    echo "  • ETA: ~80-125 hours (~3-5 days)"
    echo ""
    echo "Monitor:"
    echo "  • Resources: htop"
    echo "  • GPU: watch -n5 rocm-smi --showuse"
    echo "  • Logs: tail -f logs/worker_aggressive_*.log"
    echo ""
    echo "To check progress:"
    echo "  python3 << 'EOF'"
    echo "  import optuna"
    echo "  for i in [1,2]:"
    echo "      s = optuna.load_study(f'cappuccino_5m_aggressive_{i}', 'sqlite:///databases/optuna_cappuccino.db')"
    echo "      c = sum(1 for t in s.get_trials(deepcopy=False) if t.state == optuna.trial.TrialState.COMPLETE)"
    echo "      print(f'Study {i}: {c}/250')"
    echo "  EOF"
    echo ""
else
    echo "❌ Only $RUNNING/$N_PARALLEL studies started!"
    echo "Check logs: tail -50 logs/worker_aggressive_*.log"
    exit 1
fi
