#!/bin/bash
# Run 4 parallel Optuna studies to achieve <24 hour training time
# Each study trains 125 trials with n_envs=4

set -e

echo "=================================================="
echo "PARALLEL MULTI-STUDY TRAINING - SUB-24-HOUR MODE"
echo "=================================================="
echo ""

# Stop existing training
echo "1. Stopping existing training..."
pkill -f "1_optimize_unified.py" || echo "   (No existing training)"
sleep 3

# Configuration
N_PARALLEL=3        # Number of parallel studies (adjusted for 32GB RAM)
TRIALS_EACH=167     # 167 trials × 3 studies = 501 total
N_ENVS=4            # Vectorized envs per study (3 × 4 = 12 total parallel envs)

echo "2. Configuration:"
echo "   • Parallel studies: $N_PARALLEL"
echo "   • Trials per study: $TRIALS_EACH"
echo "   • Vector envs per study: $N_ENVS"
echo "   • Total parallel environments: $((N_PARALLEL * N_ENVS))"
echo "   • Total trials: $((N_PARALLEL * TRIALS_EACH))"
echo ""

# Function to start a single study
start_study() {
    local study_num=$1
    local study_name="cappuccino_5m_parallel_${study_num}"
    local log_file="logs/worker_parallel_${study_num}.log"

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

# Start all parallel studies
echo "3. Starting $N_PARALLEL parallel training studies..."
PIDS=()
for i in $(seq 1 $N_PARALLEL); do
    PID=$(start_study $i)
    PIDS+=($PID)
    echo "   ✅ Study $i started (PID: $PID)"
    sleep 2  # Stagger starts to avoid initialization conflicts
done

echo ""
echo "4. Waiting for initialization..."
sleep 15

# Check all processes started successfully
RUNNING=0
for PID in "${PIDS[@]}"; do
    if ps -p $PID > /dev/null 2>&1; then
        ((RUNNING++))
    fi
done

if [ $RUNNING -eq $N_PARALLEL ]; then
    echo "   ✅ All $N_PARALLEL studies running successfully"
    echo ""

    echo "=================================================="
    echo "PARALLEL TRAINING ACTIVE"
    echo "=================================================="
    echo ""
    echo "Expected Performance:"
    echo "  • ${N_PARALLEL}x parallelism (studies run simultaneously)"
    echo "  • Time per trial: ~5-8 minutes (with n_envs=4)"
    echo "  • Trials per hour: ~30-48 across all studies"
    echo "  • Total time: ~10-15 hours for 500 trials"
    echo ""
    echo "Monitor:"
    echo "  • htop or btop: watch CPU/RAM usage"
    echo "  • Logs: tail -f logs/worker_parallel_*.log"
    echo "  • GPU: watch -n1 rocm-smi --showuse"
    echo ""
    echo "PIDs: ${PIDS[@]}"
    echo ""
else
    echo "   ❌ Only $RUNNING/$N_PARALLEL studies started!"
    echo "   Check logs: tail -50 logs/worker_parallel_*.log"
    exit 1
fi
