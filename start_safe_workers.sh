#!/bin/bash
# Start 2 safe workers with GPU optimization
# Memory-safe configuration to prevent OOM

STUDY="cappuccino_auto_20260214_2059"
N_WORKERS=2
GPU=0
N_ENVS=12

cd /opt/user-data/experiment/cappuccino

echo "========================================================================"
echo "  STARTING SAFE TRAINING WORKERS"
echo "========================================================================"
echo "Study: $STUDY"
echo "Workers: $N_WORKERS (safe from OOM)"
echo "GPU Envs: $N_ENVS (GPU-accelerated)"
echo ""

mkdir -p logs

for i in $(seq 1 $N_WORKERS); do
    nohup ~/.pyenv/versions/cappuccino-rocm/bin/python -u \
        scripts/training/1_optimize_unified.py \
        --n-trials 500 \
        --gpu $GPU \
        --n-envs $N_ENVS \
        --study-name $STUDY \
        --timeframe 1h \
        --data-dir data/1h_1680 \
        > logs/worker_safe_$i.log 2>&1 &
    
    PID=$!
    echo "Worker $i started: PID $PID"
    echo "$PID $(date +%s)" >> logs/worker_pids.txt
    sleep 2
done

echo ""
echo "✅ All workers started"
echo ""
echo "Monitor:"
echo "  watch -n 10 './monitor_training.sh'"
echo "  python monitor_progress.py"
echo ""
