#!/bin/bash
# High Performance Training - Maximize GPU Utilization

set -e

cd /opt/user-data/experiment/cappuccino

echo "========================================================================"
echo "HIGH PERFORMANCE TRAINING STARTUP"
echo "========================================================================"
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Stop any existing training
pkill -f continuous_training.py 2>/dev/null || true
pkill -f pipeline_orchestrator.py 2>/dev/null || true
sleep 2

echo "Starting HIGH PERFORMANCE training..."
echo "  - 3 trials per cycle (parallel Optuna workers)"
echo "  - 60 second cooldown (faster cycles)"
echo "  - GPU 0 (RTX 3070)"
echo ""

# Start training with higher parallelization
nohup python continuous_training.py \
    --trials-per-cycle 3 \
    --cooldown 60 \
    --gpu 0 \
    > logs/training_daemon.log 2>&1 &

TRAINING_PID=$!
sleep 3

if ps -p $TRAINING_PID > /dev/null; then
    echo "✓ High performance training started (PID: $TRAINING_PID)"
else
    echo "✗ Training failed to start"
    cat logs/training_daemon.log | tail -20
    exit 1
fi

echo ""
echo "Starting pipeline orchestrator..."
nohup python -u pipeline_orchestrator.py --daemon > logs/pipeline_daemon.log 2>&1 &
PIPELINE_PID=$!
sleep 2

if ps -p $PIPELINE_PID > /dev/null; then
    echo "✓ Pipeline started (PID: $PIPELINE_PID)"
else
    echo "✗ Pipeline failed to start"
    exit 1
fi

echo ""
echo "========================================================================"
echo "HIGH PERFORMANCE MODE ACTIVE"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Training: 3 parallel trials per cycle"
echo "  Cooldown: 60 seconds (vs 300s default)"
echo "  Expected GPU usage: 60-80%"
echo "  Expected throughput: ~15-20 trials/hour"
echo ""
echo "Monitor GPU:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Monitor training:"
echo "  tail -f logs/continuous_training.log"
echo ""
echo "Check status:"
echo "  ./status_automation.sh"
echo ""
echo "========================================================================"
