#!/bin/bash
# Start training with vectorized environments for <1 day completion

set -e

echo "================================================"
echo "VECTORIZED TRAINING - SUB-1-DAY SPEEDUP"
echo "================================================"
echo ""

# Kill existing training
echo "1. Stopping existing training..."
pkill -f "1_optimize_unified.py" || echo "   (No existing training)"
sleep 3

# Number of parallel environments (8-16 recommended for RX 7900 GRE)
N_ENVS=12
echo ""
echo "2. Configuration:"
echo "   • Parallel environments: $N_ENVS"
echo "   • Study: cappuccino_5m_vectorized"
echo "   • Target: 500 trials"
echo "   • Break steps: 5k-12k (aggressive)"
echo "   • Batch size: 4k-16k (large for GPU)"
echo ""

# Start vectorized training
echo "3. Starting vectorized training..."
nohup /home/mrc/.pyenv/versions/cappuccino-rocm/bin/python -u \
    scripts/training/1_optimize_unified.py \
    --n-trials 500 \
    --gpu 0 \
    --study-name cappuccino_5m_vectorized \
    --timeframe 5m \
    --data-dir data/5m \
    --n-envs $N_ENVS \
    > logs/worker_vectorized.log 2>&1 &

TRAIN_PID=$!
echo "   ✅ Training started (PID: $TRAIN_PID)"
echo ""

# Wait for initialization
echo "4. Waiting for initialization..."
sleep 15

# Check if started successfully
if ps -p $TRAIN_PID > /dev/null; then
    echo "   ✅ Training running successfully"
    echo ""

    echo "================================================"
    echo "VECTORIZED TRAINING ACTIVE"
    echo "================================================"
    echo ""
    echo "Expected Performance:"
    echo "  • GPU utilization: 70-85% (up from 60%)"
    echo "  • Training speed: 8-12x faster per trial"
    echo "  • Time per trial: ~3-5 minutes (down from 30-45min)"
    echo "  • Total time: <20 hours for 500 trials"
    echo ""
    echo "Monitor with:"
    echo "  • Dashboard: python paper_trader_dashboard.py --training"
    echo "  • Logs: tail -f logs/worker_vectorized.log"
    echo "  • GPU: watch -n1 rocm-smi --showuse"
    echo ""
else
    echo "   ❌ Training failed to start!"
    echo "   Check logs: tail -50 logs/worker_vectorized.log"
    exit 1
fi
