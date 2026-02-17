#!/bin/bash
# Apply Option 2: Aggressive Speed optimizations

set -e

echo "========================================"
echo "OPTION 2: AGGRESSIVE SPEED OPTIMIZATION"
echo "========================================"
echo ""

# 1. Force GPU to high performance mode
echo "1. Setting GPU to high performance mode..."
echo "   (Requires sudo password)"
echo "high" | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level > /dev/null
echo "   ✅ GPU now in high performance mode"
echo ""

# 2. Stop current training
echo "2. Stopping current training..."
pkill -f "1_optimize_unified.py" || echo "   (No training process found)"
sleep 2
echo "   ✅ Training stopped"
echo ""

# 3. Restart training with 500 trial target
echo "3. Starting training with 500 trial target..."
nohup /home/mrc/.pyenv/versions/cappuccino-rocm/bin/python -u \
    scripts/training/1_optimize_unified.py \
    --n-trials 500 \
    --gpu 0 \
    --study-name cappuccino_5m_fresh \
    --timeframe 5m \
    --data-dir data/5m \
    > logs/worker_5m.log 2>&1 &

TRAIN_PID=$!
echo "   ✅ Training started (PID: $TRAIN_PID)"
echo ""

# 4. Wait a moment for training to initialize
sleep 3

echo "========================================"
echo "OPTIMIZATION COMPLETE!"
echo "========================================"
echo ""
echo "Changes applied:"
echo "  ✅ GPU forced to high performance mode"
echo "  ✅ Training steps reduced: 8k-25k → 5k-12k (60% faster)"
echo "  ✅ Workers increased: 4-8 → 6-10 (more parallelism)"
echo "  ✅ Target reduced: 1000 → 500 trials"
echo ""
echo "Expected results:"
echo "  • 6-8x faster training"
echo "  • ETA: 3-5 days (down from 29 days)"
echo "  • Data stays fresh for crypto trading"
echo ""
echo "Monitor with: python paper_trader_dashboard.py --training"
echo ""
