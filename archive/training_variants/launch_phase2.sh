#!/bin/bash
# Launch Phase 2 Training with Rolling Features

nohup python 1_optimize_unified.py \
    --n-trials 50 \
    --study-name phase2_rolling_features_20251216 \
    --gpu 0 \
    --data-dir data/1h_1680_phase2 \
    --timeframe 1h \
    --num-paths 3 \
    --k-test-groups 2 \
    > logs/phase2_training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "Phase 2 Training launched! PID: $!"
echo "Monitor with: tail -f logs/phase2_training_*.log"
