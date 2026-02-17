#!/bin/bash
# Monitor training progress

echo "=== Training Monitor ==="
echo "Started: $(date)"
echo ""

# Find the latest training log
LATEST_LOG=$(ls -t logs/training/training_14indicators_*.log 2>/dev/null | head -1)
BACKGROUND_LOG="/tmp/claude-1000/-opt-user-data-experiment-cappuccino/tasks/bce998f.output"

if [ -f "$LATEST_LOG" ]; then
    echo "Training log: $LATEST_LOG"
    tail -30 "$LATEST_LOG"
elif [ -f "$BACKGROUND_LOG" ]; then
    echo "Background training output:"
    tail -30 "$BACKGROUND_LOG"
else
    echo "No training logs found"
fi

echo ""
echo "=== Training Process ==="
ps aux | grep "1_optimize" | grep -v grep

echo ""
echo "=== GPU Usage ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"

echo ""
echo "=== Study Database ==="
ls -lh databases/*.db 2>/dev/null | tail -5
