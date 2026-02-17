#!/bin/bash
# Quick script to scale training to MINIMAL mode (2 workers)
# Use this when you need GPU for other tasks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Scaling training to MINIMAL mode (2 workers)..."
echo "This will free ~7 GB VRAM for other GPU tasks"
echo ""

CURRENT_COUNT=$(ps aux | grep "1_optimize_unified" | grep -v grep | wc -l)
TARGET_COUNT=2

if [ $CURRENT_COUNT -le $TARGET_COUNT ]; then
    echo "Already at or below $TARGET_COUNT workers"
    echo "Current: $CURRENT_COUNT workers"
    exit 0
fi

TO_STOP=$((CURRENT_COUNT - TARGET_COUNT))

echo "Will stop $TO_STOP workers (keeping $TARGET_COUNT running)"
echo ""

# Get PIDs to stop (newest workers)
PIDS=$(ps aux | grep "1_optimize_unified" | grep -v grep | \
       sort -k2 -n -r | head -n $TO_STOP | awk '{print $2}')

echo "Workers to stop:"
ps aux | grep "1_optimize_unified" | grep -v grep | \
    sort -k2 -n -r | head -n $TO_STOP | \
    awk '{printf "  PID: %s, RAM: %d MB\n", $2, $6/1024}'
echo ""

read -p "Continue? (y/n): " confirm
if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

for pid in $PIDS; do
    kill $pid && echo "  ✓ Stopped PID: $pid"
done

sleep 2

echo ""
echo "New status:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | \
    awk -F, '{printf "  VRAM: %s / %s (%.0f%%)\n", $1, $2, ($1/$2)*100}'
echo "  Workers: $(ps aux | grep 1_optimize_unified | grep -v grep | wc -l)"
echo ""
echo "✓ Training scaled to MINIMAL mode"
echo "  Training speed: ~12 trials/hour"
echo "  GPU is now available for other tasks"
