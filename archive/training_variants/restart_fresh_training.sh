#!/bin/bash
# Restart training with fresh data and automatically update automation

set -e

STUDY_NAME="cappuccino_fresh_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "Restarting Training with Fresh Data"
echo "=========================================="
echo "New study: $STUDY_NAME"
echo ""

# 1. Stop current training
echo "Stopping current training workers..."
pkill -f "1_optimize_unified.py" || true
sleep 5

# 2. Stop automation (so it doesn't interfere)
echo "Stopping automation..."
./stop_automation.sh

# 3. Start new training study
echo ""
echo "Starting 3 training workers..."
for i in 1 2 3; do
    nohup python -u 1_optimize_unified.py \
        --n-trials 200 \
        --gpu 0 \
        --study-name "$STUDY_NAME" \
        > logs/parallel_training/worker_${i}.log 2>&1 &
    PID=$!
    echo "  Worker $i: PID $PID"
    sleep 2
done

# 4. Update automation configuration
echo ""
echo "Updating automation configuration..."
sed -i "s/cappuccino_1year_20251121/$STUDY_NAME/g" start_automation.sh

# 5. Restart automation with new study
echo ""
echo "Restarting automation with new study..."
sleep 3
./start_automation.sh

echo ""
echo "=========================================="
echo "Training Restarted Successfully!"
echo "=========================================="
echo "Study: $STUDY_NAME"
echo "Workers: 3"
echo "Trials: 200 per worker"
echo ""
echo "Monitor training:"
echo "  tail -f logs/parallel_training/worker_1.log"
echo ""
echo "Check status:"
echo "  python dashboard.py"
echo ""
