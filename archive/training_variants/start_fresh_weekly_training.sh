#!/bin/bash
# Start Fresh Weekly Training Cycle
# Run this script at the end of each week to start fresh training with current data

set -e

WEEK_DATE=$(date +%Y%m%d)
NEW_STUDY="cappuccino_week_${WEEK_DATE}"
DATA_DIR="data/1h_fresh_${WEEK_DATE}"

echo "=========================================="
echo "Fresh Weekly Training Setup"
echo "=========================================="
echo "Date: $(date)"
echo "New Study: $NEW_STUDY"
echo "Data Dir: $DATA_DIR"
echo ""

# Step 1: Download fresh data
echo "[1/5] Downloading fresh training data..."
if [ ! -d "$DATA_DIR" ]; then
    echo "  This will take 5-10 minutes..."
    python3 prepare_1year_training_data.py \
        --months 12 \
        --output-dir "$DATA_DIR" \
        --train-pct 0.8
    echo "  ✓ Data downloaded"
else
    echo "  ✓ Data already exists: $DATA_DIR"
fi

# Step 2: Update configuration
echo ""
echo "[2/5] Updating system configuration..."
cat > .env.training << EOF
# Cappuccino Trading System - Centralized Configuration
# Weekly training cycle started: $(date)

# ACTIVE TRAINING STUDY
ACTIVE_STUDY_NAME="$NEW_STUDY"

# Database configuration
OPTUNA_DB="databases/optuna_cappuccino.db"

# Training configuration
TRAINING_WORKERS=3
GPU_ID=0
N_TRIALS=500

# Auto-deployer configuration
DEPLOYER_CHECK_INTERVAL=3600
DEPLOYER_MIN_IMPROVEMENT=1.0

# Paper trading configuration
PAPER_TRADING_TIMEFRAME="1h"
PAPER_TRADING_HISTORY_HOURS=120
PAPER_TRADING_POLL_INTERVAL=60
PAPER_TRADING_TICKERS="BTC/USD ETH/USD LTC/USD BCH/USD LINK/USD UNI/USD AAVE/USD"

# Risk management
MAX_POSITION_PCT=0.30
STOP_LOSS_PCT=0.10
TRAILING_STOP_PCT=0.0

# Ensemble configuration
ENSEMBLE_TOP_N=20
ENSEMBLE_UPDATE_INTERVAL=600
EOF
echo "  ✓ Updated .env.training with new study"

# Step 3: Stop old training
echo ""
echo "[3/5] Stopping old training workers..."
pkill -f "1_optimize_unified.py" || echo "  No training workers running"
sleep 2
echo "  ✓ Old training stopped"

# Step 4: Stop automation
echo ""
echo "[4/5] Stopping automation systems..."
./stop_automation.sh 2>/dev/null || echo "  Automation not running"
sleep 2

# Step 5: Start new training
echo ""
echo "[5/5] Starting fresh training cycle..."
echo "  Workers: 3"
echo "  GPU: 0"
echo "  Trials: 500 per worker"
echo ""

for i in $(seq 1 3); do
    echo "  Starting worker $i..."
    nohup python -u 1_optimize_unified.py \
        --n-trials 500 \
        --gpu 0 \
        --study-name "$NEW_STUDY" \
        --timeframe 1h \
        --data-dir "$DATA_DIR" \
        > logs/training_worker_${i}_${NEW_STUDY}.log 2>&1 &

    WORKER_PID=$!
    echo "    PID: $WORKER_PID"
    sleep 2
done

echo ""
echo "=========================================="
echo "✓ Fresh Training Cycle Started!"
echo "=========================================="
echo ""
echo "Study: $NEW_STUDY"
echo "Data: $DATA_DIR (current as of $(date +%Y-%m-%d))"
echo "Workers: 3 (check with: pgrep -f 1_optimize_unified)"
echo ""
echo "Next steps:"
echo "  1. Wait for ~100 trials to complete (check dashboard)"
echo "  2. Start automation: ./start_automation.sh"
echo "  3. Monitor: tail -f logs/training_worker_1_${NEW_STUDY}.log"
echo ""
echo "By end of week, you'll have 100+ models trained on fresh data!"
echo ""
