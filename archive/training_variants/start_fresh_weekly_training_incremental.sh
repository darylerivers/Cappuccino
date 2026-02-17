#!/bin/bash
# Start Fresh Weekly Training Cycle - INCREMENTAL VERSION
# Uses incremental data updates (much faster than full re-download!)

set -e

WEEK_DATE=$(date +%Y%m%d)
NEW_STUDY="cappuccino_week_${WEEK_DATE}"
OLD_DATA_DIR="data/1h_fresh_$(ls -t data/ | grep 1h_fresh | head -1 | cut -d_ -f3)"
NEW_DATA_DIR="data/1h_fresh_${WEEK_DATE}"

echo "=========================================="
echo "Fresh Weekly Training Setup (Incremental)"
echo "=========================================="
echo "Date: $(date)"
echo "New Study: $NEW_STUDY"
echo "Old Data: $OLD_DATA_DIR"
echo "New Data: $NEW_DATA_DIR"
echo ""

# Step 1: Update data incrementally (MUCH FASTER!)
echo "[1/5] Updating training data incrementally..."

if [ -d "$OLD_DATA_DIR" ]; then
    echo "  Found existing data: $OLD_DATA_DIR"
    echo "  Downloading only NEW data (30 seconds instead of 5-10 minutes!)"

    python3 update_data_incremental.py \
        --data-dir "$OLD_DATA_DIR" \
        --output-dir "$NEW_DATA_DIR"

    echo "  ✓ Data updated incrementally"
else
    echo "  No existing data found - doing full download"
    echo "  This will take 5-10 minutes..."

    python3 prepare_1year_training_data.py \
        --months 12 \
        --output-dir "$NEW_DATA_DIR" \
        --train-pct 0.8

    echo "  ✓ Data downloaded"
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

# Data configuration - FRESH DATA (Updated $WEEK_DATE)
DATA_DIR="$NEW_DATA_DIR"

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
TRAILING_STOP_PCT=0.015

# Ensemble configuration
ENSEMBLE_TOP_N=20
ENSEMBLE_UPDATE_INTERVAL=600

# Two-Phase Training - DISABLED FOR NOW
TWO_PHASE_ENABLED=false
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

# Step 5: Start new training with 72-hour timeout
echo ""
echo "[5/5] Starting fresh training cycle with 72h timeout..."
echo "  Workers: 3"
echo "  GPU: 0"
echo "  Time limit: 72 hours (auto-stops)"
echo ""

MAX_HOURS=72
TIMEOUT_SECONDS=$((MAX_HOURS * 3600))

for i in $(seq 1 3); do
    echo "  Starting worker $i (timeout: ${MAX_HOURS}h)..."

    nohup timeout ${TIMEOUT_SECONDS}s python -u 1_optimize_unified.py \
        --n-trials 99999 \
        --gpu 0 \
        --study-name "$NEW_STUDY" \
        --timeframe 1h \
        --data-dir "$NEW_DATA_DIR" \
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
echo "Data: $NEW_DATA_DIR (updated incrementally - super fast!)"
echo "Workers: 3 with 72-hour auto-stop"
echo ""
echo "Next steps:"
echo "  1. Wait for ~100 trials to complete (check dashboard)"
echo "  2. Start automation: ./start_automation.sh"
echo "  3. Monitor: tail -f logs/training_worker_1_${NEW_STUDY}.log"
echo ""
echo "Workers will automatically stop after 72 hours."
echo ""
