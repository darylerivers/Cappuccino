#!/bin/bash
# Start Training with Automatic Ensemble Sync
# Launches training workers and ensures ensemble stays in sync

set -e

# Initialize pyenv for ROCm environment
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"

# Load centralized configuration (if available)
if [ -f ".env.training" ]; then
    source .env.training
    echo "✓ Loaded configuration from .env.training"
    STUDY_NAME="${ACTIVE_STUDY_NAME}"
    N_TRIALS="${N_TRIALS:-500}"
    N_WORKERS="${TRAINING_WORKERS:-3}"
    GPU_START="${GPU_ID:-0}"
else
    # Fallback to old behavior
    STUDY_NAME=""
    N_TRIALS=500
    N_WORKERS=3
    GPU_START=0
fi

# These can still be overridden by command line
TIMEFRAME="1h"
DATA_DIR="data/1h_1680"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --study)
            STUDY_NAME="$2"
            shift 2
            ;;
        --n-trials)
            N_TRIALS="$2"
            shift 2
            ;;
        --workers)
            N_WORKERS="$2"
            shift 2
            ;;
        --timeframe)
            TIMEFRAME="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --gpu-start)
            GPU_START="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./start_training.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --study NAME         Study name (required)"
            echo "  --n-trials N         Number of trials per worker (default: 500)"
            echo "  --workers N          Number of parallel workers (default: 3)"
            echo "  --timeframe TF       Timeframe (default: 1h)"
            echo "  --data-dir DIR       Data directory (default: data/1h_1680)"
            echo "  --gpu-start N        First GPU ID (default: 0)"
            echo ""
            echo "Example:"
            echo "  ./start_training.sh --study cappuccino_new_20251129 --n-trials 1000 --workers 3"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$STUDY_NAME" ]; then
    echo "Error: --study is required"
    echo "Use --help for usage information"
    exit 1
fi

echo "=========================================="
echo "Starting Training with Auto-Sync"
echo "=========================================="
echo ""
echo "Study:      $STUDY_NAME"
echo "Trials:     $N_TRIALS per worker"
echo "Workers:    $N_WORKERS"
echo "Timeframe:  $TIMEFRAME"
echo "Data:       $DATA_DIR"
echo "GPUs:       $GPU_START to $((GPU_START + N_WORKERS - 1))"
echo ""
echo "=========================================="
echo ""

# Check if training is already running
if pgrep -f "1_optimize_unified.py" > /dev/null; then
    echo "⚠️  Warning: Training workers are already running!"
    echo ""
    read -p "Stop existing training? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Stopping existing training workers..."
        pkill -f "1_optimize_unified.py" || true
        sleep 5
    else
        echo "Aborted. Stop existing training first."
        exit 1
    fi
fi

# Step 1: Sync ensemble and automation to this study
echo "[1/3] Syncing ensemble and automation to $STUDY_NAME..."
echo ""
./sync_training_study.sh "$STUDY_NAME"
echo ""

# Step 2: Start training workers
echo "[2/3] Starting $N_WORKERS training workers..."
echo ""

for i in $(seq 1 $N_WORKERS); do
    GPU_ID=$((GPU_START + i - 1))
    LOG_FILE="logs/training_worker_${i}_${STUDY_NAME}.log"

    echo "Starting Worker $i on GPU $GPU_ID..."

    nohup python -u scripts/training/1_optimize_unified.py \
        --n-trials $N_TRIALS \
        --gpu $GPU_ID \
        --study-name "$STUDY_NAME" \
        --timeframe "$TIMEFRAME" \
        --data-dir "$DATA_DIR" \
        > "$LOG_FILE" 2>&1 &

    WORKER_PID=$!
    echo "  PID: $WORKER_PID (log: $LOG_FILE)"

    # Small delay between worker starts
    sleep 2
done

echo ""

# Step 3: Deploy ensemble to paper trading
echo "[3/3] Deploying ensemble to paper trading..."
echo ""

# Stop any existing paper traders
pkill -f "paper_trader_alpaca_polling.py" 2>/dev/null || true
sleep 2

# Start ensemble paper trading
nohup python -u scripts/deployment/paper_trader_alpaca_polling.py \
    --model-dir train_results/ensemble \
    --tickers BTC/USD ETH/USD LTC/USD BCH/USD LINK/USD UNI/USD AAVE/USD \
    --timeframe 1h \
    --history-hours 120 \
    --poll-interval 60 \
    --gpu -1 \
    --log-file paper_trades/ensemble_${STUDY_NAME}.csv \
    --max-position-pct 0.30 \
    --stop-loss-pct 0.10 \
    --trailing-stop-pct 0.10 \
    > logs/paper_trading_ensemble.log 2>&1 &

TRADER_PID=$!
echo "✓ Paper trader started (PID: $TRADER_PID)"
echo ""

echo "=========================================="
echo "✓ Training Started Successfully!"
echo "=========================================="
echo ""
echo "Study: $STUDY_NAME"
echo "Workers: $N_WORKERS (PIDs: check with 'pgrep -f 1_optimize_unified')"
echo "Paper Trader: PID $TRADER_PID"
echo ""
echo "Automation Status:"
echo "  - Ensemble auto-updater: Syncs top 10 trials every 10 min"
echo "  - Auto-deployer: Monitors for improvements every hour"
echo "  - Watchdog: Restarts crashed processes"
echo ""
echo "Monitoring:"
echo "  Training workers:  tail -f logs/training_worker_*.log"
echo "  Paper trading:     tail -f logs/paper_trading_ensemble.log"
echo "  Automation:        ./status_automation.sh"
echo ""
echo "Stop training:"
echo "  pkill -f 1_optimize_unified.py"
echo "  ./stop_automation.sh"
echo ""
