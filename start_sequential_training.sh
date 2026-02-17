#!/usr/bin/env bash
#
# Start Sequential Trial Training
#
# Runs trials one at a time to avoid OOM kills
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Get current study
STUDY=$(cat .current_study 2>/dev/null || echo "")

if [ -z "$STUDY" ]; then
    echo "❌ No current study found in .current_study"
    echo "   Please specify study name:"
    read -p "Study name: " STUDY
fi

# Default parameters
N_TRIALS="${1:-50}"
GPU="${2:-0}"
N_ENVS="${3:-12}"
DATA_DIR="${4:-data/1h_1680}"
TIMEFRAME="${5:-1h}"

echo "========================================================================"
echo "  SEQUENTIAL TRIAL SCHEDULER (GPU-OPTIMIZED)"
echo "========================================================================"
echo ""
echo "Study:      $STUDY"
echo "Trials:     $N_TRIALS"
echo "GPU:        $GPU"
echo "GPU Envs:   $N_ENVS parallel environments (moves data to VRAM)"
echo "Data:       $DATA_DIR"
echo "Timeframe:  $TIMEFRAME"
echo ""
echo "This will run trials ONE AT A TIME to avoid memory issues."
echo "Press Ctrl+C to stop gracefully between trials."
echo ""
read -p "Start sequential training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# Kill any existing parallel workers first
if pgrep -f "1_optimize_unified" > /dev/null; then
    echo ""
    echo "⚠️  Found existing parallel workers running"
    read -p "Kill them and switch to sequential mode? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Stopping parallel workers..."
        pkill -f "1_optimize_unified" || true
        sleep 2
        echo "✅ Parallel workers stopped"
    else
        echo "❌ Cannot run sequential scheduler with parallel workers active"
        exit 1
    fi
fi

# Start sequential scheduler
echo ""
echo "🚀 Starting sequential trial scheduler..."
echo "   Log: logs/sequential_training.log"
echo ""

mkdir -p logs

nohup python -u scripts/automation/sequential_trial_scheduler.py \
    --study "$STUDY" \
    --n-trials "$N_TRIALS" \
    --gpu "$GPU" \
    --n-envs "$N_ENVS" \
    --data-dir "$DATA_DIR" \
    --timeframe "$TIMEFRAME" \
    > logs/sequential_training.log 2>&1 &

PID=$!
echo "✅ Sequential scheduler started (PID: $PID)"
echo ""
echo "Monitor progress:"
echo "  tail -f logs/sequential_training.log"
echo ""
echo "Stop scheduler:"
echo "  pkill -f sequential_trial_scheduler"
echo ""
