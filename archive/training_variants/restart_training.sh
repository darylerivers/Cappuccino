#!/usr/bin/env bash
#
# Automated Training Restart Script
#
# This script safely stops old training workers and starts new ones
# with the latest code and a fresh study name.
#
# Usage:
#   ./restart_training.sh [OPTIONS]
#
# Options:
#   --study-name NAME    Custom study name (default: auto-generated)
#   --n-trials N         Number of trials per worker (default: 300)
#   --workers N          Number of workers to start (default: 3)
#   --gpu ID             GPU ID to use (default: 0)
#   --continue-study     Continue existing study instead of creating new one
#

set -e

# Default values
N_TRIALS=300
N_WORKERS=3
GPU_ID=0
CONTINUE_STUDY=false
STUDY_NAME=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --study-name)
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
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --continue-study)
            CONTINUE_STUDY=true
            shift
            ;;
        --help|-h)
            head -n 20 "$0" | grep "^#" | sed 's/^# //'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Generate study name if not provided
if [ -z "$STUDY_NAME" ]; then
    if [ "$CONTINUE_STUDY" = true ]; then
        # First: Try to read from .env.training
        if [ -f ".env.training" ]; then
            source .env.training
            if [ -n "$ACTIVE_STUDY_NAME" ]; then
                STUDY_NAME="$ACTIVE_STUDY_NAME"
                echo "Using active study from .env.training: $STUDY_NAME"
            fi
        fi

        # Fallback: Try to detect current study from running workers
        if [ -z "$STUDY_NAME" ]; then
            CURRENT_STUDY=$(pgrep -af "1_optimize_unified" | grep -oP '(?<=--study-name )\S+' | head -1)
            if [ -n "$CURRENT_STUDY" ]; then
                STUDY_NAME="$CURRENT_STUDY"
                echo "Detected current study: $STUDY_NAME"
            else
                echo "ERROR: No running study found to continue"
                exit 1
            fi
        fi
    else
        # Generate new study name with timestamp
        STUDY_NAME="cappuccino_$(date +%Y%m%d_%H%M%S)"
    fi
fi

echo "=========================================="
echo "Cappuccino Training Restart"
echo "=========================================="
echo "Study Name: $STUDY_NAME"
echo "Trials per Worker: $N_TRIALS"
echo "Number of Workers: $N_WORKERS"
echo "GPU ID: $GPU_ID"
echo "=========================================="
echo

# Step 1: Stop old training workers
echo "[1/3] Stopping old training workers..."
OLD_PIDS=$(pgrep -f "1_optimize_unified" || true)
if [ -n "$OLD_PIDS" ]; then
    echo "Found $(echo "$OLD_PIDS" | wc -l) old worker(s)"
    pkill -f "1_optimize_unified" || true
    sleep 2

    # Force kill if still running
    REMAINING=$(pgrep -f "1_optimize_unified" || true)
    if [ -n "$REMAINING" ]; then
        echo "Force killing remaining workers..."
        kill -9 $REMAINING || true
        sleep 1
    fi

    echo "✓ All old workers stopped"
else
    echo "No old workers found"
fi
echo

# Step 2: Create logs directory
mkdir -p logs

# Step 3: Start new training workers
echo "[2/3] Starting $N_WORKERS new training worker(s)..."
for i in $(seq 1 $N_WORKERS); do
    LOG_FILE="logs/training_worker${i}_$(date +%Y%m%d_%H%M%S).log"

    nohup python -u 1_optimize_unified.py \
        --n-trials $N_TRIALS \
        --gpu $GPU_ID \
        --study-name "$STUDY_NAME" \
        > "$LOG_FILE" 2>&1 &

    WORKER_PID=$!
    echo "  Worker $i started: PID $WORKER_PID | Log: $LOG_FILE"
    sleep 1
done
echo

# Step 4: Verify workers are running
echo "[3/3] Verifying workers..."
sleep 3
RUNNING_COUNT=$(pgrep -f "1_optimize_unified" | wc -l)

if [ "$RUNNING_COUNT" -eq "$N_WORKERS" ]; then
    echo "✓ All $N_WORKERS workers running successfully"
else
    echo "⚠️  Warning: Expected $N_WORKERS workers, but $RUNNING_COUNT are running"
    if [ "$RUNNING_COUNT" -eq 0 ]; then
        echo "❌ No workers started! Check logs for errors:"
        ls -lt logs/training_worker*.log | head -3
        exit 1
    fi
fi
echo

echo "=========================================="
echo "Training Restart Complete!"
echo "=========================================="
echo
echo "Monitor with:"
echo "  python dashboard.py"
echo "  tail -f logs/training_worker1_*.log"
echo
echo "Stop training:"
echo "  pkill -f 1_optimize_unified"
echo
