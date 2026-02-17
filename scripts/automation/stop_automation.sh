#!/bin/bash
# Stop all automation: Training + Pipeline

cd /opt/user-data/experiment/cappuccino

echo "========================================================================"
echo "STOPPING AUTOMATION"
echo "========================================================================"
echo ""

# Stop training
echo "Stopping continuous training..."
TRAINING_PID=$(pgrep -f "continuous_training.py" || true)
if [ ! -z "$TRAINING_PID" ]; then
    kill $TRAINING_PID
    echo "✓ Sent stop signal to training (PID: $TRAINING_PID)"
    
    # Wait for graceful shutdown
    for i in {1..5}; do
        if ! ps -p $TRAINING_PID > /dev/null 2>&1; then
            echo "✓ Training stopped"
            break
        fi
        sleep 1
    done
    
    # Force kill if still running
    if ps -p $TRAINING_PID > /dev/null 2>&1; then
        kill -9 $TRAINING_PID 2>/dev/null || true
        echo "✓ Training force stopped"
    fi
else
    echo "  (Training not running)"
fi

echo ""

# Stop pipeline
echo "Stopping pipeline orchestrator..."
PIPELINE_PID=$(pgrep -f "pipeline_orchestrator.py --daemon" || true)
if [ ! -z "$PIPELINE_PID" ]; then
    kill $PIPELINE_PID
    echo "✓ Sent stop signal to pipeline (PID: $PIPELINE_PID)"
    
    # Wait for graceful shutdown
    for i in {1..5}; do
        if ! ps -p $PIPELINE_PID > /dev/null 2>&1; then
            echo "✓ Pipeline stopped"
            break
        fi
        sleep 1
    done
    
    # Force kill if still running
    if ps -p $PIPELINE_PID > /dev/null 2>&1; then
        kill -9 $PIPELINE_PID 2>/dev/null || true
        echo "✓ Pipeline force stopped"
    fi
else
    echo "  (Pipeline not running)"
fi

echo ""
echo "========================================================================"
echo "AUTOMATION STOPPED"
echo "========================================================================"
