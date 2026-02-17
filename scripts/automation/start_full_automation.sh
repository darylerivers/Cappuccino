#!/bin/bash
# Start full automation: Training + Pipeline

set -e

cd /opt/user-data/experiment/cappuccino

echo "========================================================================"
echo "FULL AUTOMATION STARTUP"
echo "========================================================================"
echo ""

# Check if training is already running
TRAINING_PID=$(pgrep -f "continuous_training.py" || true)
if [ ! -z "$TRAINING_PID" ]; then
    echo "⚠️  Training already running (PID: $TRAINING_PID)"
else
    echo "Starting continuous training..."
    nohup python continuous_training.py > logs/training_daemon.log 2>&1 &
    TRAINING_PID=$!
    sleep 2
    
    if ps -p $TRAINING_PID > /dev/null; then
        echo "✓ Training started (PID: $TRAINING_PID)"
    else
        echo "✗ Training failed to start"
        exit 1
    fi
fi

echo ""

# Check if pipeline is already running
PIPELINE_PID=$(pgrep -f "pipeline_orchestrator.py --daemon" || true)
if [ ! -z "$PIPELINE_PID" ]; then
    echo "⚠️  Pipeline already running (PID: $PIPELINE_PID)"
else
    echo "Starting pipeline orchestrator..."
    nohup python -u pipeline_orchestrator.py --daemon > logs/pipeline_daemon.log 2>&1 &
    PIPELINE_PID=$!
    sleep 2
    
    if ps -p $PIPELINE_PID > /dev/null; then
        echo "✓ Pipeline started (PID: $PIPELINE_PID)"
    else
        echo "✗ Pipeline failed to start"
        exit 1
    fi
fi

echo ""
echo "========================================================================"
echo "FULL AUTOMATION RUNNING"
echo "========================================================================"
echo ""
echo "Training PID: ${TRAINING_PID:-<already running>}"
echo "Pipeline PID: ${PIPELINE_PID:-<already running>}"
echo ""
echo "Monitor training:"
echo "  tail -f logs/continuous_training.log"
echo ""
echo "Monitor pipeline:"
echo "  tail -f logs/pipeline_orchestrator.log"
echo ""
echo "Check status:"
echo "  ps aux | grep -E '(continuous_training|pipeline_orchestrator)' | grep -v grep"
echo ""
echo "Stop both:"
echo "  pkill -f continuous_training.py"
echo "  pkill -f 'pipeline_orchestrator.py --daemon'"
echo ""
echo "========================================================================"
