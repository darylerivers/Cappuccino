#!/bin/bash
# Check automation status

cd /opt/user-data/experiment/cappuccino

echo "========================================================================"
echo "AUTOMATION STATUS"
echo "========================================================================"
echo ""

# Check training
TRAINING_PID=$(pgrep -f "continuous_training.py" || true)
if [ ! -z "$TRAINING_PID" ]; then
    echo "Training: ✓ RUNNING (PID: $TRAINING_PID)"
    
    # Show last few log lines
    if [ -f "logs/continuous_training.log" ]; then
        echo "  Last activity:"
        tail -3 logs/continuous_training.log | sed 's/^/    /'
    fi
else
    echo "Training: ✗ NOT RUNNING"
fi

echo ""

# Check pipeline
PIPELINE_PID=$(pgrep -f "pipeline_orchestrator.py --daemon" || true)
if [ ! -z "$PIPELINE_PID" ]; then
    echo "Pipeline: ✓ RUNNING (PID: $PIPELINE_PID)"
    
    # Show last few log lines
    if [ -f "logs/pipeline_orchestrator.log" ]; then
        echo "  Last activity:"
        tail -3 logs/pipeline_orchestrator.log | sed 's/^/    /'
    fi
else
    echo "Pipeline: ✗ NOT RUNNING"
fi

echo ""
echo "========================================================================"

# Show recent trial count
if [ -f "databases/optuna_cappuccino.db" ]; then
    TOTAL_TRIALS=$(sqlite3 databases/optuna_cappuccino.db "SELECT COUNT(*) FROM trials WHERE state='COMPLETE';" 2>/dev/null || echo "?")
    echo "Total completed trials: $TOTAL_TRIALS"
fi

# Show pipeline state
if [ -f "deployments/pipeline_state.json" ]; then
    TRIALS_IN_PIPELINE=$(python -c "import json; data=json.load(open('deployments/pipeline_state.json')); print(len(data.get('trials', {})))" 2>/dev/null || echo "?")
    echo "Trials in pipeline: $TRIALS_IN_PIPELINE"
fi

echo "========================================================================"
