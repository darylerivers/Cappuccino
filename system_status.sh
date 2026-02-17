#!/bin/bash
# Comprehensive system status for all running processes

echo "========================================"
echo "CAPPUCCINO TRAINING SYSTEM STATUS"
echo "========================================"
echo ""

# Training processes
echo "TRAINING PROCESSES:"
TRAINING=$(ps aux | grep "1_optimize_unified" | grep -v grep)
if [ -z "$TRAINING" ]; then
    echo "  ❌ No training processes running"
else
    echo "$TRAINING" | awk '{printf "  ✅ PID %s: %s CPU, %s MB RAM\n", $2, $3"%", int($6/1024)}'
    echo "  Total: $(echo "$TRAINING" | wc -l) processes"
fi
echo ""

# Pipeline orchestrator
echo "PIPELINE ORCHESTRATOR:"
ORCHESTRATOR=$(ps aux | grep "pipeline_orchestrator" | grep -v grep)
if [ -z "$ORCHESTRATOR" ]; then
    echo "  ❌ Not running"
else
    PID=$(echo "$ORCHESTRATOR" | awk '{print $2}')
    echo "  ✅ Running (PID $PID)"
    echo "  Database: $(grep optuna_db_path config/pipeline_config.json | cut -d'"' -f4)"
    echo "  Study: $(grep study_name config/pipeline_config.json | grep -v null | cut -d'"' -f4)"
fi
echo ""

# Kill switch
echo "KILL SWITCH:"
KILLSWITCH=$(ps aux | grep "kill_switch.sh" | grep -v grep)
if [ -z "$KILLSWITCH" ]; then
    echo "  ❌ Not running"
else
    PID=$(echo "$KILLSWITCH" | awk '{print $2}')
    echo "  ✅ Running (PID $PID)"
    LAST_CHECK=$(tail -1 logs/kill_switch.log)
    echo "  Last check: $LAST_CHECK"
fi
echo ""

# GPU status
echo "GPU STATUS:"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits | awk -F, '{printf "  Compute: %s%%  |  Memory BW: %s%%  |  VRAM: %sMB / %sMB (%d%%)\n  Power: %sW  |  Temp: %s°C\n", $1, $2, $3, $4, int($3*100/$4), $5, $6}'
echo ""

# RAM status
echo "RAM STATUS:"
free -h | grep Mem | awk '{printf "  Used: %s / %s (%s%%)\n  Available: %s\n", $3, $2, int($3*100/$2), $7}'
echo ""

# Recent training trials
echo "RECENT TRIALS (from main study):"
sqlite3 databases/optuna_stable_max.db "SELECT t.number, tv.value, datetime(t.datetime_complete) FROM trials t JOIN trial_values tv ON t.trial_id = tv.trial_id WHERE t.state = 'COMPLETE' ORDER BY t.number DESC LIMIT 5" 2>/dev/null | awk -F'|' '{printf "  Trial %s: score %.6f (completed %s)\n", $1, $2, $3}' || echo "  No completed trials yet"
echo ""

echo "========================================"
echo "Quick commands:"
echo "  ./summary_status.sh       - Quick GPU/RAM status"
echo "  tail -f logs/training_*.log - Watch training"
echo "  tail -f logs/pipeline_orchestrator.log - Watch orchestrator"
echo "  pkill -f 1_optimize_unified - Emergency stop training"
echo "========================================"
