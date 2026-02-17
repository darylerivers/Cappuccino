#!/bin/bash
# Start All Automation Systems
# Launches model deployer, watchdog, and performance monitor

# Initialize pyenv for ROCm environment
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"

# Load centralized configuration
if [ -f ".env.training" ]; then
    source .env.training
    echo "✓ Loaded configuration from .env.training"
    echo "  Active Study: $ACTIVE_STUDY_NAME"
else
    echo "❌ ERROR: .env.training not found!"
    echo "   Please create .env.training with ACTIVE_STUDY_NAME"
    exit 1
fi

echo "=========================================="
echo "Starting Cappuccino Automation Systems"
echo "=========================================="
echo "Active Study: $ACTIVE_STUDY_NAME"
echo ""

# Create logs directory
mkdir -p logs deployments

# 1. Auto-Model Deployer
echo "Starting Auto-Model Deployer..."
nohup python -u scripts/deployment/auto_model_deployer.py \
    --study "$ACTIVE_STUDY_NAME" \
    --check-interval ${DEPLOYER_CHECK_INTERVAL:-3600} \
    --min-improvement ${DEPLOYER_MIN_IMPROVEMENT:-1.0} \
    --daemon \
    > logs/auto_deployer_console.log 2>&1 &
DEPLOYER_PID=$!
echo "  PID: $DEPLOYER_PID"
echo $DEPLOYER_PID > deployments/auto_deployer.pid
sleep 2

# 2. System Watchdog
echo "Starting System Watchdog..."
nohup python -u monitoring/system_watchdog.py \
    --check-interval 60 \
    --max-restarts 3 \
    --restart-cooldown 300 \
    > logs/watchdog_console.log 2>&1 &
WATCHDOG_PID=$!
echo "  PID: $WATCHDOG_PID"
echo $WATCHDOG_PID > deployments/watchdog.pid
sleep 2

# 3. Performance Monitor
echo "Starting Performance Monitor..."
nohup python -u monitoring/performance_monitor.py \
    --study "$ACTIVE_STUDY_NAME" \
    --check-interval 300 \
    > logs/performance_monitor_console.log 2>&1 &
MONITOR_PID=$!
echo "  PID: $MONITOR_PID"
echo $MONITOR_PID > deployments/performance_monitor.pid
sleep 2

# 4. Ensemble Auto-Updater
echo "Starting Ensemble Auto-Updater..."
nohup python -u models/ensemble_auto_updater.py \
    --study "$ACTIVE_STUDY_NAME" \
    --ensemble-dir train_results/ensemble \
    --top-n ${ENSEMBLE_TOP_N:-20} \
    --interval ${ENSEMBLE_UPDATE_INTERVAL:-600} \
    > logs/ensemble_updater_console.log 2>&1 &
ENSEMBLE_PID=$!
echo "  PID: $ENSEMBLE_PID"
echo $ENSEMBLE_PID > deployments/ensemble_updater.pid
sleep 2

# 5. Two-Phase Training Scheduler
if [ "${TWO_PHASE_ENABLED:-false}" = "true" ]; then
    echo "Starting Two-Phase Training Scheduler..."
    nohup python -u two_phase_scheduler.py \
        --daemon \
        > logs/two_phase_scheduler_console.log 2>&1 &
    TWO_PHASE_PID=$!
    echo "  PID: $TWO_PHASE_PID"
    echo $TWO_PHASE_PID > deployments/two_phase_scheduler.pid
    sleep 2
else
    echo "Two-Phase Training Scheduler: DISABLED"
    echo "  (Set TWO_PHASE_ENABLED=true in .env.training to enable)"
fi

echo "=========================================="
echo "All automation systems started!"
echo "=========================================="
echo ""
echo "PIDs:"
echo "  Auto-Model Deployer:  $DEPLOYER_PID"
echo "  System Watchdog:      $WATCHDOG_PID"
echo "  Performance Monitor:  $MONITOR_PID"
echo "  Ensemble Updater:     $ENSEMBLE_PID"
if [ "${TWO_PHASE_ENABLED:-false}" = "true" ]; then
    echo "  Two-Phase Scheduler:  $TWO_PHASE_PID"
fi
echo ""
echo "Logs:"
echo "  Auto-Deployer:    logs/auto_deployer.log"
echo "  Watchdog:         logs/watchdog.log"
echo "  Monitor:          logs/performance_monitor.log"
echo "  Ensemble Updater: logs/ensemble_updater_console.log"
if [ "${TWO_PHASE_ENABLED:-false}" = "true" ]; then
    echo "  Two-Phase:        logs/two_phase_scheduler.log"
fi
echo ""
echo "Control:"
echo "  ./stop_automation.sh    - Stop all automation"
echo "  ./status_automation.sh  - Check automation status"
echo ""
