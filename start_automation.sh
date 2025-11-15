#!/bin/bash
# Start All Automation Systems
# Launches model deployer, watchdog, and performance monitor

echo "=========================================="
echo "Starting Cappuccino Automation Systems"
echo "=========================================="

# Create logs directory
mkdir -p logs deployments

# 1. Auto-Model Deployer
echo "Starting Auto-Model Deployer..."
nohup python -u auto_model_deployer.py \
    --study cappuccino_3workers_20251102_2325 \
    --check-interval 3600 \
    --min-improvement 1.0 \
    --daemon \
    > logs/auto_deployer_console.log 2>&1 &
DEPLOYER_PID=$!
echo "  PID: $DEPLOYER_PID"
echo $DEPLOYER_PID > deployments/auto_deployer.pid
sleep 2

# 2. System Watchdog
echo "Starting System Watchdog..."
nohup python -u system_watchdog.py \
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
nohup python -u performance_monitor.py \
    --study cappuccino_3workers_20251102_2325 \
    --check-interval 300 \
    > logs/performance_monitor_console.log 2>&1 &
MONITOR_PID=$!
echo "  PID: $MONITOR_PID"
echo $MONITOR_PID > deployments/performance_monitor.pid
sleep 2

echo "=========================================="
echo "All automation systems started!"
echo "=========================================="
echo ""
echo "PIDs:"
echo "  Auto-Model Deployer: $DEPLOYER_PID"
echo "  System Watchdog:     $WATCHDOG_PID"
echo "  Performance Monitor: $MONITOR_PID"
echo ""
echo "Logs:"
echo "  Auto-Deployer:  logs/auto_deployer.log"
echo "  Watchdog:       logs/watchdog.log"
echo "  Monitor:        logs/performance_monitor.log"
echo ""
echo "Control:"
echo "  ./stop_automation.sh    - Stop all automation"
echo "  ./status_automation.sh  - Check automation status"
echo ""
