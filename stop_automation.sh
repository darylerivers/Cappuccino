#!/bin/bash
# Stop All Automation Systems

echo "=========================================="
echo "Stopping Cappuccino Automation Systems"
echo "=========================================="

# Stop auto-model deployer
if [ -f deployments/auto_deployer.pid ]; then
    PID=$(cat deployments/auto_deployer.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "Stopping Auto-Model Deployer (PID: $PID)..."
        kill $PID
        sleep 2
        # Force kill if still running
        if ps -p $PID > /dev/null 2>&1; then
            kill -9 $PID
        fi
    fi
    rm -f deployments/auto_deployer.pid
fi

# Stop watchdog
if [ -f deployments/watchdog.pid ]; then
    PID=$(cat deployments/watchdog.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "Stopping System Watchdog (PID: $PID)..."
        kill $PID
        sleep 2
        if ps -p $PID > /dev/null 2>&1; then
            kill -9 $PID
        fi
    fi
    rm -f deployments/watchdog.pid
fi

# Stop performance monitor
if [ -f deployments/performance_monitor.pid ]; then
    PID=$(cat deployments/performance_monitor.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "Stopping Performance Monitor (PID: $PID)..."
        kill $PID
        sleep 2
        if ps -p $PID > /dev/null 2>&1; then
            kill -9 $PID
        fi
    fi
    rm -f deployments/performance_monitor.pid
fi

echo "All automation systems stopped"
