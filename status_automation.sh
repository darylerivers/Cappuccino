#!/bin/bash
# Check Status of All Automation Systems

echo "=========================================="
echo "Cappuccino Automation Status"
echo "=========================================="
echo ""

# Function to check process status
check_process() {
    local name=$1
    local pid_file=$2
    local pattern=$3

    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        if ps -p $PID > /dev/null 2>&1; then
            UPTIME=$(ps -o etime= -p $PID | xargs)
            CPU=$(ps -o %cpu= -p $PID | xargs)
            MEM=$(ps -o %mem= -p $PID | xargs)
            echo "✓ $name"
            echo "  Status: RUNNING"
            echo "  PID:    $PID"
            echo "  Uptime: $UPTIME"
            echo "  CPU:    ${CPU}%"
            echo "  MEM:    ${MEM}%"
            return 0
        else
            echo "✗ $name"
            echo "  Status: NOT RUNNING (stale PID: $PID)"
            return 1
        fi
    else
        # Check by process pattern
        PID=$(pgrep -f "$pattern" | head -1)
        if [ -n "$PID" ]; then
            UPTIME=$(ps -o etime= -p $PID | xargs)
            echo "⚠ $name"
            echo "  Status: RUNNING (no PID file)"
            echo "  PID:    $PID"
            echo "  Uptime: $UPTIME"
            return 0
        else
            echo "✗ $name"
            echo "  Status: NOT RUNNING"
            return 1
        fi
    fi
}

# Check each automation system
check_process "Auto-Model Deployer" "deployments/auto_deployer.pid" "auto_model_deployer.py"
echo ""

check_process "System Watchdog" "deployments/watchdog.pid" "system_watchdog.py"
echo ""

check_process "Performance Monitor" "deployments/performance_monitor.pid" "performance_monitor.py"
echo ""

echo "=========================================="
echo "Deployment Status"
echo "=========================================="

if [ -f "deployments/deployment_state.json" ]; then
    echo "Last deployed model:"
    python3 -c "
import json
try:
    with open('deployments/deployment_state.json', 'r') as f:
        state = json.load(f)

    if state.get('last_deployed_trial'):
        print(f\"  Trial ID:    {state['last_deployed_trial']}\")
        print(f\"  Value:       {state['last_deployed_value']:.6f}\")
        print(f\"  Deployed at: {state['last_deployment_time']}\")

        if state.get('deployment_history'):
            print(f\"\\nTotal deployments: {len(state['deployment_history'])}\")
    else:
        print('  No deployments yet')
except Exception as e:
    print(f'  Error reading state: {e}')
"
else
    echo "  No deployment state file found"
fi

echo ""
echo "=========================================="
echo "Recent Alerts"
echo "=========================================="

if [ -f "deployments/watchdog_state.json" ]; then
    python3 -c "
import json
from datetime import datetime
try:
    with open('deployments/watchdog_state.json', 'r') as f:
        state = json.load(f)

    alerts = state.get('alerts', [])
    if alerts:
        recent = alerts[-5:]  # Last 5 alerts
        for alert in recent:
            ts = alert['timestamp'][:19]  # Trim subseconds
            severity = alert['severity']
            process = alert['process']
            message = alert['message']
            print(f\"[{ts}] [{severity}] {process}: {message}\")
    else:
        print('No alerts')
except Exception as e:
    print(f'Error reading alerts: {e}')
"
else
    echo "No watchdog state file found"
fi

echo ""
echo "=========================================="
echo "Logs"
echo "=========================================="
echo "  tail -f logs/auto_deployer.log"
echo "  tail -f logs/watchdog.log"
echo "  tail -f logs/performance_monitor.log"
echo ""
