#!/bin/bash
# Quick start script for pipeline orchestrator

set -e

cd /opt/user-data/experiment/cappuccino

echo "========================================================================"
echo "AUTOMATED TRADING PIPELINE"
echo "========================================================================"
echo ""

# Check if running
PID=$(pgrep -f "pipeline_orchestrator.py --daemon" || true)

if [ ! -z "$PID" ]; then
    echo "Pipeline is already running (PID: $PID)"
    echo ""
    echo "Options:"
    echo "  1. View status:  python pipeline_orchestrator.py --status"
    echo "  2. View logs:    tail -f logs/pipeline_orchestrator.log"
    echo "  3. Stop:         kill $PID"
    echo ""
    exit 0
fi

# Check configuration
if [ ! -f "config/pipeline_config.json" ]; then
    echo "ERROR: Configuration file not found!"
    echo "Expected: config/pipeline_config.json"
    exit 1
fi

# Test mode or daemon mode
if [ "$1" == "--test" ]; then
    echo "Running in TEST mode (one-time check)..."
    echo ""
    python pipeline_orchestrator.py --once
    exit 0
fi

if [ "$1" == "--dry-run" ]; then
    echo "Running in DRY-RUN mode (no actual deployments)..."
    echo ""
    python pipeline_orchestrator.py --dry-run --once
    exit 0
fi

# Start in daemon mode
echo "Starting pipeline orchestrator in DAEMON mode..."
echo ""

nohup python -u pipeline_orchestrator.py --daemon > logs/pipeline_daemon.log 2>&1 &
PID=$!

sleep 2

# Check if still running
if ps -p $PID > /dev/null; then
    echo "✓ Pipeline started successfully (PID: $PID)"
    echo ""
    echo "Monitor with:"
    echo "  tail -f logs/pipeline_orchestrator.log"
    echo ""
    echo "Check status with:"
    echo "  python pipeline_orchestrator.py --status"
    echo ""
    echo "Stop with:"
    echo "  kill $PID"
    echo ""
else
    echo "✗ Failed to start pipeline"
    echo "Check logs/pipeline_daemon.log for errors"
    exit 1
fi

echo "========================================================================"
