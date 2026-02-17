#!/bin/bash
# Monitor paper trading in real-time

cd "$(dirname "$0")"

if [ ! -f "trading.pid" ]; then
    echo "⚠️  Trading not running (no trading.pid found)"
    echo "Start trading with: ./START_TRADING.sh"
    exit 1
fi

PID=$(cat trading.pid)

if ! ps -p $PID > /dev/null 2>&1; then
    echo "⚠️  Trading process $PID is not running"
    echo "Start trading with: ./START_TRADING.sh"
    rm trading.pid
    exit 1
fi

echo "Launching monitor..."
echo "Press Ctrl+C to exit (trading will continue)"
echo ""

python3 monitor.py
