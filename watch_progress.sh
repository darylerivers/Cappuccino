#!/bin/bash
# Auto-refresh progress monitor

while true; do
    clear
    python monitor_progress.py
    sleep 30
    echo ""
    echo "Refreshing in 30s... (Ctrl+C to exit)"
    sleep 5
done
