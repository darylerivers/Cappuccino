#!/bin/bash
# Continuous pipeline monitoring

while true; do
    clear
    python pipeline_viewer.py
    echo ""
    echo "Press Ctrl+C to stop watching"
    echo "Refreshing in 10 seconds..."
    sleep 10
done
