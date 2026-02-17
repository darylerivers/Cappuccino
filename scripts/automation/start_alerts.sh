#!/bin/bash
# Quick launcher for alert system

cd /opt/user-data/experiment/cappuccino

echo "🔔 Starting Paper Trading Alert System"
echo "======================================"
echo ""

# Check if psutil is installed
python3 -c "import psutil" 2>/dev/null || {
    echo "⚠️  psutil not installed. Installing..."
    pip install psutil
    echo ""
}

# Run alert system (auto-detects CSVs and PIDs)
python3 alert_system.py --check-interval 300

# If you want custom settings, use:
# python3 alert_system.py \
#     --ensemble-csv paper_trades/ensemble_fixed_20260117.csv \
#     --single-csv paper_trades/single_fixed_20260117.csv \
#     --check-interval 300
