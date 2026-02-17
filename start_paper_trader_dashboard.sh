#!/bin/bash
# Start Paper Trader Live Dashboard with Countdown Timer

cd /opt/user-data/experiment/cappuccino

echo "🚀 Starting Paper Trader Dashboard..."
echo "   - Live updates with countdown timer"
echo "   - Auto-refresh every second"
echo "   - Press Ctrl+C to exit"
echo ""

python paper_trader_dashboard.py --poll-interval 3600 --refresh 1
