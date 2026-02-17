#!/bin/bash
# Start Discord Paper Trader Notifier
# Sends hourly updates to Discord

cd /opt/user-data/experiment/cappuccino

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Start in watch mode (sends updates every hour)
nohup python paper_trader_discord_notifier.py --watch --interval 3600 \
  > logs/discord_notifier.log 2>&1 &

echo "Discord notifier started (PID: $!)"
echo "  Logs: logs/discord_notifier.log"
echo "  Updates sent every hour"
