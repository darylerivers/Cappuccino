#!/bin/bash
cd /opt/user-data/experiment/cappuccino
source activate_rocm_env.sh

export PYTHONPATH=/opt/user-data/experiment/cappuccino:$PYTHONPATH

echo "Restarting all paper traders..."

# Kill old traders
pkill -f "paper_trader_alpaca_polling.*model_"

# Start trial 91  
echo "Starting Trial #91..."
nohup python3 scripts/deployment/paper_trader_alpaca_polling.py \
  --model-dir deployments/model_0 \
  --poll-interval 60 \
  >> logs/paper_trader_trial91.log 2>&1 &
echo "  Trial 91 PID: $!"

# Start trial 100
echo "Starting Trial #100..."
nohup python3 scripts/deployment/paper_trader_alpaca_polling.py \
  --model-dir deployments/model_1 \
  --poll-interval 60 \
  >> logs/paper_trader_trial100.log 2>&1 &
echo "  Trial 100 PID: $!"

# Start trial 250  
echo "Starting Trial #250..."
nohup python3 scripts/deployment/paper_trader_alpaca_polling.py \
  --model-dir deployments/trial_250_live \
  --poll-interval 60 \
  >> logs/paper_trader_trial250.log 2>&1 &
echo "  Trial 250 PID: $!"

sleep 3

echo ""
ps aux | grep paper_trader_alpaca | grep -v grep
echo ""
echo "Monitor with: tail -f logs/paper_trader_trial*.log"
