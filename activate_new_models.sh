#!/bin/bash
# Activate newly retrained models for trials 91 and 100

set -e
cd /opt/user-data/experiment/cappuccino
source activate_rocm_env.sh

echo "========================================================================"
echo "ACTIVATING NEW MODELS"
echo "========================================================================"
echo ""

# Check that new models exist
if [ ! -d "deployments/model_91_new" ] || [ ! -d "deployments/model_100_new" ]; then
    echo "✗ New models not found!"
    echo "  Run ./deploy_retrained_models.sh first"
    exit 1
fi

echo "Step 1: Stopping old broken traders..."
pkill -f 'paper_trader_alpaca.*model_[01]' || true
sleep 2

echo ""
echo "Step 2: Backing up old models..."
if [ -d "deployments/model_0" ]; then
    mv deployments/model_0 deployments/model_0_backup_$(date +%Y%m%d_%H%M%S)
fi
if [ -d "deployments/model_1" ]; then
    mv deployments/model_1 deployments/model_1_backup_$(date +%Y%m%d_%H%M%S)
fi

echo ""
echo "Step 3: Activating new models..."
mv deployments/model_91_new deployments/model_0
mv deployments/model_100_new deployments/model_1

echo ""
echo "Step 4: Starting new traders..."
export PYTHONPATH=/opt/user-data/experiment/cappuccino:$PYTHONPATH

# Start trial 91 (now in model_0)
echo "Starting Trial #91..."
nohup python3 scripts/deployment/paper_trader_alpaca_polling.py \
  --model-dir deployments/model_0 \
  --poll-interval 60 \
  > logs/paper_trader_trial91.log 2>&1 &
TRIAL91_PID=$!
echo "  PID: $TRIAL91_PID"

# Start trial 100 (now in model_1)
echo "Starting Trial #100..."
nohup python3 scripts/deployment/paper_trader_alpaca_polling.py \
  --model-dir deployments/model_1 \
  --poll-interval 60 \
  > logs/paper_trader_trial100.log 2>&1 &
TRIAL100_PID=$!
echo "  PID: $TRIAL100_PID"

sleep 3

echo ""
echo "Step 5: Verifying traders started..."
if ps -p $TRIAL91_PID > /dev/null; then
    echo "  ✓ Trial #91 running (PID $TRIAL91_PID)"
else
    echo "  ✗ Trial #91 failed to start - check logs/paper_trader_trial91.log"
fi

if ps -p $TRIAL100_PID > /dev/null; then
    echo "  ✓ Trial #100 running (PID $TRIAL100_PID)"
else
    echo "  ✗ Trial #100 failed to start - check logs/paper_trader_trial100.log"
fi

echo ""
echo "========================================================================"
echo "ACTIVATION COMPLETE!"
echo "========================================================================"
echo ""
echo "Paper traders now running:"
echo "  Trial #91  → deployments/model_0 (new compatible model)"
echo "  Trial #100 → deployments/model_1 (new compatible model)"
echo "  Trial #250 → deployments/trial_250_live (unchanged)"
echo "  Trial #965 → paper_trading_trial965 (unchanged)"
echo ""
echo "Monitor with:"
echo "  tail -f logs/paper_trader_trial91.log"
echo "  tail -f logs/paper_trader_trial100.log"
echo ""
echo "Check dashboard to see all traders active!"
echo "========================================================================"
