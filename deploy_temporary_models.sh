#!/bin/bash
# Deploy working model temporarily to trials 91 and 100

cd /opt/user-data/experiment/cappuccino
source activate_rocm_env.sh

echo "Deploying Trial #13 (Sharpe 0.1720) to slots 91 and 100..."

# Backup old broken models
if [ -d "deployments/model_0" ]; then
    mv deployments/model_0 deployments/model_0_broken_backup 2>/dev/null || true
fi
if [ -d "deployments/model_1" ]; then
    mv deployments/model_1 deployments/model_1_broken_backup 2>/dev/null || true
fi

# Create new directories
mkdir -p deployments/model_0
mkdir -p deployments/model_1

# Copy trial 13 to both slots
for dir in deployments/model_0 deployments/model_1; do
    cp databases/trials/trial_13.pkl "$dir/best_trial"
    echo '{"trial_number": 13, "study": "cappuccino_ft_16gb_optimized", "sharpe": 0.1720, "temporary": true}' > "$dir/metadata.json"
done

echo "✓ Models deployed"
echo ""
echo "Now starting paper traders..."

export PYTHONPATH=/opt/user-data/experiment/cappuccino:$PYTHONPATH

# Kill old broken traders
pkill -f 'paper_trader_alpaca.*model_[01]' 2>/dev/null || true
sleep 2

# Start trial 91
nohup python3 scripts/deployment/paper_trader_alpaca_polling.py \
  --model-dir deployments/model_0 \
  --poll-interval 60 \
  > logs/paper_trader_trial91.log 2>&1 &
echo "Trial #91 started (PID: $!)"

# Start trial 100
nohup python3 scripts/deployment/paper_trader_alpaca_polling.py \
  --model-dir deployments/model_1 \
  --poll-interval 60 \
  > logs/paper_trader_trial100.log 2>&1 &
echo "Trial #100 started (PID: $!)"

sleep 3

echo ""
echo "✓ Temporary fix complete!"
echo ""
echo "All traders now running:"
echo "  Trial #91  → Using trial #13 (temporary)"
echo "  Trial #100 → Using trial #13 (temporary)"
echo "  Trial #250 → Original"  
echo "  Trial #965 → Original"
echo ""
echo "Dashboard should show all traders active now!"
