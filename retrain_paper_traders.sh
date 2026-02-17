#!/bin/bash
# Retrain paper trading models with current environment

cd /opt/user-data/experiment/cappuccino
source activate_rocm_env.sh

echo "========================================================================"
echo "RETRAINING PAPER TRADING MODELS"
echo "========================================================================"
echo ""
echo "This will train 2 new models compatible with the current environment"
echo "Using the same hyperparameters that work in your current training."
echo ""

# Use the current best hyperparameters from your training
python3 scripts/training/1_optimize_unified.py \
  --study-name paper_traders_retrain \
  --n-trials 10 \
  --force-ft \
  --timeframe 1h \
  --data-dir data/1h_1680

echo ""
echo "Training complete! New models saved to:"
echo "  databases/optuna_cappuccino.db (study: paper_traders_retrain)"
echo ""
echo "To deploy the best trial:"
echo "  python3 deploy_best_trial.py --study paper_traders_retrain"
