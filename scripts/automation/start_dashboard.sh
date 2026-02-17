#!/bin/bash
# Quick launcher for paper trading dashboard

cd /opt/user-data/experiment/cappuccino

# Find most recent CSV files (prefer fixed versions)
ENSEMBLE_CSV=$(ls -t paper_trades/ensemble_fixed_*.csv paper_trades/watchdog_session_*.csv 2>/dev/null | head -1)
SINGLE_CSV=$(ls -t paper_trades/single_fixed_*.csv paper_trades/single_model_trial861.csv 2>/dev/null | head -1)

if [ -z "$ENSEMBLE_CSV" ]; then
    echo "⚠️  No ensemble trader CSV found!"
    echo "Looking for any recent paper trading sessions..."
    ENSEMBLE_CSV=$(ls -t paper_trades/*.csv 2>/dev/null | head -1)
fi

if [ -z "$SINGLE_CSV" ]; then
    echo "⚠️  No single model trader CSV found!"
    SINGLE_CSV="paper_trades/single_model_trial861.csv"
fi

echo "📊 Paper Trading Dashboard"
echo "=========================="
echo "Ensemble: $ENSEMBLE_CSV"
echo "Single:   $SINGLE_CSV"
echo ""
echo "Press Ctrl+C to exit"
echo ""

# Run dashboard
python paper_trading_dashboard.py \
    --ensemble-csv "$ENSEMBLE_CSV" \
    --single-csv "$SINGLE_CSV" \
    --refresh 60
