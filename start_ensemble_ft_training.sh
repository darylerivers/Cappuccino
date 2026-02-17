#!/bin/bash
# Ensemble + FT-Transformer Training Campaign
# Launches 5 parallel studies to maximize GPU utilization

cd /opt/user-data/experiment/cappuccino

# Create logs directory
mkdir -p logs/training

echo "🚀 Starting Ensemble + FT-Transformer Training Campaign"
echo "============================================================"
echo "This will launch 5 parallel training workers:"
echo ""
echo "  ENSEMBLE (Baseline MLP - for voting):"
echo "    1. Conservative (low LR, high regularization)"
echo "    2. Balanced (middle ground)"
echo "    3. Aggressive (high LR, low regularization)"
echo ""
echo "  FT-TRANSFORMER (Deep dive):"
echo "    4. FT Small (fast, lightweight)"
echo "    5. FT Large (powerful, heavy)"
echo ""
echo "Each study will run 100 trials (~2-3 days total)"
echo "============================================================"
echo ""

# Ensemble Study 1: Conservative
echo "📊 Starting Ensemble Conservative..."
nohup python scripts/training/1_optimize_unified.py \
    --study-name ensemble_conservative_$(date +%Y%m%d) \
    --n-trials 100 \
    --gpu 0 \
    --timeframe 1h \
    --force-baseline \
    --storage sqlite:///databases/ensemble_ft_campaign.db \
    > logs/training/ensemble_conservative.log 2>&1 &
PID1=$!
echo "  ✅ Started (PID: $PID1)"

sleep 5

# Ensemble Study 2: Balanced
echo "📊 Starting Ensemble Balanced..."
nohup python scripts/training/1_optimize_unified.py \
    --study-name ensemble_balanced_$(date +%Y%m%d) \
    --n-trials 100 \
    --gpu 0 \
    --timeframe 1h \
    --force-baseline \
    --storage sqlite:///databases/ensemble_ft_campaign.db \
    > logs/training/ensemble_balanced.log 2>&1 &
PID2=$!
echo "  ✅ Started (PID: $PID2)"

sleep 5

# Ensemble Study 3: Aggressive
echo "📊 Starting Ensemble Aggressive..."
nohup python scripts/training/1_optimize_unified.py \
    --study-name ensemble_aggressive_$(date +%Y%m%d) \
    --n-trials 100 \
    --gpu 0 \
    --timeframe 1h \
    --force-baseline \
    --storage sqlite:///databases/ensemble_ft_campaign.db \
    > logs/training/ensemble_aggressive.log 2>&1 &
PID3=$!
echo "  ✅ Started (PID: $PID3)"

sleep 5

# FT-Transformer Study 1: Small Architecture
echo "🤖 Starting FT-Transformer Small..."
nohup python scripts/training/1_optimize_unified.py \
    --study-name ft_transformer_small_$(date +%Y%m%d) \
    --n-trials 100 \
    --gpu 0 \
    --timeframe 1h \
    --force-ft \
    --storage sqlite:///databases/ensemble_ft_campaign.db \
    > logs/training/ft_small.log 2>&1 &
PID4=$!
echo "  ✅ Started (PID: $PID4)"

sleep 5

# FT-Transformer Study 2: Large Architecture
echo "🤖 Starting FT-Transformer Large..."
nohup python scripts/training/1_optimize_unified.py \
    --study-name ft_transformer_large_$(date +%Y%m%d) \
    --n-trials 100 \
    --gpu 0 \
    --timeframe 1h \
    --force-ft \
    --storage sqlite:///databases/ensemble_ft_campaign.db \
    > logs/training/ft_large.log 2>&1 &
PID5=$!
echo "  ✅ Started (PID: $PID5)"

echo ""
echo "============================================================"
echo "✅ All 5 studies launched successfully!"
echo "============================================================"
echo ""
echo "Process IDs:"
echo "  Ensemble Conservative: $PID1"
echo "  Ensemble Balanced:     $PID2"
echo "  Ensemble Aggressive:   $PID3"
echo "  FT Small:              $PID4"
echo "  FT Large:              $PID5"
echo ""
echo "Monitor progress:"
echo "  tail -f logs/training/ensemble_conservative.log"
echo "  tail -f logs/training/ft_small.log"
echo ""
echo "Check all processes:"
echo "  ps aux | grep 1_optimize_unified"
echo ""
echo "Monitor GPU:"
echo "  watch -n 5 nvidia-smi"
echo ""
echo "Database: databases/ensemble_ft_campaign.db"
echo "Expected completion: 2-3 days"
echo ""
