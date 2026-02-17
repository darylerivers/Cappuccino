#!/bin/bash
# Launch 5min ensemble + FT-Transformer training campaign
# Run this AFTER 1hr training completes or in parallel

STUDY_DB="databases/5min_campaign.db"
TIMEFRAME="5m"
N_TRIALS=100
GPU=0

echo "================================================================================"
echo "5-MINUTE TIMEFRAME TRAINING CAMPAIGN"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Timeframe: $TIMEFRAME"
echo "  Trials per study: $N_TRIALS"
echo "  Total trials: 500 (5 studies × 100)"
echo "  Database: $STUDY_DB"
echo "  GPU: $GPU"
echo ""
echo "Studies:"
echo "  1. ensemble_5m_conservative (baseline MLP, conservative risk)"
echo "  2. ensemble_5m_balanced     (baseline MLP, balanced risk)"
echo "  3. ensemble_5m_aggressive   (baseline MLP, aggressive risk)"
echo "  4. ft_5m_small              (FT-Transformer, smaller)"
echo "  5. ft_5m_large              (FT-Transformer, larger)"
echo ""
echo "================================================================================"
echo ""

# Create logs directory
mkdir -p logs/training

# 1. Ensemble Conservative
echo "Launching ensemble_5m_conservative..."
nohup python scripts/training/1_optimize_unified.py \
    --study-name ensemble_5m_conservative_$(date +%Y%m%d) \
    --n-trials $N_TRIALS --gpu $GPU --timeframe $TIMEFRAME \
    --force-baseline \
    --storage sqlite:///$STUDY_DB \
    > logs/training/ensemble_5m_conservative.log 2>&1 &
PID1=$!
echo "  PID: $PID1"

sleep 2

# 2. Ensemble Balanced
echo "Launching ensemble_5m_balanced..."
nohup python scripts/training/1_optimize_unified.py \
    --study-name ensemble_5m_balanced_$(date +%Y%m%d) \
    --n-trials $N_TRIALS --gpu $GPU --timeframe $TIMEFRAME \
    --force-baseline \
    --storage sqlite:///$STUDY_DB \
    > logs/training/ensemble_5m_balanced.log 2>&1 &
PID2=$!
echo "  PID: $PID2"

sleep 2

# 3. Ensemble Aggressive
echo "Launching ensemble_5m_aggressive..."
nohup python scripts/training/1_optimize_unified.py \
    --study-name ensemble_5m_aggressive_$(date +%Y%m%d) \
    --n-trials $N_TRIALS --gpu $GPU --timeframe $TIMEFRAME \
    --force-baseline \
    --storage sqlite:///$STUDY_DB \
    > logs/training/ensemble_5m_aggressive.log 2>&1 &
PID3=$!
echo "  PID: $PID3"

sleep 2

# 4. FT-Transformer Small
echo "Launching ft_5m_small..."
nohup python scripts/training/1_optimize_unified.py \
    --study-name ft_5m_small_$(date +%Y%m%d) \
    --n-trials $N_TRIALS --gpu $GPU --timeframe $TIMEFRAME \
    --force-ft \
    --storage sqlite:///$STUDY_DB \
    > logs/training/ft_5m_small.log 2>&1 &
PID4=$!
echo "  PID: $PID4"

sleep 2

# 5. FT-Transformer Large
echo "Launching ft_5m_large..."
nohup python scripts/training/1_optimize_unified.py \
    --study-name ft_5m_large_$(date +%Y%m%d) \
    --n-trials $N_TRIALS --gpu $GPU --timeframe $TIMEFRAME \
    --force-ft \
    --storage sqlite:///$STUDY_DB \
    > logs/training/ft_5m_large.log 2>&1 &
PID5=$!
echo "  PID: $PID5"

echo ""
echo "================================================================================"
echo "5-MINUTE TRAINING CAMPAIGN LAUNCHED!"
echo "================================================================================"
echo ""
echo "Process IDs:"
echo "  ensemble_5m_conservative: $PID1"
echo "  ensemble_5m_balanced:     $PID2"
echo "  ensemble_5m_aggressive:   $PID3"
echo "  ft_5m_small:              $PID4"
echo "  ft_5m_large:              $PID5"
echo ""
echo "Monitor progress:"
echo "  python monitor_training.py --db databases/5min_campaign.db"
echo "  tail -f logs/training/ensemble_5m_conservative.log"
echo ""
echo "Discord notifications enabled: Training updates every 10 trials"
echo ""
echo "Estimated completion: 48-72 hours"
echo "================================================================================"
