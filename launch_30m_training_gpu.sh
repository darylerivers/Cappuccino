#!/bin/bash
# Launch 30m Training Campaign on GPU
# 5 parallel studies: 3 ensemble (baseline), 2 finetune (FT-Transformer)

TIMEFRAME="30m"
DATA_DIR="data/30m"
STORAGE="sqlite:///databases/30m_campaign.db"
LOG_DIR="logs/training"

mkdir -p "$LOG_DIR"
mkdir -p databases

echo "🚀 Launching 30m training campaign on GPU..."
echo ""
echo "Data: $DATA_DIR"
echo "Storage: $STORAGE"
echo "Logs: $LOG_DIR"
echo ""

# Study 1: Ensemble Baseline (Conservative)
STUDY_NAME="ensemble_30m_conservative_$(date +%Y%m%d)"
echo "Starting: $STUDY_NAME"
python scripts/training/1_optimize_unified.py \
  --n-trials 300 \
  --data-dir "$DATA_DIR" \
  --study-name "$STUDY_NAME" \
  --storage "$STORAGE" \
  --timeframe "$TIMEFRAME" \
  --gpu 0 \
  --force-baseline \
  > "$LOG_DIR/ensemble_30m_conservative_gpu.log" 2>&1 &
echo "  PID: $!"

# Study 2: Ensemble Baseline (Balanced)
STUDY_NAME="ensemble_30m_balanced_$(date +%Y%m%d)"
echo "Starting: $STUDY_NAME"
python scripts/training/1_optimize_unified.py \
  --n-trials 300 \
  --data-dir "$DATA_DIR" \
  --study-name "$STUDY_NAME" \
  --storage "$STORAGE" \
  --timeframe "$TIMEFRAME" \
  --gpu 0 \
  --force-baseline \
  > "$LOG_DIR/ensemble_30m_balanced_gpu.log" 2>&1 &
echo "  PID: $!"

# Study 3: Ensemble Baseline (Aggressive)
STUDY_NAME="ensemble_30m_aggressive_$(date +%Y%m%d)"
echo "Starting: $STUDY_NAME"
python scripts/training/1_optimize_unified.py \
  --n-trials 300 \
  --data-dir "$DATA_DIR" \
  --study-name "$STUDY_NAME" \
  --storage "$STORAGE" \
  --timeframe "$TIMEFRAME" \
  --gpu 0 \
  --force-baseline \
  > "$LOG_DIR/ensemble_30m_aggressive_gpu.log" 2>&1 &
echo "  PID: $!"

# Study 4: Finetune (Small)
STUDY_NAME="ft_30m_small_$(date +%Y%m%d)"
echo "Starting: $STUDY_NAME"
python scripts/training/1_optimize_unified.py \
  --n-trials 300 \
  --data-dir "$DATA_DIR" \
  --study-name "$STUDY_NAME" \
  --storage "$STORAGE" \
  --timeframe "$TIMEFRAME" \
  --gpu 0 \
  --force-ft \
  > "$LOG_DIR/ft_30m_small_gpu.log" 2>&1 &
echo "  PID: $!"

# Study 5: Finetune (Large)
STUDY_NAME="ft_30m_large_$(date +%Y%m%d)"
echo "Starting: $STUDY_NAME"
python scripts/training/1_optimize_unified.py \
  --n-trials 300 \
  --data-dir "$DATA_DIR" \
  --study-name "$STUDY_NAME" \
  --storage "$STORAGE" \
  --timeframe "$TIMEFRAME" \
  --gpu 0 \
  --force-ft \
  > "$LOG_DIR/ft_30m_large_gpu.log" 2>&1 &
echo "  PID: $!"

echo ""
echo "✅ All 5 studies launched!"
echo ""
echo "Monitor with:"
echo "  watch -n 5 'ps aux | grep optimize_unified'"
echo "  tail -f $LOG_DIR/ensemble_30m_conservative_gpu.log"
echo ""
echo "Check database:"
echo "  sqlite3 databases/30m_campaign.db \"SELECT name FROM studies;\""
