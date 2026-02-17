# Running Systems Monitor

Generated: 2025-11-02 23:21

## Active Systems

### 1. Training (3 Parallel Workers)
- **PIDs**: 20704, 20756, 20817
- **Study**: cappuccino_3workers_20251102_2325
- **Status**: Running Trials 0, 1, 2 in parallel
- **GPU**: RTX 3070 (99% utilization, 113W)
- **GPU Memory**: 500MB + 390MB + 328MB = 1.2GB
- **CPU Usage**: ~110% per worker (330% total)
- **Trials**: 100 per worker (300 total)
- **Tickers**: BTC, ETH, LTC, BCH, LINK, UNI, AAVE

#### Monitor Training
```bash
# Watch all 3 workers
tail -f logs/parallel_training/worker_1.log
tail -f logs/parallel_training/worker_2.log
tail -f logs/parallel_training/worker_3.log

# Check GPU usage
watch -n 2 nvidia-smi

# Check Optuna study progress
sqlite3 optuna_cappuccino.db "SELECT COUNT(*) FROM trials WHERE study_id=(SELECT study_id FROM studies WHERE study_name='cappuccino_3workers_20251102_2325') AND state='COMPLETE'"

# Count processes
ps aux | grep "1_optimize_unified.py" | grep -v grep | wc -l
```

### 2. Paper Trading (PID 17710)
- **Model**: train_results/cwd_tests/trial_13_1h
- **Status**: Active polling
- **Timeframe**: 1h bars
- **Poll Interval**: 60 seconds
- **Tickers**: BTC/USD, ETH/USD, LTC/USD, BCH/USD, LINK/USD, UNI/USD, AAVE/USD
- **Log File**: paper_trades/alpaca_session_20251102_230855.csv
- **API**: Alpaca Paper Trading

#### Monitor Paper Trading
```bash
# Watch trading log (new rows every hour)
watch -n 10 "tail -5 paper_trades/alpaca_session_20251102_230855.csv"

# Count trades
wc -l paper_trades/alpaca_session_20251102_230855.csv

# Check process status
ps aux | grep paper_trader | grep -v grep

# View last trade
tail -1 paper_trades/alpaca_session_20251102_230855.csv | cut -d',' -f1-4
```

## Important Notes

### Training
- Each trial takes ~10-20 minutes with 6 splits
- 10 trials = ~2-3 hours total
- Safe to Ctrl+C and resume with same command
- Results stored in: train_results/cwd_tests/

### Paper Trading
- Polls Alpaca API every 60 seconds
- New trade rows appear when new 1h bars close
- Next bar expected at top of next hour
- Uses dummy sentiment (zeros) by default
- Portfolio starts with $1M virtual cash

## Stop Commands

```bash
# Stop training only
pkill -f "1_optimize_unified.py"

# Stop paper trading only
pkill -f "paper_trader_alpaca_polling.py"

# Stop both
pkill -f "1_optimize_unified.py|paper_trader_alpaca_polling.py"
```

## Startup Commands

### Resume/Start Training (3 Parallel Workers)
```bash
cd /home/mrc/experiment/cappuccino
STUDY_NAME="cappuccino_3workers_20251102_2325"
mkdir -p logs/parallel_training
for i in 1 2 3; do
  python -u 1_optimize_unified.py --n-trials 100 --gpu 0 --study-name "$STUDY_NAME" --tickers BTC ETH LTC BCH LINK UNI AAVE > "logs/parallel_training/worker_${i}.log" 2>&1 &
  sleep 3
done
echo "All 3 workers started"
```

### Resume/Start Paper Trading
```bash
cd /home/mrc/experiment/cappuccino
LOG_FILE="paper_trades/alpaca_session_$(date +%Y%m%d_%H%M%S).csv"
python paper_trader_alpaca_polling.py \
  --model-dir train_results/cwd_tests/trial_13_1h \
  --tickers BTC/USD ETH/USD LTC/USD BCH/USD LINK/USD UNI/USD AAVE/USD \
  --timeframe 1h \
  --history-hours 120 \
  --poll-interval 60 \
  --log-file "$LOG_FILE" \
  --gpu -1 > logs/paper_trading.log 2>&1 &
```

## Logs Location
- Training Worker 1: `logs/parallel_training/worker_1.log`
- Training Worker 2: `logs/parallel_training/worker_2.log`
- Training Worker 3: `logs/parallel_training/worker_3.log`
- Paper trading: `logs/paper_trading.log` (buffered, may be empty)
- Paper trades CSV: `paper_trades/alpaca_session_20251102_230855.csv`

## Current GPU Status
- **Utilization**: 99% (excellent!)
- **Power**: 113W / 220W
- **Temperature**: 53°C
- **Memory**: 2.6GB / 8.2GB used (1.2GB for training, 1.4GB for desktop)
