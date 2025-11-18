# Alpaca Paper Trading Model - Training Status

**Started:** 2025-10-30 09:40
**Study Name:** `cappuccino_alpaca`
**Target:** 300 trials (3 parallel processes × 100 trials each)

---

## Configuration

**Tickers (Alpaca-compatible):**
- BTC/USD
- ETH/USD
- LTC/USD
- BCH/USD
- LINK/USD
- UNI/USD
- AAVE/USD

**Training Strategy:**
- Using Trial #13 hyperparameter ranges (exploitation mode)
- 3 parallel processes on GPU 0
- Each process: 100 trials
- Total: 300 trials

**Hyperparameter Ranges (from Trial #13):**
- Learning rate: 1.0e-6 to 3.0e-6
- Batch size: [1536, 2048, 3072]
- Gamma: 0.95 to 0.99
- Net dimension: 1408 to 1664
- (Full config in `TRIAL_13_BEST_CONFIG.py`)

---

## Current Status

✅ **Training Active**
- 3 processes running
- GPU utilization: ~50-96%
- Trials started: 3 (Trial #0, #1, #2)
- Completed: 0 (trials take ~15-20 minutes each)

---

## Monitoring Commands

### Real-time Monitor (Recommended)
```bash
python monitor.py --study-name cappuccino_alpaca
```

### Check Individual Process Logs
```bash
tail -f logs/alpaca_training/process_1.log
tail -f logs/alpaca_training/process_2.log
tail -f logs/alpaca_training/process_3.log
```

### Check GPU Usage
```bash
nvidia-smi
# Or watch continuously:
watch -n 1 nvidia-smi
```

### Check Running Processes
```bash
ps aux | grep 1_optimize_unified.py | grep -v grep
```

### Database Query (for detailed analysis)
```bash
sqlite3 optuna_cappuccino.db "SELECT trial_id, value FROM trial_values WHERE study_id = (SELECT study_id FROM studies WHERE study_name = 'cappuccino_alpaca') ORDER BY value DESC LIMIT 10;"
```

---

## Expected Timeline

**Per Trial:** ~15-20 minutes
**3 parallel processes:** ~5-7 trials per hour
**300 trials total:** ~40-60 hours (1.5-2.5 days)

**Checkpoints:**
- After 30 trials: Check if best trial found
- After 100 trials: Evaluate convergence
- After 200 trials: Consider early stopping if converged
- After 300 trials: Select best model for paper trading

---

## When Training Completes

### 1. Find Best Trial
```bash
python monitor.py --study-name cappuccino_alpaca
```

### 2. Deploy to Paper Trading
```bash
# Get best trial number from monitor
BEST_TRIAL=XX

# Launch paper trader with best model
nohup python -u paper_trader_alpaca_polling.py \
    --model-dir train_results/cwd_tests/trial_${BEST_TRIAL}_1h/ \
    --tickers BTC/USD ETH/USD LTC/USD BCH/USD LINK/USD UNI/USD AAVE/USD \
    --timeframe 1h \
    --history-hours 120 \
    --poll-interval 3600 \
    --gpu 0 \
    --log-file paper_trades/alpaca_session.csv \
    > logs/paper_trading_alpaca.log 2>&1 &
```

### 3. Monitor Paper Trading
```bash
# Check logs
tail -f logs/paper_trading_alpaca.log

# Check trades
tail -f paper_trades/alpaca_session.csv
```

---

## Troubleshooting

### Training Stopped?
```bash
# Check if processes are running
ps aux | grep 1_optimize_unified.py

# Restart if needed
./train_alpaca_model.sh 100 3
```

### GPU Out of Memory?
```bash
# Kill training and reduce parallel processes
pkill -f 1_optimize_unified.py
./train_alpaca_model.sh 100 2  # Only 2 parallel processes
```

### Database Locked?
This is normal with 3 parallel processes writing to SQLite. Optuna handles this automatically with retries.

---

## Files Created

- `train_alpaca_model.sh` - Training launch script
- `logs/alpaca_training/` - Individual process logs
- `logs/alpaca_training_master.log` - Master launch log
- `optuna_cappuccino.db` - SQLite database with trial results
- `train_results/cwd_tests/trial_*_1h/` - Model checkpoints for each trial

---

## Next Steps After Training

1. ✅ **Backtest** - Test on historical out-of-sample data
2. ✅ **Validate** - Test on recent market data
3. ✅ **Paper Trade** - Deploy to Alpaca paper trading (2+ weeks)
4. ⚠️ **Live Trade** - Deploy with real capital (only after paper success)

See `DEPLOYMENT_ROADMAP.md` for full deployment pipeline.

---

*Training started: 2025-10-30 09:40*
*Expected completion: ~2025-11-01 (48-60 hours)*
