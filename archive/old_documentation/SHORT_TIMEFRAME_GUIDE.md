# Short Timeframe Trading Guide

This guide explains how to train models on shorter timeframes (5m, 15m, 30m) for more frequent trading.

## Why Shorter Timeframes?

| Timeframe | Trades Per Day | Use Case |
|-----------|---------------|----------|
| 1h (current) | 24 | Long-term strategies, lower fees |
| 30m | 48 | Medium-frequency trading |
| 15m | 96 | Active trading, more opportunities |
| 5m | 288 | High-frequency, requires monitoring |

## Quick Start

### 1. Train on 15-minute candles (Recommended to start)

```bash
./train_short_timeframe.sh 15m 2
```

This will:
- Create a config file for 15m timeframe
- Download 15m historical data
- Start training with 2 workers
- Auto-scale training windows (4x more candles than 1h)

### 2. Monitor Training

**Enhanced Dashboard:**
```bash
python3 dashboard_training_detailed.py
```

Shows:
- Study overview with all timeframes
- Convergence analysis
- Training velocity (trials/hour)
- Top 10 trials
- Recent completions
- Hyperparameter statistics for top 10%

**Original Dashboard (all systems):**
```bash
python3 dashboard.py
```

### 3. Paper Trade with Trained Model

Once training completes (find best trial):

```bash
# Find best 15m model
python3 dashboard_training_detailed.py --refresh 0 | grep "15m"

# Start paper trading (example with trial 1234)
python -u paper_trader_alpaca_polling.py \
  --model-dir train_results/cwd_tests/trial_1234_15m \
  --tickers AAVE/USD AVAX/USD BTC/USD LINK/USD ETH/USD LTC/USD UNI/USD \
  --timeframe 15m \
  --history-hours 120 \
  --poll-interval 60 \
  --gpu -1 \
  --log-file paper_trades/session_15m_$(date +%Y%m%d_%H%M%S).csv
```

## Timeframe Comparison

### Data Requirements

| Timeframe | Candles/Day | Training Data | Download Time |
|-----------|-------------|---------------|---------------|
| 1h | 24 | ~60 days | ~2 min |
| 30m | 48 | ~60 days | ~4 min |
| 15m | 96 | ~60 days | ~8 min |
| 5m | 288 | ~60 days | ~25 min |

### Trading Frequency

**1h (Current Setup):**
- Trades: Once per hour
- Next trade: When hourly candle completes
- Monitoring: Check every hour
- Best for: Swing trading, lower stress

**15m (Recommended Next):**
- Trades: Every 15 minutes (96/day)
- Next trade: Every quarter hour (:00, :15, :30, :45)
- Monitoring: Check every 15-30 minutes
- Best for: Active day trading

**5m (Advanced):**
- Trades: Every 5 minutes (288/day)
- Next trade: Very frequent
- Monitoring: Continuous
- Best for: High-frequency trading, requires automation

## Training Tips

### For 15m Timeframe

```bash
# Balanced approach - good for most users
./train_short_timeframe.sh 15m 2
```

**Pros:**
- 4x more trading opportunities than 1h
- Still manageable to monitor
- Good balance of frequency vs. stability

**Cons:**
- More transaction fees
- Needs more attention than 1h

### For 5m Timeframe

```bash
# Aggressive - for experienced traders
./train_short_timeframe.sh 5m 3
```

**Pros:**
- Maximum trading opportunities
- Can catch quick price movements
- Higher potential returns

**Cons:**
- Much higher transaction fees
- Requires constant monitoring
- More noise in signals
- Longer training time

## Monitoring Training

### Enhanced Dashboard Features

```bash
python3 dashboard_training_detailed.py --refresh 10
```

Shows every 10 seconds:

**Study Overview:**
- All studies with trial counts
- Best and average values per study
- Compare different timeframes

**Convergence Analysis:**
- Total trials completed
- Current best value
- Average of last 100/500 trials
- Improvement rate
- Standard deviation

**Training Velocity:**
- Trials completed in last hour
- Trials completed in last 6 hours
- Trials completed in last 24 hours
- Trials per hour rate

**Top 10 Trials:**
- Ranked by Sharpe ratio
- Shows timeframe and folder
- Color-coded (green=top 3, cyan=4-5)

**Recent Completions:**
- Last 8 trials
- Duration and completion time
- Performance metrics

**Hyperparameter Analysis (Top 10%):**
- Mean ± std for numeric params
- Mode for categorical params
- Ranges: [min, max]
- Key params: learning_rate, gamma, net_dimension, batch_size, lookback

## Example Workflow

### Day 1: Start 15m Training

```bash
# Morning
./train_short_timeframe.sh 15m 2

# Check progress
python3 dashboard_training_detailed.py
```

### Day 2-3: Monitor Training

```bash
# Check training progress
python3 dashboard_training_detailed.py --refresh 30

# Look for:
# - Convergence (is best value improving?)
# - Velocity (trials/hour - should be ~1-2 for 2 workers)
# - Top trials (looking for Sharpe > 1.5)
```

### Day 4: Deploy Best Model

```bash
# Find best trial from dashboard
# Example: trial_5432_15m

# Start paper trading
python -u paper_trader_alpaca_polling.py \
  --model-dir train_results/cwd_tests/trial_5432_15m \
  --tickers AAVE/USD AVAX/USD BTC/USD LINK/USD ETH/USD LTC/USD UNI/USD \
  --timeframe 15m \
  --history-hours 120 \
  --poll-interval 60 \
  --gpu -1 \
  --log-file paper_trades/best_15m_session.csv

# Create symlink for dashboard
ln -sf best_15m_session.csv paper_trades/alpaca_session.csv
ln -sf paper_trading_live.log logs/paper_trading_live.log
```

## Troubleshooting

### Data Download Fails

```bash
# Check your .env file
cat .env | grep APCA

# Manually download
python3 0_dl_trainval_data.py --timeframe 15m
```

### Training Too Slow

```bash
# Check GPU usage
nvidia-smi

# Reduce workers
pkill -f "1_optimize_unified"
./train_short_timeframe.sh 15m 1  # Use 1 worker instead of 2
```

### Paper Trading Not Updating

Check the timeframe matches:
```bash
# Model timeframe
cat train_results/cwd_tests/trial_1234_15m/best_trial | grep timeframe

# Paper trader timeframe
ps aux | grep paper_trader | grep timeframe
```

## Best Practices

1. **Start with 15m**
   - Good balance of frequency and manageability
   - Not too aggressive for beginners

2. **Train for 150+ trials**
   - Let the optimizer explore
   - More trials = better convergence

3. **Monitor regularly**
   - Use enhanced dashboard
   - Check convergence every few hours
   - Stop if no improvement after 100 trials

4. **Test before going live**
   - Paper trade for at least 1 week
   - Compare different timeframes
   - Monitor transaction costs

5. **Compare timeframes**
   ```bash
   # Train multiple timeframes
   ./train_short_timeframe.sh 1h 2   # Baseline
   ./train_short_timeframe.sh 15m 2  # Medium frequency
   ./train_short_timeframe.sh 5m 2   # High frequency

   # Compare in dashboard
   python3 dashboard_training_detailed.py
   ```

## Performance Expectations

### Typical Training Times

| Timeframe | Workers | Trials/Hour | Time to 150 Trials |
|-----------|---------|-------------|-------------------|
| 1h | 2 | 1-2 | 3-5 days |
| 15m | 2 | 0.5-1 | 6-10 days |
| 5m | 3 | 0.3-0.5 | 10-20 days |

### Transaction Cost Impact

**1h timeframe:**
- 24 trades/day × 0.1% fee = 2.4% daily
- Monthly: ~50% in fees

**15m timeframe:**
- 96 trades/day × 0.1% fee = 9.6% daily
- Monthly: ~200% in fees
- **Need higher Sharpe ratio to overcome fees!**

**5m timeframe:**
- 288 trades/day × 0.1% fee = 28.8% daily
- Monthly: ~600% in fees
- **Only viable with very high Sharpe ratio (>3.0)**

## Next Steps

1. Try the enhanced dashboard:
   ```bash
   python3 dashboard_training_detailed.py
   ```

2. Start 15m training:
   ```bash
   ./train_short_timeframe.sh 15m 2
   ```

3. Monitor and compare with 1h results

4. Deploy best model to paper trading

5. Iterate based on results
