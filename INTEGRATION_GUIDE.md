# System Integration Guide

## Overview

This guide explains how the new robust training, conviction scoring, and Coinbase live trading components integrate together.

---

## Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1_optimize_unified.py (existing)                           │
│         ↓                                                    │
│  robust_training_wrapper.py (NEW)                           │
│    - Retry logic (OOM errors)                               │
│    - Checkpointing                                          │
│    - Model registry database                                │
│    - Atomic saves                                           │
│         ↓                                                    │
│  databases/model_registry.db                                │
│    - Track all trained models                               │
│    - Status: training/completed/failed                      │
│    - Hyperparameters, Sharpe, paths                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   DEPLOYMENT PHASE                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  PAPER TRADING (Alpaca)                                     │
│    paper_trader_alpaca_polling.py (existing)                │
│      - Load trained model                                   │
│      - Execute paper trades                                 │
│      - Log to CSV                                           │
│         ↓                                                    │
│  live_performance_monitor.py (NEW)                          │
│    - Monitor Sharpe vs backtest                             │
│    - Discord alerts                                         │
│    - Auto-stop on severe degradation                        │
│                                                              │
│  LIVE TRADING (Coinbase)                                    │
│    live_trader_coinbase.py (NEW)                            │
│      - Load trained model                                   │
│      - Fetch market data (Coinbase API)                     │
│      - Get model predictions                                │
│      - Filter by conviction (NEW)                           │
│      - Execute trades (maker-optimized)                     │
│      - Risk management                                      │
│         ↓                                                    │
│  conviction_scorer.py (NEW)                                 │
│    - Multi-factor scoring                                   │
│    - Filter 300+ signals → 75-100 trades/month              │
│    - Position size, Sharpe momentum, ensemble agreement     │
│    - Volatility filter, portfolio context                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Robust Training System

### Purpose
Fix the 50-96% trial failure rate by adding retry logic and error handling.

### Components
- **`robust_training_wrapper.py`**: Wraps existing training with robustness
- **`model_registry.db`**: SQLite database tracking all models
- **Features**:
  - Automatic retry on OOM errors (3 attempts)
  - Checkpoint saving every N episodes
  - Atomic file operations (no partial saves)
  - Disk space checks before training
  - Model registry for tracking all attempts

### Usage

#### Basic Training with Robustness
```bash
python scripts/training/robust_training_wrapper.py \
    --study-name ensemble_robust_5m \
    --timeframe 5m \
    --n-trials 100 \
    --gpu 0 \
    --checkpoint-frequency 100 \
    --max-retries 3
```

#### Integration with Existing Training
```python
from scripts.training.robust_training_wrapper import RobustTrainingWrapper

# Create objective function (your existing training code)
def my_objective(trial):
    # Your existing training logic
    return sharpe_ratio

# Wrap with robustness
wrapper = RobustTrainingWrapper(
    study_name="my_study",
    timeframe="5m",
    n_trials=100,
    gpu=0,
    max_retries=3
)

# Run with retry logic and error handling
wrapper.run_training_campaign(my_objective)
```

### Model Registry Database

Query best models:
```python
from scripts.training.robust_training_wrapper import ModelRegistry

registry = ModelRegistry()

# Get top 5 models by Sharpe
top_models = registry.get_best_models("ensemble_5m", top_n=5)

for model_id, trial_num, sharpe, path, status in top_models:
    print(f"Trial #{trial_num}: Sharpe {sharpe:.4f} - {status}")
    print(f"  Path: {path}")
```

Direct SQL queries:
```bash
sqlite3 databases/model_registry.db

# View all completed models
SELECT trial_number, sharpe_ratio, training_duration_seconds, model_path
FROM models
WHERE status='completed'
ORDER BY sharpe_ratio DESC
LIMIT 10;

# Count failures by error type
SELECT substr(error_message, 1, 50) as error, COUNT(*)
FROM models
WHERE status='failed'
GROUP BY substr(error_message, 1, 50)
ORDER BY COUNT(*) DESC;

# Check training progress
SELECT status, COUNT(*)
FROM models
GROUP BY status;
```

---

## 2. Conviction Scoring System

### Purpose
Reduce 300+ trades/month to 75-100 high-quality trades by filtering low-conviction signals.

### Components
- **`utils/conviction_scorer.py`**: Multi-factor scoring system
- **`ConvictionFilter`**: Simple wrapper for integration

### Scoring Factors (weights sum to 1.0)

1. **Position Size (30%)**: Larger actions = higher model confidence
2. **Sharpe Momentum (20%)**: Recent performance trend
3. **Ensemble Agreement (20%)**: Multiple models agreeing
4. **Volatility Filter (15%)**: Avoid high-volatility periods
5. **Portfolio Context (15%)**: Diversification needs, cash levels

### Usage in Paper Trader

```python
from utils.conviction_scorer import ConvictionFilter

# Initialize filter
conviction_filter = ConvictionFilter(min_score=0.6)

# In trading loop
raw_actions = model.predict(state)  # -1 to +1 for each ticker

# Build state dict
state_dict = {
    'positions': current_positions,
    'cash': available_cash,
    'portfolio_value': total_value,
    'prices': current_prices,
    'price_history': recent_prices
}

# Filter actions
filtered_actions = conviction_filter.filter_actions(
    raw_actions,
    state_dict
)

# Execute only high-conviction trades
for ticker, action in filtered_actions.items():
    execute_trade(ticker, action)

# Update performance tracking
conviction_filter.update_performance(portfolio_return)

# Get statistics
stats = conviction_filter.get_stats()
print(f"Projected monthly trades: {stats['projected_monthly_trades']:.0f}")
```

### Tuning Conviction Thresholds

```python
# More selective (fewer trades, higher quality)
filter = ConvictionFilter(
    min_score=0.7,  # Higher threshold
    position_size_weight=0.4,  # Emphasize strong signals
    sharpe_momentum_weight=0.3  # Only trade when performing well
)

# Less selective (more trades, lower quality)
filter = ConvictionFilter(
    min_score=0.5,  # Lower threshold
    position_size_weight=0.2,  # Accept weaker signals
    volatility_filter_weight=0.1  # Trade even in volatility
)
```

### Monitoring Conviction Performance

```python
stats = filter.get_stats()

print(f"Total trades: {stats['total_trades']}")
print(f"Average conviction: {stats['avg_conviction']:.3f}")
print(f"Trades per day: {stats['trades_per_day']:.1f}")
print(f"Projected monthly: {stats['projected_monthly_trades']:.0f}")
```

---

## 3. Coinbase Live Trader

### Purpose
Deploy trained models to Coinbase with maker optimization, risk management, and conviction filtering.

### Components
- **`scripts/deployment/live_trader_coinbase.py`**: Complete live trading system
- **Features**:
  - Automatic model loading from trained models
  - Market data fetching (Coinbase Advanced API)
  - Conviction scoring integration
  - Maker order optimization (target 70%+ maker ratio)
  - Risk management (daily limits, consecutive losses)
  - Discord notifications
  - Fee tier tracking

### Prerequisites

1. **Install Coinbase SDK**
```bash
pip install coinbase-advanced-py
```

2. **Get API Credentials**
- Go to Coinbase Advanced → Settings → API
- Create new API key with trade permissions
- Save key and secret to `.env`:
```bash
COINBASE_API_KEY=your_api_key_here
COINBASE_API_SECRET=your_api_secret_here
```

3. **Fund Account**
- Transfer $1,000 from Coinbase to Coinbase Advanced
- Verify funds available for trading

### Usage

#### Deploy Trained Model
```bash
python scripts/deployment/live_trader_coinbase.py \
    --model-dir train_results/trial_250 \
    --tickers BTC-USD ETH-USD SOL-USD AVAX-USD \
    --timeframe 5m \
    --initial-capital 1000 \
    --poll-interval 300 \
    --gpu 0 \
    --log-file live_trades/coinbase_trial250.csv
```

#### Paper Trading First (Recommended)
```bash
# Test on Alpaca paper first
python scripts/deployment/paper_trader_alpaca_polling.py \
    --model-dir train_results/trial_250 \
    --tickers BTC/USD ETH/USD SOL/USD AVAX/USD \
    --timeframe 5m

# Monitor performance for 48-72 hours
tail -f paper_trades/trial250_session.csv

# If Sharpe > 0 and stable, deploy to Coinbase
python scripts/deployment/live_trader_coinbase.py ...
```

### Risk Management Settings

Adjust in code or pass as parameters:
```python
trader = CoinbaseLiveTrader(
    ...,
    maker_preference=0.7,  # Target 70% maker orders
    max_daily_loss_pct=0.02,  # Stop if down 2% in a day
    max_consecutive_losses=3,  # Stop after 3 losses in a row
)
```

### Fee Optimization

**Maker vs Taker Orders:**
- **Maker** (limit orders): 0.05-0.15% fees
- **Taker** (market orders): 0.10-0.50% fees

**Strategy:**
- Try maker order first (70% preference)
- Place limit order slightly better than market (0.05% away)
- Wait up to 2 minutes for fill
- If not filled, cancel and use taker order
- Target: 70%+ maker ratio for 30-40% cost savings

**Fee Tier Progression:**
| Tier | Maker | Taker | 30-Day Volume |
|------|-------|-------|---------------|
| Intro 1 | 0.60% | 1.20% | $0 |
| Advanced 1 | 0.25% | 0.50% | $25k |
| VIP 1 | 0.06% | 0.125% | $500k |
| VIP 3 | 0.04% | 0.085% | $5M |

Monitor your progress:
```bash
# View live trader logs
tail -f logs/live_trader.log

# Check maker ratio
tail live_trades/coinbase_trial250.csv | awk -F',' '{print $11}'
```

---

## 4. Live Performance Monitoring

### Purpose
Monitor paper/live trading performance and alert on degradation.

### Components
- **`monitoring/live_performance_monitor.py`**: Continuous monitoring
- **Features**:
  - Hourly Sharpe calculation from CSV
  - Compare to backtest expectations
  - Discord alerts at thresholds
  - Auto-stop on severe degradation

### Usage

Already running for Trial #250:
```bash
# Check status
ps aux | grep live_performance_monitor

# View logs
tail -f logs/live_performance_monitor.log

# View state
cat monitoring/live_monitor_state.json
```

Deploy for new model:
```bash
python monitoring/live_performance_monitor.py \
    --paper-trader-csv paper_trades/trial300_session.csv \
    --backtest-sharpe 0.25 \
    --check-interval 3600 \
    --warning-threshold 0.5 \
    --critical-threshold -1.0 \
    --emergency-threshold -2.0 \
    --emergency-duration 24
```

### Alert Thresholds

| Level | Condition | Action |
|-------|-----------|--------|
| ✅ OK | Sharpe within 0.5 of backtest | No alerts |
| ⚠️ WARNING | Sharpe < (backtest - 0.5) | Discord notification |
| 🔴 CRITICAL | Sharpe < -1.0 | Discord notification |
| 🚨 EMERGENCY | Sharpe < -2.0 for 24h | Auto-stop trading |

---

## 5. Complete Deployment Workflow

### Phase 1: Training (GPU Required)

```bash
# 1. Download 5min data (if not done)
python scripts/data/prepare_multi_timeframe_data.py \
    --timeframe 5m \
    --duration 6mo

# 2. Start robust training campaign
python scripts/training/robust_training_wrapper.py \
    --study-name ensemble_5m_robust \
    --timeframe 5m \
    --n-trials 100 \
    --gpu 0 \
    --max-retries 3

# 3. Monitor progress
tail -f logs/training/robust_wrapper_ensemble_5m_robust.log

# 4. Query best models when complete
sqlite3 databases/model_registry.db \
  "SELECT trial_number, sharpe_ratio, model_path
   FROM models
   WHERE status='completed'
   ORDER BY sharpe_ratio DESC
   LIMIT 5;"
```

### Phase 2: Paper Trading Validation (Alpaca)

```bash
# Deploy best model to paper trading
python scripts/deployment/paper_trader_alpaca_polling.py \
    --model-dir train_results/trial_XXX \
    --tickers BTC/USD ETH/USD SOL/USD AVAX/USD \
    --timeframe 5m

# Start performance monitor
python monitoring/live_performance_monitor.py \
    --paper-trader-csv paper_trades/trialXXX_session.csv \
    --backtest-sharpe 0.XX \
    --check-interval 3600 &

# Monitor for 48-72 hours
# Check Discord for alerts
# View dashboard: python infrastructure/status_dashboard.py
```

### Phase 3: Live Trading Deployment (Coinbase)

**Only proceed if paper trading shows:**
- ✅ Positive Sharpe ratio (>0.1)
- ✅ No critical alerts from monitor
- ✅ Stable performance for 48+ hours
- ✅ Reasonable trade frequency (2-5 trades/day)
- ✅ Maker ratio >50% if using conviction filtering

```bash
# Deploy to Coinbase with $1,000
python scripts/deployment/live_trader_coinbase.py \
    --model-dir train_results/trial_XXX \
    --tickers BTC-USD ETH-USD SOL-USD AVAX-USD \
    --timeframe 5m \
    --initial-capital 1000 \
    --poll-interval 300 \
    --gpu 0 \
    --log-file live_trades/coinbase_prod_v1.csv

# Monitor closely first 24 hours
tail -f logs/live_trader.log

# Check trades
tail -f live_trades/coinbase_prod_v1.csv

# View Discord for trade notifications
```

### Phase 4: Incremental Scaling

After 1 week of successful live trading:
```bash
# Stop trader (Ctrl+C)

# Deploy with $2,000
python scripts/deployment/live_trader_coinbase.py \
    --initial-capital 2000 \
    ...

# After 2 more weeks, deploy with $3,000 (full capital)
```

---

## 6. Monitoring & Maintenance

### Daily Checks (5 minutes)

```bash
# 1. Check Discord notifications
# Look for: alerts, trade confirmations, daily summaries

# 2. View live trader status
tail -20 logs/live_trader.log

# 3. Check today's performance
tail -20 live_trades/coinbase_prod_v1.csv

# 4. View maker ratio (should be >50%)
tail -1 live_trades/coinbase_prod_v1.csv | awk -F',' '{print $11}'

# 5. Check risk limits not hit
grep "HALTED" logs/live_trader.log
```

### Weekly Maintenance (30 minutes)

```bash
# 1. Calculate weekly Sharpe
python -c "
import pandas as pd
df = pd.read_csv('live_trades/coinbase_prod_v1.csv')
returns = df['portfolio_value'].pct_change()
sharpe = returns.mean() / returns.std() * (252**0.5)
print(f'Weekly Sharpe: {sharpe:.4f}')
"

# 2. Review conviction statistics
# Check logs for conviction filter pass rate

# 3. Check fee tier progress
# View in logs or Coinbase dashboard

# 4. Archive logs
tar -czf logs_archive_$(date +%Y%m%d).tar.gz logs/
```

### Monthly Maintenance (1-2 hours)

```bash
# 1. Full performance analysis
python monitoring/analyze_live_performance.py \
    --csv live_trades/coinbase_prod_v1.csv

# 2. Model performance review
# Compare live Sharpe to backtest
# Check if degradation threshold near

# 3. Consider retraining
# If live performance degraded >50%
# Retrain with recent 6mo data including live period

# 4. Fee tier optimization
# Check current tier vs target
# Adjust maker preference if needed

# 5. Capital reallocation
# If consistently profitable, increase capital
# If degrading, reduce or halt
```

---

## 7. Troubleshooting

### Training Failures Still High (>30%)

**Check model registry:**
```sql
SELECT substr(error_message, 1, 100) as error, COUNT(*)
FROM models
WHERE status='failed'
GROUP BY substr(error_message, 1, 100);
```

**Common issues:**
- OOM errors → Reduce batch size, increase RAM, use smaller model
- NaN/Inf values → Check hyperparameter ranges, add gradient clipping
- Timeout → Increase training time limit, reduce episodes

### Conviction Filter Too Strict (Few Trades)

```python
# Lower minimum score
filter = ConvictionFilter(min_score=0.5)  # Instead of 0.6

# Reduce weight on strict factors
filter = ConvictionFilter(
    sharpe_momentum_weight=0.1,  # Less emphasis on recent performance
    volatility_filter_weight=0.05  # Less emphasis on volatility
)
```

### Conviction Filter Too Loose (Too Many Trades)

```python
# Raise minimum score
filter = ConvictionFilter(min_score=0.7)

# Increase weight on strict factors
filter = ConvictionFilter(
    position_size_weight=0.4,  # Only strong signals
    ensemble_agreement_weight=0.3  # Require high agreement
)
```

### Maker Ratio Too Low (<50%)

**Possible causes:**
1. Orders not filling → Widen limit price spread (0.1% instead of 0.05%)
2. Volatile market → Increase timeout (180s instead of 120s)
3. Large positions → Split into multiple smaller orders
4. Wrong timeframe → 5min moves too fast, consider 15min

**Fixes:**
```python
# Adjust limit order placement
if side == 'buy':
    limit_price = current_price * 0.999  # 0.1% below (was 0.05%)
else:
    limit_price = current_price * 1.001  # 0.1% above

# Increase timeout
filled = self._wait_for_fill(order_id, timeout=180)  # Was 120

# Higher maker preference
trader = CoinbaseLiveTrader(maker_preference=0.8, ...)  # Was 0.7
```

### Live Performance Degrading

**Immediate actions:**
1. Check Discord alerts from monitor
2. Review recent trades for anomalies
3. Check if market regime changed
4. Verify data quality (missing bars, stale prices)

**If Sharpe drops below -1.0:**
1. Monitor will send critical alert
2. Manually review last 20 trades
3. Consider halting if losses persist
4. Retrain with recent data

**If Sharpe drops below -2.0:**
1. Monitor will auto-stop after 24 hours
2. Don't override unless you have good reason
3. Investigate root cause before redeploying
4. Possibly retrain or switch to backup model

### Coinbase API Errors

**Rate limiting (429 errors):**
- Increase poll interval (600s instead of 300s)
- Reduce number of tickers
- Use cached candle data

**Authentication errors (401):**
- Verify API key/secret in `.env`
- Check API key permissions (trade permission required)
- Regenerate API key if compromised

**Order errors (insufficient funds, invalid size):**
- Check minimum order sizes per ticker
- Verify cash balance before trades
- Round quantities to proper precision

---

## 8. Performance Targets

### Training Phase (Robust Wrapper)
- ✅ Trial completion rate: >70% (was 4-50%)
- ✅ Checkpoint saves: Every 100 episodes
- ✅ Model registry: Track all attempts
- ✅ Retry on OOM: 3 attempts per trial

### Paper Trading (Validation)
- ✅ Sharpe ratio: >0.1 (vs backtest)
- ✅ Max drawdown: <5%
- ✅ Trade frequency: 2-5 trades/day
- ✅ No critical alerts: 48+ hours

### Live Trading (Production)
- ✅ Sharpe ratio: >0.15
- ✅ Maker ratio: >70%
- ✅ Daily loss limit: Never hit (<-2%)
- ✅ Monthly trades: 75-100 (conviction filtered)
- ✅ Fee costs: <0.3% of capital/month

### Capital Growth (Pure Compounding)
- 📈 Target: 4% monthly return
- 📈 Starting: $1,000 → $3,000
- 📈 Timeline: 5-7 years to $40-50k
- 📈 Risk: Maximum 2% daily drawdown

---

## Summary

**You now have:**
1. ✅ **Robust training** - Fix 50-96% failure rate
2. ✅ **Conviction filtering** - Reduce 300+ trades to 75-100
3. ✅ **Coinbase integration** - Live trading with maker optimization
4. ✅ **Performance monitoring** - Real-time alerts and auto-stop

**Deployment path:**
1. Train with robust wrapper (GPU, 2-4 days)
2. Validate on paper trading (48-72 hours)
3. Deploy $1k to Coinbase live (1 week)
4. Scale to $2k (2 weeks) then $3k (ongoing)
5. Maintain 5-10 hours/week

**Maintenance mode:**
- Daily: 5min Discord check + log review
- Weekly: 30min performance analysis
- Monthly: 1-2hr deep review + retraining decision

**You're ready to deploy!** 🚀
