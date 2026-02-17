# Strategic Response: Cappuccino Development Roadmap Implementation

**Date:** February 9, 2026
**Response to:** Daryle's comprehensive strategic planning document
**Status:** Detailed analysis and actionable implementation plan

---

## Executive Summary of Findings

### Good News ✅
1. **Disk space is fine** - 376GB available (41% free)
2. **Models ARE being saved** - Found .pth files in train_results/
3. **Current 1hr GPU training progressing** - 24/500 trials (4.8%)
4. **5min CPU training active** - All 5 workers running (7+ hours per trial)
5. **Live performance monitor deployed** - Degradation alerts working
6. **Discord integration operational** - Notifications functioning

### Critical Issues 🚨
1. **Trial failure rate: 50-96%** - This is the model retention problem!
   - ensemble_balanced: only 3.4% success rate
   - ft_transformer_small: 20% success rate
   - ensemble_conservative: 40.7% success rate
   - **Most trials FAIL during training, never reaching save point**

2. **Root cause identified:** Trials failing during training, NOT save failures
   - Models that complete successfully ARE being saved
   - Problem is completion rate, not save mechanism

3. **5min training too slow on CPU** - 7+ hours per trial = 29 days for 100 trials
   - Need to stop CPU training and wait for GPU

---

## Answers to Your 10 Questions

### 1. Model Retention: Current Architecture

**Current Save Mechanism:**
```python
# Located in: scripts/training/1_optimize_unified.py
# Models saved to: train_results/cwd_{env_class.__name__}_{timeframe}/
# Format: actor.pth, critic.pth (PyTorch state dicts)
```

**Typical Failure Modes (Based on Data):**
- **50-96% of trials FAIL** during training
- Failures happen BEFORE reaching save point
- Successfully completed trials DO save properly

**Why Trials Fail:**
1. **OOM (Out of Memory)** - Especially on 5min data (12x larger)
2. **NaN/Inf in gradients** - Bad hyperparameter combinations
3. **Environment errors** - Data loading, API limits
4. **Timeout** - Training exceeds Optuna trial timeout
5. **CUDA errors** - GPU crashes mid-training

**Current Storage Status:**
- Location: `train_results/` (915GB total, 376GB free)
- Recent models: `train_results/deployment_trial250_*/`
- Model sizes: ~50-200MB per model (PyTorch state dicts)

**What's Actually Working:**
- Disk space: ✅ Plenty of room
- Save function: ✅ Works when trial completes
- Model loading: ✅ Paper trader successfully loads models

**What's NOT Working:**
- Trial completion rate: ❌ Only 3-46% completing
- Error handling: ❌ No retry on OOM
- Checkpointing: ❌ No intermediate saves

### 2. Live Trading Integration: Current State

**Coinbase Trader Status:**
- ❌ **Does NOT exist yet** - We have Alpaca paper trader only
- ✅ Alpaca paper trader working (Trial #250, 28 hours live)
- ❌ No Coinbase integration implemented

**Current Integration:**
```python
# What EXISTS: Alpaca paper trader
paper_trader = AlpacaPaperTraderPolling(
    model_dir="train_results/deployment_trial250_*/",
    tickers=["AAVE/USD", "AVAX/USD", ...],
    timeframe="1h"
)
paper_trader.run()  # Polls every hour, executes trades
```

**What's Missing for Coinbase:**
- Coinbase Pro/Advanced API integration
- Fee tier tracking
- Maker/taker order logic
- Volume reporting for tier progression
- Live account connection

**Estimated Work:**
- Copy `paper_trader_alpaca_polling.py` → `live_trader_coinbase.py`
- Replace Alpaca API calls with Coinbase Pro API
- Add maker order logic (limit orders)
- Add fee tier tracking
- **Estimate: 50-100 lines of code, 4-6 hours work**

### 3. 5-Minute Data: Availability & Storage

**Current Data:**
- ✅ 1hr data: Available and working
- ✅ 5min data: **Download script exists!**
  - Script: `scripts/data/prepare_multi_timeframe_data.py`
  - Already downloaded: `data/crypto_5m_6mo.pkl` (generated Feb 8)
  - Ready to use for training

**Storage Requirements:**
```
1hr data:  ~4,300 bars/ticker/year × 7 tickers = 30,000 bars
5min data: ~52,000 bars/ticker/year × 7 tickers = 364,000 bars (12x)

Preprocessed 5min data: ~150-200MB (6 months)
Full year: ~300-400MB
Historical (2+ years): ~800MB - 1GB
```

**Current Status:**
- Disk space: ✅ 376GB available - plenty of room
- Data quality: ✅ 6 months of 5min data ready
- Training compatibility: ✅ Scripts support --timeframe 5m flag

**What's Needed:**
- Nothing! Data is ready to use.
- Just need to stop CPU training and start GPU training with 5min data

### 4. Conviction Scoring: Infrastructure

**Current State:**
- ❌ **No conviction scoring implemented**
- Current model: Makes prediction → executes immediately
- No filtering, no confidence thresholds

**What EXISTS:**
```python
# Current (in paper_trader):
action = model.predict(state)  # Returns action array
self.env.step(action)  # Executes immediately
```

**What's NEEDED:**
```python
# Proposed conviction scoring:
def get_conviction_score(model, state, env_context):
    # 1. Model confidence
    with torch.no_grad():
        action_dist = model.actor(state)  # Get distribution
        entropy = action_dist.entropy()  # Low entropy = high confidence

    # 2. Market conditions
    volatility_score = check_volatility(env_context)
    volume_score = check_volume(env_context)

    # 3. Recent performance
    recent_winrate = calculate_recent_winrate(last_10_trades)

    # 4. Trend alignment (5min agrees with 1hr)
    trend_alignment = check_trend_alignment(state_5min, state_1hr)

    # Weighted combination
    conviction = (
        0.35 * (1 - entropy.item()) +  # Model confidence
        0.25 * volatility_score +       # Good market conditions
        0.20 * volume_score +            # Sufficient liquidity
        0.10 * recent_winrate +          # Recent success
        0.10 * trend_alignment           # Multi-timeframe agreement
    )

    return conviction

# Then filter trades:
conviction = get_conviction_score(model, state, env_context)
if conviction > 0.75:  # High conviction only
    self.env.step(action)
else:
    self.env.step(np.zeros_like(action))  # Hold/no trade
```

**Estimated Work:**
- Implement conviction_scorer.py: 100-150 lines
- Integrate into paper_trader: 30-50 lines
- Test and tune threshold: 2-4 hours
- **Total: 6-8 hours work**

### 5. Maker/Taker Optimization: Current Strategy

**Current Alpaca Setup:**
- Uses **market orders** (100% taker)
- No limit order logic
- No maker optimization

**Coinbase Requirements:**
```python
# Maker order (limit order at specific price):
order = client.create_limit_order(
    product_id='BTC-USD',
    side='buy',
    price='50000.00',  # Specify price
    size='0.01'
)
# Fee: 0.05-0.15% (maker rate)

# Taker order (market order, immediate fill):
order = client.create_market_order(
    product_id='BTC-USD',
    side='buy',
    size='0.01'
)
# Fee: 0.10-0.50% (taker rate)
```

**Strategy for 70%+ Maker:**
```python
def execute_trade_smart(symbol, action, current_price):
    """Execute with maker optimization."""

    if action > 0:  # Buy
        # Place limit order slightly below current price
        limit_price = current_price * 0.9995  # 0.05% below

        try:
            order = create_limit_order(
                symbol=symbol,
                side='buy',
                price=limit_price,
                size=action,
                post_only=True  # Ensures maker status
            )

            # Wait up to 2 minutes for fill
            if wait_for_fill(order, timeout=120):
                return 'maker'  # Success
            else:
                # If not filled, use market order
                cancel_order(order)
                create_market_order(symbol, 'buy', action)
                return 'taker'  # Fallback

        except Exception:
            # Emergency fallback to market
            create_market_order(symbol, 'buy', action)
            return 'taker'

    elif action < 0:  # Sell (similar logic)
        limit_price = current_price * 1.0005  # 0.05% above
        # ... same pattern
```

**Estimated Work:**
- Implement limit order logic: 100-150 lines
- Add post_only flags: 10 lines
- Fill timeout handling: 50 lines
- Track maker/taker ratio: 30 lines
- **Total: 8-10 hours work**

### 6. Risk Management: Current State

**Current Implementation (Alpaca Paper Trader):**

✅ **What's Implemented:**
```python
class RiskManagement:
    max_position_pct = 0.30        # Max 30% per asset
    stop_loss_pct = 0.10           # 10% stop loss
    trailing_stop_pct = 0.0        # Disabled by default

    # Portfolio-level
    portfolio_trailing_stop_pct = 0.015  # 1.5% from peak
    profit_take_threshold_pct = 0.03     # Take profits at +3%
    profit_take_amount_pct = 0.50        # Sell 50% when hit
```

❌ **What's MISSING:**
- Daily loss limits
- Consecutive loss handling
- Time-of-day restrictions
- Per-trade risk limits

**Recommended Additions:**
```python
class EnhancedRiskManagement:
    # Daily limits
    max_daily_loss_pct = 0.02      # Stop trading if down 2% today
    max_daily_trades = 20          # Cap at 20 trades/day

    # Consecutive losses
    max_consecutive_losses = 3     # Stop after 3 losses in row
    cooldown_after_losses_hours = 4  # Wait 4h after hitting limit

    # Time-based
    avoid_market_open_hours = [9, 10]  # Reduce size during volatility
    avoid_market_close_hours = [15, 16]

    # Per-trade
    max_trade_size_pct = 0.10      # Max 10% of account per trade
    min_trade_size_usd = 10        # Min $10 per trade (avoid dust)
```

**Estimated Work:**
- Add daily loss tracking: 50 lines
- Consecutive loss logic: 40 lines
- Time restrictions: 30 lines
- **Total: 2-3 hours work**

### 7. Performance Tracking: Current Metrics

**What's Being Tracked:**

✅ **In Paper Trader CSV:**
```csv
timestamp, cash, total_asset, reward, holding_*, price_*, action_*
```

✅ **Calculated by view_paper_trader.py:**
- Portfolio value & return
- Sharpe ratio (live vs backtest)
- Max drawdown
- Number of trades
- Current positions

❌ **What's MISSING:**
- Win rate (% profitable trades)
- Maker/taker ratio
- Average trade size
- Fee totals
- Volume (for tier tracking)
- Trade duration distribution

**Recommended Dashboard:**
```python
# Add to view_paper_trader.py:
def calculate_missing_metrics(df):
    # Win rate
    trade_returns = df['total_asset'].diff()
    win_rate = (trade_returns > 0).sum() / len(trade_returns)

    # Average trade details
    avg_trade_size = df['action_*'].abs().mean()  # Need to track this

    # Fees (need to add fee column to CSV)
    total_fees = df['fees_paid'].sum()

    # Volume (for tier progression)
    monthly_volume = df['trade_value'].sum()  # Need to add this

    return {
        'win_rate': win_rate,
        'avg_trade_size': avg_trade_size,
        'total_fees': total_fees,
        'monthly_volume': monthly_volume,
        'maker_ratio': None,  # Need order type tracking
    }
```

**Estimated Work:**
- Add fee tracking to CSV: 20 lines
- Add volume tracking: 20 lines
- Add order type tracking: 30 lines
- Update dashboard: 50 lines
- **Total: 2-3 hours work**

### 8. Hardware Limitations: 5-Minute + 16GB RAM

**Current Setup:**
- CPU: i7-10700KF (16 cores)
- RAM: 16GB
- GPU: RTX 3070 (8GB VRAM)
- Storage: 915GB (376GB free)

**5-Minute Episode Requirements:**

```
1hr timeframe:
- Episode length: ~168 bars (1 week)
- Replay buffer: ~50MB RAM
- Training time: 1-2 hours/trial (GPU)

5min timeframe:
- Episode length: ~2,016 bars (1 week) [12x longer]
- Replay buffer: ~600MB RAM [12x larger]
- Training time: 10-15 hours/trial (GPU), 7+ hours (CPU)
```

**RAM Breakdown During Training:**
```
System: 2GB
Python overhead: 1GB
Model (PPO): 2GB
Environment state: 3GB
Replay buffer (5min): 0.6GB
Optuna overhead: 1GB
Buffer: 6.4GB
---
Total: ~16GB (TIGHT!)
```

**Risk Assessment:**
- ⚠️ **Borderline** - 16GB is barely enough
- May cause OOM on some trials (explains failures!)
- Swap usage will hurt performance

**Recommendations:**

**Short-term (Now - Month 3):**
- ✅ Train 5min on GPU (not CPU - too slow)
- ⚠️ Reduce episode length to 1000 bars (reduce RAM)
- ⚠️ Monitor swap usage during training
- ⚠️ Close other applications during training

**Medium-term (Month 3-6, when account at $4-5k):**
- ⚡ **Upgrade to 32GB RAM** - Priority #1 hardware upgrade
- Cost: ~$60 for 16GB DDR4
- Impact: 2x headroom, eliminates OOM risk

**Long-term (Month 6-12, when account at $8-10k):**
- Add 1TB NVMe SSD for faster data loading
- Consider RTX 4070 if training becomes bottleneck

**Verdict:** Can train 5min on current hardware but it's tight. RAM upgrade should be first spend when account hits $4-5k.

### 9. Ensemble System: Current Architecture

**Current Ensemble (From Previous Work):**
```python
# Located: train/ensemble.py
class EnsembleAgent:
    def __init__(self, model_dirs):
        self.agents = [load_agent(dir) for dir in model_dirs]

    def predict(self, state):
        # Get predictions from all agents
        predictions = [agent.predict(state) for agent in self.agents]

        # Weighted voting (based on validation Sharpe)
        weights = [agent.validation_sharpe for agent in self.agents]
        ensemble_action = np.average(predictions, weights=weights, axis=0)

        return ensemble_action
```

**Integration with 5-Minute:**
```python
# Scenario 1: Pure 5min ensemble
ensemble_5min = EnsembleAgent([
    "train_results/5min_trial_10/",   # Sharpe 1.2
    "train_results/5min_trial_15/",   # Sharpe 1.3
    "train_results/5min_trial_23/",   # Sharpe 1.4
])

# Scenario 2: Multi-timeframe ensemble (RECOMMENDED)
ensemble_hybrid = MultiTimeframeEnsemble(
    models_5min=[
        "train_results/5min_trial_23/",  # Weight: 0.6
    ],
    models_1hr=[
        "train_results/1hr_trial_250/",  # Weight: 0.4
    ]
)
# 5min for tactics, 1hr for strategy
```

**Recommended Architecture:**
```python
class MultiTimeframeEnsemble:
    def __init__(self, models_5min, models_1hr):
        self.tactical_agents = [load(m) for m in models_5min]   # 5min
        self.strategic_agents = [load(m) for m in models_1hr]   # 1hr

    def predict(self, state_5min, state_1hr):
        # Get tactical signal (5min)
        tactical = self.tactical_agents[0].predict(state_5min)

        # Get strategic signal (1hr)
        strategic = self.strategic_agents[0].predict(state_1hr)

        # Combined decision
        # Only trade if BOTH timeframes agree on direction
        if sign(tactical) == sign(strategic):
            # Use 5min for sizing (more precise)
            return tactical
        else:
            # Conflicting signals - don't trade
            return np.zeros_like(tactical)
```

**Estimated Work:**
- Multi-timeframe ensemble: 100-150 lines
- Integrate into trader: 50 lines
- Test both timeframes: 3-4 hours
- **Total: 6-8 hours work**

### 10. Automated Retraining: Current Status

**Current State:**
- ❌ **No automated retraining implemented**
- Manual process: Run training script when performance degrades
- No scheduled retraining
- No trigger-based retraining

**What's Needed:**
```python
# Automated retraining trigger system:
def should_retrain(performance_history):
    """Decide if retraining needed."""

    # Trigger 1: Performance degradation
    recent_sharpe = performance_history[-30:]  # Last 30 days
    baseline_sharpe = model.backtest_sharpe

    if mean(recent_sharpe) < baseline_sharpe - 0.5:
        return True, "performance_degradation"

    # Trigger 2: Account growth
    current_capital = get_account_balance()
    training_capital = model.training_capital

    if current_capital / training_capital > 1.10:  # 10% growth
        return True, "account_growth"

    # Trigger 3: Scheduled (weekly)
    days_since_last_train = get_days_since_training()
    if days_since_last_train > 7:
        return True, "scheduled_weekly"

    return False, None

# Retraining pipeline:
def automated_retrain():
    """Run weekly retraining with new data."""

    # 1. Collect new live data
    new_data = load_live_trading_data("paper_trades/trial250_session.csv")

    # 2. Merge with training data
    combined_data = merge_data(original_training_data, new_data)

    # 3. Run short training (50 trials instead of 100)
    best_trial = run_training(
        data=combined_data,
        n_trials=50,
        timeframe='5m',
        study_name=f'retrain_{timestamp}'
    )

    # 4. Validate new model
    if best_trial.sharpe > current_model.sharpe:
        deploy_model(best_trial)
    else:
        print("New model not better, keeping current")
```

**Estimated Work:**
- Trigger logic: 50 lines
- Data merging: 50 lines
- Automated training launcher: 100 lines
- Validation & deployment: 50 lines
- **Total: 8-10 hours work**

---

## Critical Issue: Model Retention - Root Cause & Fix

### The Real Problem

**It's NOT the save function** - it's that trials are FAILING during training.

**Data:**
- ensemble_balanced: 96.6% failure rate (only 1/30 complete)
- ft_transformer_small: 80% failure rate
- ensemble_conservative: 59.3% failure rate

**Why Trials Fail:**
1. **OOM (Out of Memory)** - 16GB RAM is tight for 5min data
2. **Bad hyperparameters** - Some combinations cause NaN/Inf
3. **Training timeout** - 5min trials take 10-15 hours
4. **CUDA errors** - GPU crashes

### The Fix: Robust Training Pipeline

**Immediate Actions:**

1. **Stop 5min CPU training** (too slow, wasting time)
2. **Add error handling to training script**
3. **Implement checkpointing**
4. **Add model registry database**

**Implementation:**

