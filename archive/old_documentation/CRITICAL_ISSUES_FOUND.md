# CRITICAL ISSUES - Dec 18, 2025

## Issue 1: Ensemble OVERTRADING and UNDERPERFORMING ⚠️

### Performance (13 hours)
- **Agent Return:** -3.37%
- **Market Return:** +3.19%
- **Alpha:** -6.56% (TERRIBLE)

### Trading Pattern
- **Total trades:** 89 in 13 hours (6.8 trades/hour)
- **Sells:** 71
- **Buys:** 18
- **Pattern:** Constantly churning between AAVE ↔ LINK/UNI every 1-2 hours

### Example Behavior
```
Hour 05: Sell everything → Buy AAVE
Hour 06: Sell AAVE → Buy LINK/UNI
Hour 07: Sell LINK/UNI → Buy AAVE
Hour 08: Sell AAVE → Buy LINK/UNI
... (repeats)
```

### Root Cause Analysis
1. **Overfitting:** Models trained on historical data, reacting to noise in live market
2. **No position holding:** Models don't "wait" - they trade every hour
3. **Transaction costs:** Even at 0.25% fees, churning kills profits
4. **Possible ensemble issue:** All 20 models might be making similar mistakes

### Backtest vs Live
| Metric | Backtest | Live (13h) |
|--------|----------|------------|
| Sharpe | 0.14-0.15 | Negative |
| Alpha | +10-15% annually | -6.56% |
| Trades/hour | ??? | 6.8 |

**The models that looked great in backtest are failing in live trading.**

---

## Issue 2: Two-Phase Training NOT Running ❌

### Status
- **Phase 1:** Complete (Dec 17)
- **Phase 2:** Never started (phase2_results: null)
- **Dashboard:** Shows stale data from Dec 17
- **Scheduler:** Waiting for Sunday Dec 21, 2:00 AM

### Why This Happened
- Two-phase is configured for **WEEKLY** schedule
- Only runs Sunday 2AM
- No manual training has been run

### Impact
- No new models being trained
- System using OLD models from weeks ago
- Models getting staler by the day

---

## Issue 3: Old Training Workers Running (FIXED ✓)

### What Was Wrong
```
PID 5284, 5351, 5429 - Running for 16+ hours
Study: cappuccino_week_20251206 (OLD)
```

### Fixed
- Killed all three processes
- Freed ~1.5GB memory
- Memory usage: 6.4GB / 15GB (43%)

---

## Issue 4: Dashboard Showing Stale Data

### Two-Phase Training Page (Page 9)
- Shows "2 running" but nothing is running
- Shows "3/200 trials" from old test
- Last updated: Dec 17, 00:01 (40+ hours ago)

### Cause
- Reading from checkpoint file (two_phase_checkpoint.json)
- Checkpoint not being updated
- No active training to update it

---

## Root Cause: Model Overfitting

The fundamental problem is **the models are overfitted to historical data**.

### Evidence
1. **Backtest Sharpe:** 0.14-0.15 (excellent)
2. **Live Sharpe:** Negative (terrible)
3. **Trading pattern:** Erratic, reacting to noise
4. **No consistency:** Switches positions every hour

### Why Overfitting Happened
- Models trained on specific market conditions (Oct-Nov 2025)
- Optimized hyperparameters for past data
- Market regime may have changed (Dec 2025 is different)
- Models lack generalization

### Classic ML Problem
```
Training Data → Great performance
Validation Data → Great performance
Live Market → FAILS
```

This is the #1 problem in algorithmic trading.

---

## Recommended Actions

### Immediate (Next 2 hours)

1. **Stop paper trading ensemble temporarily**
   - It's losing money (-6.56% alpha)
   - Need to diagnose before continuing
   - Command: Find PID and kill

2. **Investigate ensemble loading**
   - Check if all 20 models actually loaded
   - Verify model files exist
   - Check for loading errors in logs

3. **Test single best model (Trial 686)**
   - Deploy just Trial 686 (Sharpe 0.1566)
   - Compare performance vs ensemble
   - Might perform better than ensemble

### Short-term (Next 24 hours)

4. **Run Phase 2 training manually**
   - Don't wait until Sunday
   - Train on recent data (last 7-14 days)
   - Get fresh models adapted to current market

5. **Add trading constraints**
   - Max trades per day (e.g., 10)
   - Min holding period (e.g., 2 hours)
   - Transaction cost penalty in reward function

6. **Backtest on recent data**
   - Test top models on Dec 1-17 data
   - See if they would have performed well
   - If not, models are too old

### Medium-term (Next week)

7. **Implement ensemble voting threshold**
   - Only trade if 75%+ models agree
   - Reduces false signals
   - Less churning

8. **Add regime detection**
   - Detect when market conditions change
   - Switch to cash mode in uncertain periods
   - Only trade when confident

9. **Weekly retraining (automated)**
   - Keep models fresh with recent data
   - Rolling window training
   - Auto-deploy only if beats current ensemble

---

## Questions to Answer

1. **Are all 20 models actually loading?**
   - Check logs for "Loaded model X/20"
   - Verify 20 actor.pth files exist

2. **What do individual models predict?**
   - Log each model's action
   - See if they disagree or all make same mistake

3. **Is the ensemble voting working?**
   - Check ensemble logic (mean? majority?)
   - Possible bug in averaging

4. **Why so much churning?**
   - Models might have very low confidence threshold
   - Every small price movement triggers trade

5. **Can we add a "do nothing" action?**
   - Currently: Models must pick buy/sell/hold weights
   - Maybe need explicit "no trade" option

---

## Next Steps (Priority Order)

### Priority 1: Stop the Bleeding
```bash
# Find and stop paper trader
ps aux | grep paper_trader | grep ensemble_best
kill <PID>
```

### Priority 2: Diagnose Ensemble
```bash
# Check model loading
grep -i "model" logs/paper_trading_BEST_console.log | head -30

# Verify model files exist
ls -la train_results/ensemble_best/model_*/actor.pth | wc -l
```

### Priority 3: Test Single Model
```bash
# Deploy Trial 686 alone
python paper_trader_alpaca_polling.py \
  --model-dir train_results/cwd_tests/trial_686_1h \
  --tickers BTC/USD ETH/USD LTC/USD BCH/USD LINK/USD UNI/USD AAVE/USD \
  --timeframe 1h \
  --poll-interval 60 \
  --log-file logs/paper_trading_trial686.log
```

### Priority 4: Manual Training
```bash
# Don't wait for Sunday - train NOW on recent data
python run_two_phase_training.py --mode mini
```

---

## Learnings

1. **Backtest performance ≠ Live performance**
   - Models with Sharpe 0.14 are losing money live
   - Overfitting to training data

2. **Ensemble doesn't always help**
   - If all models overfit similarly, ensemble fails too
   - "Wisdom of crowds" only works if crowds are wise

3. **Need continuous retraining**
   - Models get stale quickly (weeks, not months)
   - Market conditions change
   - Weekly retraining is ESSENTIAL

4. **Transaction costs matter**
   - 6.8 trades/hour is excessive
   - Need to reduce trading frequency

5. **Live testing reveals truth**
   - This is why we paper trade first
   - Discovered problem before real money

---

## Status Summary

✓ **Fixed:** Old training workers killed, memory freed
⚠️ **Active Problem:** Ensemble losing money (-6.56% alpha)
⚠️ **Active Problem:** Models overfitted, churning positions
❌ **Not Fixed:** Two-phase training not running (weekly schedule)
❌ **Not Fixed:** Dashboard showing stale data

**RECOMMENDATION:** Pause paper trading, diagnose ensemble, train fresh models.
