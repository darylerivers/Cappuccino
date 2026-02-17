# Trading Performance Monitoring Guide

## Current Situation

**Ensemble Started:** Dec 17, 22:25 (1.7 hours ago)
**Trading Data:** 1 hour (insufficient for meaningful analysis)
**Status:** Need to wait for data accumulation

---

## Timeline for Performance Evaluation

### Hour 1-6: Initial Setup ✓ (You are here)
- Ensemble loads and makes initial trades
- Very limited data (1-6 hours)
- P&L will fluctuate wildly
- **NOT ENOUGH DATA TO JUDGE**

### Hour 6-24: Early Pattern Formation
- 6-24 hours of trading
- Start to see trading patterns
- Can identify if agent is making reasonable decisions
- Still too early for statistical significance

### Hour 24-48: First Meaningful Comparison ⭐
- Full day+ of trading
- Can compare vs buy-and-hold
- Alpha calculation becomes meaningful
- Can see if agent adapts to market conditions

### Day 3-7: Trend Confirmation
- Week of trading data
- Sharpe ratio becomes reliable
- Win rate and trade statistics meaningful
- Can evaluate if ensemble is working as expected

### Week 2+: Full Performance Assessment
- Multiple weeks of data
- Account for different market conditions
- Evaluate consistency
- Compare to backtest Sharpe (0.14-0.15)

---

## What You Can See Right Now

### 1. Current Status (Real-time)
```bash
python show_current_trading_status.py
```
Shows:
- Current positions
- P&L since start
- Recent trades
- Portfolio value

### 2. Performance Comparison (vs Market)
```bash
python compare_trading_performance.py
```
Shows:
- Agent return vs buy-and-hold
- Alpha calculation
- Hourly performance breakdown
- Individual asset performance

### 3. Dashboard (Full View)
```bash
python dashboard.py
```
Pages:
- **Page 2:** Live ensemble trading status
- **Page 4:** Trade history (will grow over time)
- **Page 5:** Performance metrics and charts

### 4. Live Monitoring
```bash
# Watch trading activity as it happens
tail -f logs/paper_trading_BEST_console.log

# Watch trading log file
tail -f logs/paper_trading_BEST.log
```

---

## Performance Metrics to Watch

### Short-term (24-48 hours)
- **Alpha vs Market:** Is agent beating buy-and-hold?
- **Trade Frequency:** How often is it trading?
- **Position Duration:** How long does it hold positions?
- **Win Rate:** % of profitable trades

### Medium-term (1-2 weeks)
- **Sharpe Ratio:** Risk-adjusted returns
  - Target: 0.14-0.15 (matching backtest)
- **Maximum Drawdown:** Largest peak-to-trough loss
  - Should trigger stop-loss at 10% per position
- **Consistency:** Steady alpha or erratic?

### Long-term (1+ month)
- **Annual Return:** Extrapolated yearly performance
- **Volatility:** Standard deviation of returns
- **Benchmark Comparison:** vs BTC, vs equal-weighted portfolio
- **Market Condition Adaptation:** Performance in bull/bear/sideways

---

## What to Expect from the Ensemble

### Best-Case Scenario
Based on backtest Sharpe of 0.14-0.15:
- **Expected Annual Return:** 10-15% above market
- **Monthly Alpha:** +0.8% to +1.2%
- **Daily Alpha:** +0.03% to +0.04%

### Realistic Expectations
- **First Week:** Highly variable, could be +/- 5%
- **First Month:** Should start trending toward positive alpha
- **Quarter 1:** Should establish consistent outperformance

### Risk Management
The ensemble has:
- **Stop-loss:** 10% per position
- **Trailing stop:** 8% from high water mark
- **Portfolio protection:** 1.5% trailing stop from peak
- **Max position:** 30% per asset

If any of these are violated, CHECK IMMEDIATELY.

---

## Red Flags to Watch For

### 🚨 Immediate Action Required
- Stop-loss not triggering (position down > 10%)
- Portfolio down > 20% from start
- Trading stopped (no new bars for 2+ hours)
- Memory usage > 80%

### ⚠️ Monitor Closely
- Alpha consistently negative for 48+ hours
- Win rate < 35% (expected: 40-45%)
- Position concentration > 40% in single asset
- Frequent position changes (> 10 trades/hour)

### ✓ Normal Behavior
- Daily P&L swings +/- 2-3%
- 1-3 trades per hour
- Some losing days (expected)
- Alpha fluctuating around 0% in first 24h

---

## Comparison Baselines

### What to Compare Against

1. **Buy-and-Hold BTC**
   - Simplest baseline
   - Should beat this to justify active trading

2. **Equal-Weighted Portfolio**
   - All 7 coins, equal allocation
   - Better baseline than single-asset

3. **Previous Ensemble** (trial_9698)
   - Sharpe: 0.009 (BAD)
   - Currently down 32% vs market
   - New ensemble should VASTLY outperform

4. **Backtest Performance** (top 20 models)
   - Mean Sharpe: 0.1435
   - This is the target to match

---

## Action Items for Next 48 Hours

### Hour 6 (Dec 18, 04:25)
- ✓ Check: `python show_current_trading_status.py`
- Look for: 5-6 hours of data
- Expect: Some trades executed

### Hour 12 (Dec 18, 10:25)
- ✓ Check: `python compare_trading_performance.py`
- Look for: 12 hours of data, trading pattern emerging
- Expect: Alpha starting to show (could be +/- 2%)

### Hour 24 (Dec 18, 22:25)
- ✓ Check: Full dashboard `python dashboard.py` → Page 5
- Look for: 24 hours of data, performance trend
- Expect: Alpha more stable, can compare to backtest

### Hour 48 (Dec 19, 22:25)
- ✓ Check: Performance comparison and trade history
- Look for: Consistent alpha, win rate ~40-45%
- Decide: Is ensemble performing as expected?

---

## How to Extract Insights

### From Trade Log
```bash
# Count total trades
wc -l logs/paper_trading_BEST.log

# View recent trades
tail -20 logs/paper_trading_BEST.log

# Extract specific ticker trades
grep "action_BTC" logs/paper_trading_BEST.log
```

### From Trade History Analyzer
```bash
python trade_history_analyzer.py
```
This shows:
- All individual trades
- Completed round-trip trades with P&L
- Win/loss statistics
- Best/worst trades
- Total fees paid

### From Dashboard
```bash
python dashboard.py
```
Navigate:
- **Page 2:** Current ensemble status
- **Page 4:** Trade history table
- **Page 5:** Performance charts and metrics

---

## Current File Locations

| Data | Location |
|------|----------|
| Trading log (CSV) | `logs/paper_trading_BEST.log` |
| Console output | `logs/paper_trading_BEST_console.log` |
| Current positions | `paper_trades/positions_state.json` |
| Ensemble manifest | `train_results/ensemble_best/ensemble_manifest.json` |
| Performance monitor | `logs/performance_monitor.log` |

---

## Summary

**Right now:** You have 1 hour of data - not enough to judge anything.

**Next 24 hours:** Monitor accumulation, look for reasonable trading behavior.

**At 48 hours:** Meaningful performance comparison possible.

**At 1 week:** Can evaluate if ensemble matches backtest performance.

**Be patient!** The ensemble has top models (Sharpe 0.14-0.15), but needs time to prove itself in live trading.

---

**Quick Check (Run every 6-12 hours):**
```bash
python compare_trading_performance.py
```

This will show you exactly where you stand vs the market.
