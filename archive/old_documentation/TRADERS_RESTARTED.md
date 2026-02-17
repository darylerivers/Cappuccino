# Paper Traders Restarted with Concentration Fix ✅

**Date:** January 17, 2026, 05:14 UTC
**Status:** Both traders running with fix applied

---

## 🚀 Running Traders

### Ensemble Trader
- **PID:** 1105329
- **Model:** Top-20 ensemble
- **Log:** `logs/ensemble_fixed.log`
- **CSV:** `paper_trades/ensemble_fixed_20260117.csv`
- **Started:** 2026-01-17 05:13 UTC
- **Status:** ✅ Running

### Single Model Trader (Trial #861)
- **PID:** 1105587
- **Model:** Trial #861 (Sharpe 0.0140)
- **Log:** `logs/single_fixed.log`
- **CSV:** `paper_trades/single_fixed_20260117.csv`
- **Started:** 2026-01-17 05:14 UTC
- **Status:** ✅ Running

---

## 🛡️ Concentration Fix Status

Both traders now have the concentration limit fix applied:
- ✅ Limit enforced on scaled actions (actual trade quantities)
- ✅ Max 30% per asset (enforced in environment)
- ✅ Logging enabled for capped trades

---

## 📊 Monitoring Commands

### Check Trader Status
```bash
ps aux | grep paper_trader_alpaca_polling.py | grep -v grep
```

### View Logs
```bash
# Ensemble trader
tail -f logs/ensemble_fixed.log

# Single model trader
tail -f logs/single_fixed.log
```

### Watch for Concentration Limit Messages
```bash
# Live monitoring (shows when fix caps trades)
./monitor_concentration_fix.sh

# Or manually:
tail -f logs/ensemble_fixed.log logs/single_fixed.log | grep "Concentration limit"
```

### Check Dashboard
```bash
# Real-time dashboard (auto-refreshes)
./start_dashboard.sh

# Or quick snapshot
python dashboard_snapshot.py
```

---

## 🔍 What to Look For

### First Hour (05:14 - 06:14 UTC)

**Good signs:**
- Both traders polling every 60 seconds ✓
- New bars processed when hour completes ✓
- Actions executed ✓
- Max concentration ≤30% in dashboard ✓

**Fix working:**
```
🛡️  Concentration limit: LINK/USD buy capped from 35.50 to 21.43 shares
    Current: 0.0%, Would be: 49.7%, Limit: 30%
```

**If you see this:** Fix is working! Model wanted 49.7%, got capped to 30%.

**If you DON'T see this:** Either:
1. Model isn't trying to exceed 30% (naturally diversified)
2. No buy signals yet (waiting for market conditions)
3. Still waiting for new hourly bar

---

## 📈 Expected Behavior

### Immediately
- Both traders polling for new bars
- Waiting for next hourly candle (06:00 UTC)
- No trades until new bar arrives

### At Next Hour (06:00 UTC)
- New bars detected
- Models generate actions
- Risk management applies
- **Concentration fix may trigger if model wants >30%**
- Trades execute
- CSV files updated

### Over Next 24 Hours
- Compare ensemble vs single model performance
- Watch concentration distribution
- Verify no asset exceeds 30%
- Look for fix messages in logs

---

## 🎯 Success Criteria

After 24 hours, verify:

1. **No concentration violations**
   ```bash
   python -c "
   import pandas as pd
   df = pd.read_csv('paper_trades/ensemble_fixed_20260117.csv')
   # Calculate max concentration
   # Should be ≤30%
   "
   ```

2. **Fix messages in logs** (if model tried to exceed limit)
   ```bash
   grep -c "Concentration limit" logs/*.log
   ```

3. **Better diversification** than before
   - Before: Single asset 48.6%
   - After: All assets ≤30%

4. **Dashboard shows green**
   - Max Concentration in green/yellow zone
   - No red warnings

---

## 🔄 If You Need to Restart

### Stop Traders
```bash
# Get PIDs
ps aux | grep paper_trader_alpaca_polling

# Stop gracefully
kill -SIGTERM 1105329  # Ensemble
kill -SIGTERM 1105587  # Single model

# Wait and verify stopped
sleep 3
ps aux | grep paper_trader
```

### Start Traders
```bash
# Ensemble
nohup python -u paper_trader_alpaca_polling.py \
    --model-dir train_results/ensemble \
    --tickers BTC/USD ETH/USD LTC/USD BCH/USD LINK/USD UNI/USD AAVE/USD \
    --timeframe 1h --poll-interval 60 --gpu -1 \
    --log-file paper_trades/ensemble_fixed_20260117.csv \
    --max-position-pct 0.30 --stop-loss-pct 0.10 \
    > logs/ensemble_fixed.log 2>&1 &

# Single model
nohup python -u paper_trader_alpaca_polling.py \
    --model-dir train_results/cwd_tests/trial_861_1h \
    --tickers BTC/USD ETH/USD LTC/USD BCH/USD LINK/USD UNI/USD AAVE/USD \
    --timeframe 1h --poll-interval 60 --gpu -1 \
    --log-file paper_trades/single_fixed_20260117.csv \
    --max-position-pct 0.30 --stop-loss-pct 0.10 \
    > logs/single_fixed.log 2>&1 &
```

---

## 📁 Related Files

- `BUG_REPORT_CONCENTRATION_LIMIT.md` - Original bug report
- `CONCENTRATION_LIMIT_FIX_APPLIED.md` - Fix documentation
- `test_concentration_limit_fix.py` - Test suite (passed ✅)
- `monitor_concentration_fix.sh` - Live monitoring script
- `TRADERS_RESTARTED.md` - This file

---

## 🎉 Summary

✅ **Both traders running with concentration fix**
✅ **Old buggy traders stopped**
✅ **New CSV files for clean comparison**
✅ **Monitoring scripts ready**

**Next:** Wait for next hour (06:00 UTC) to see first trades with fix applied.

---

**Restart Time:** January 17, 2026, 05:13-05:14 UTC
**Next Hourly Bar:** 06:00 UTC (~45 minutes)
**First Fix Verification:** After first trades execute
