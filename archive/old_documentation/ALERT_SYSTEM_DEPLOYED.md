# Alert System - DEPLOYED ✅

**Date:** January 17, 2026, 04:03 UTC
**Status:** Running in background

---

## 🎉 Alert System Active

The paper trading alert system is now monitoring both traders 24/7.

### Running Processes

```
PID: 1105329 - Ensemble Trader (train_results/ensemble)
PID: 1105587 - Single Model Trader (trial_861_1h)
PID: 1162586 - Alert System (checking every 60 seconds)
```

---

## 📊 What It's Monitoring

### Both Traders (Ensemble + Single Model)

✅ **Process Health**
- Checks if traders are still running
- Detects crashes or zombie processes
- Alert: 🚨 CRITICAL if process dies

✅ **Concentration Limits**
- Monitors position sizes vs 30% limit
- Alert: 🚨 CRITICAL if >30%
- Alert: ⚠️ WARNING if 28-30%

✅ **Portfolio Losses**
- Tracks total portfolio value
- Alert: 🚨 CRITICAL if down >10%
- Alert: ⚠️ WARNING if down 5-10%

✅ **Trading Activity**
- Counts trades in last 24 hours
- Alert: ⚠️ WARNING if zero trades

✅ **Performance Trends**
- Compares recent vs past 24h
- Alert: ⚠️ WARNING if degrading >5%

✅ **Data Freshness**
- Checks last update timestamp
- Alert: 🚨 CRITICAL if >6h old
- Alert: ⚠️ WARNING if 3-6h old

---

## 📁 Files Being Monitored

**Ensemble Trader:**
- CSV: `paper_trades/ensemble_fixed_20260117.csv`
- PID: 1105329
- Log: `logs/ensemble_fixed.log`

**Single Model Trader:**
- CSV: `paper_trades/single_fixed_20260117.csv`
- PID: 1105587
- Log: `logs/single_fixed.log`

**Alert System:**
- Log: `logs/alert_system.log`
- Alert Log: `logs/alerts.log`

---

## 🔍 How to Monitor

### Check Alert System Status
```bash
ps aux | grep alert_system
```

Expected output:
```
mrc   1162586  ...  python alert_system.py --check-interval 60
```

### View Recent Alerts
```bash
tail -f logs/alerts.log
```

If no alerts yet, file may not exist (created on first alert).

### Check System Log
```bash
tail -f logs/alert_system.log
```

Should show periodic check messages:
```
[10:15:23] Running checks...
[10:15:23] Checks complete.
```

### Monitor All Logs
```bash
tail -f logs/alerts.log logs/ensemble_fixed.log logs/single_fixed.log
```

---

## 🎯 What to Expect

### First Hour
- Alert system running every 60 seconds
- No alerts if everything is healthy ✅
- Checking:
  - Both traders still running ✓
  - CSV files updating ✓
  - Concentration limits ✓
  - No major losses ✓

### First Alert
When something goes wrong, you'll see:
```
🚨 [CRITICAL] Alert Title
Trader: ensemble
Time: 2026-01-17 10:15:23
Detailed message about the issue...
```

**In terminal** (if running in foreground)
**In logs/alerts.log** (always)

### Normal Operation
Most of the time, you'll see:
- No terminal output (silence is good!)
- Periodic check messages in log
- No entries in alerts.log
- This means: **Everything is working perfectly** ✅

---

## 🚨 When Alerts Fire

### CRITICAL Alerts - Act Immediately

**Process Crashed:**
1. Check trader logs for errors
2. Restart trader
3. Investigate root cause

**Concentration >30%:**
1. Run dashboard: `python dashboard_snapshot.py`
2. Verify violation
3. If true: Concentration fix may not be working!

**Portfolio Loss >10%:**
1. Review recent trades in CSV
2. Check if stop-loss triggered
3. Consider pausing trader

**Data Stale >6h:**
1. Check if process running
2. Check logs for errors
3. Restart if needed

### WARNING Alerts - Monitor

These are informational, usually don't require immediate action:
- Concentration 28-30%: Watch closely
- No trades 24h: May be normal in flat markets
- Portfolio loss 5-10%: Monitor for improvement
- Performance degrading: Compare to benchmark

---

## 🛠️ Management Commands

### Check Status
```bash
# All processes
ps aux | grep "alert_system\|paper_trader" | grep -v grep

# Just alert system
ps aux | grep alert_system | grep -v grep
```

### View Logs
```bash
# Alert log (when issues detected)
tail -f logs/alerts.log

# System log (periodic checks)
tail -f logs/alert_system.log

# Both trader logs
tail -f logs/ensemble_fixed.log logs/single_fixed.log
```

### Stop Alert System
```bash
# Find PID
ps aux | grep alert_system | grep -v grep

# Stop gracefully
kill -SIGTERM 1162586

# Or force stop
kill -9 1162586
```

### Restart Alert System
```bash
# Stop old one
kill $(pgrep -f alert_system.py)

# Start new one
nohup python alert_system.py --check-interval 60 > logs/alert_system.log 2>&1 &

# Or use launcher
./start_alerts.sh
```

---

## 📈 Configuration

### Current Settings
- **Check interval:** 60 seconds (1 minute)
- **Ensemble CSV:** Auto-detected
- **Single CSV:** Auto-detected
- **Ensemble PID:** Auto-detected (1105329)
- **Single PID:** Auto-detected (1105587)
- **Cooldowns:**
  - CRITICAL: 15 minutes
  - WARNING: 30 minutes
  - INFO: 60 minutes

### Adjust Check Interval
```bash
# Stop current
kill $(pgrep -f alert_system.py)

# Start with different interval
nohup python alert_system.py --check-interval 300 > logs/alert_system.log 2>&1 &
# 300 = 5 minutes (recommended for production)
```

---

## 📊 Integration with Dashboard

### Run Both Simultaneously

**Option 1: Two Terminals**
```bash
# Terminal 1: Dashboard
./start_dashboard.sh

# Terminal 2: Alerts (already running in background)
tail -f logs/alerts.log
```

**Option 2: tmux (Recommended)**
```bash
tmux new -s monitoring

# Split horizontal: Ctrl+B, then "
# Top pane: Dashboard
./start_dashboard.sh

# Bottom pane: Alerts
Ctrl+B, then down arrow
tail -f logs/alerts.log

# Detach: Ctrl+B, then D
# Reattach anytime: tmux attach -t monitoring
```

---

## 🧪 Testing

### Test Process Monitoring
```bash
# Temporarily stop a trader
kill -STOP 1105587

# Wait up to 1 minute
# Should alert: "Trader Process Crashed"

# Resume
kill -CONT 1105587
```

### Test Concentration Monitoring
Wait for next trade where model wants >30% in an asset.

If concentration fix is working:
- Trade will be capped at 30%
- NO alert (everything working)

If concentration fix broken:
- Trade will exceed 30%
- 🚨 CRITICAL alert fires

---

## 📁 Files Created

```
✅ alert_system.py              # Main system (500+ lines)
✅ start_alerts.sh               # Quick launcher
✅ ALERT_SYSTEM_README.md        # Quick start guide
✅ ALERT_SYSTEM_GUIDE.md         # Detailed guide
✅ ALERT_SYSTEM_DEPLOYED.md      # This file
✅ logs/alert_system.log         # System log
✅ logs/alerts.log               # Alert log (created on first alert)
```

---

## ✅ Deployment Checklist

- ✅ Alert system built and tested
- ✅ Dependencies installed (psutil)
- ✅ Auto-detection working
- ✅ Process monitoring active
- ✅ Concentration checks active
- ✅ Portfolio loss checks active
- ✅ Trading activity checks active
- ✅ Performance checks active
- ✅ Staleness checks active
- ✅ Running in background (PID 1162586)
- ✅ Logs configured
- ✅ Documentation complete

---

## 🎯 Next Steps

1. **Let it run** - Alert system is monitoring 24/7
2. **Check daily** - Review `logs/alerts.log` for any issues
3. **Respond to alerts** - Act on CRITICAL immediately
4. **Adjust if needed** - Tune thresholds or interval

---

## 🎉 Summary

✅ **Alert system deployed and running**
✅ **Monitoring both traders (ensemble + single)**
✅ **6 types of checks every 60 seconds**
✅ **Smart cooldowns prevent spam**
✅ **Logs to file and terminal**
✅ **Auto-detects CSV files and PIDs**

**Status:** 🟢 Active
**PID:** 1162586
**Check Interval:** 60 seconds
**Last Started:** January 17, 2026, 04:03 UTC

---

The alert system is your safety net. It watches your traders 24/7 and tells you immediately if something goes wrong. Combined with the dashboard for visual monitoring, you now have complete visibility into your paper trading operation!
