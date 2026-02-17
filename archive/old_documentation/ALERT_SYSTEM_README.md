# Alert System - Quick Start ✅

**Status:** Ready to use
**Built:** January 17, 2026

---

## What It Does

Monitors your paper traders 24/7 and alerts you when something goes wrong:

- 🚨 **Process crashed**
- 🚨 **Concentration violated** (>30%)
- 🚨 **Portfolio loss** (>10%)
- ⚠️ **No trading activity** (24h)
- ⚠️ **Performance degrading**
- ⚠️ **Data going stale**

---

## Quick Start

```bash
cd /opt/user-data/experiment/cappuccino

# Start alert system (auto-detects everything)
./start_alerts.sh

# Or run in background
nohup ./start_alerts.sh > logs/alert_system.log 2>&1 &
```

---

## What You'll See

### When Everything Is OK
```
[10:15:23] Running checks...
[10:15:23] Checks complete.
[10:20:23] Running checks...
[10:20:23] Checks complete.
```

Silence is good! No news is good news.

### When Issues Are Detected
```
🚨 [CRITICAL] Concentration Limit Violated
Trader: single
Time: 2026-01-17 10:15:23
LINK/USD: 48.6% of portfolio (limit: 30%)
Position value: $415.44
Total portfolio: $854.70
🚨 FIX MAY NOT BE WORKING!
```

Color-coded alerts with full details.

---

## Features

✅ **Auto-detection** - Finds CSV files and process IDs automatically
✅ **Smart cooldowns** - Prevents alert spam (won't repeat same alert for 15-60 min)
✅ **Multiple channels** - Terminal + log file (email coming soon)
✅ **6 check types** - Process, concentration, losses, activity, performance, staleness
✅ **Continuous** - Runs 24/7 checking every 5 minutes

---

## Configuration

### Check Interval
```bash
# Every 5 minutes (default)
./start_alerts.sh

# Every 10 minutes (custom)
python alert_system.py --check-interval 600

# Every minute (testing)
python alert_system.py --check-interval 60
```

### Manual CSV/PID
```bash
python alert_system.py \
    --ensemble-csv paper_trades/ensemble_fixed_20260117.csv \
    --single-csv paper_trades/single_fixed_20260117.csv \
    --ensemble-pid 1105329 \
    --single-pid 1105587
```

---

## Monitoring

### Check Alert System Status
```bash
ps aux | grep alert_system
```

### View Alert Log
```bash
# Live tail
tail -f logs/alerts.log

# Last 50 alerts
tail -50 logs/alerts.log

# Critical alerts only
grep CRITICAL logs/alerts.log
```

### Stop Alert System
```bash
kill -SIGTERM <PID>
# Or Ctrl+C if in foreground
```

---

## Integration

### With Dashboard
```bash
# Terminal 1
./start_dashboard.sh

# Terminal 2
./start_alerts.sh
```

### In tmux (Recommended)
```bash
tmux new -s monitoring
# Split: Ctrl+B then "
# Top: ./start_dashboard.sh
# Bottom: ./start_alerts.sh
# Detach: Ctrl+B then D
```

---

## Alert Types

| Alert | Level | Cooldown | Action Required |
|-------|-------|----------|----------------|
| Process Crashed | 🚨 CRITICAL | 15 min | Restart trader immediately |
| Concentration >30% | 🚨 CRITICAL | 15 min | Check if fix broken |
| Portfolio Loss >10% | 🚨 CRITICAL | 15 min | Review trades, consider stopping |
| Data Stale >6h | 🚨 CRITICAL | 15 min | Check logs, restart if needed |
| Concentration 28-30% | ⚠️ WARNING | 30 min | Monitor, ensure doesn't cross |
| No Trades 24h | ⚠️ WARNING | 30 min | Check market conditions |
| Portfolio Loss 5-10% | ⚠️ WARNING | 30 min | Monitor closely |
| Performance Degrading | ⚠️ WARNING | 30 min | Compare to benchmark |
| Data Delayed 3-6h | ⚠️ WARNING | 30 min | Check if trader is stuck |

---

## Dependencies

```bash
# Required
pip install psutil

# Already installed if you ran start_alerts.sh
```

---

## Files

```
✅ alert_system.py              # Main system (500+ lines)
✅ start_alerts.sh               # Quick launcher
✅ ALERT_SYSTEM_README.md        # This file
✅ ALERT_SYSTEM_GUIDE.md         # Detailed guide
✅ logs/alerts.log               # Alert log (created on first alert)
```

---

## Example Usage

### Basic (Auto-detect everything)
```bash
./start_alerts.sh
```

### Background Daemon
```bash
nohup ./start_alerts.sh > logs/alert_system.log 2>&1 &
echo $! > logs/alert_system.pid
```

### Stop Background Daemon
```bash
kill $(cat logs/alert_system.pid)
```

---

## Troubleshooting

**No CSV files found?**
```bash
ls paper_trades/*.csv
# If empty, traders aren't running or haven't created files yet
```

**No processes found?**
```bash
ps aux | grep paper_trader
# Manually specify PIDs with --ensemble-pid and --single-pid
```

**psutil not installed?**
```bash
pip install psutil
```

---

## What's Next

After starting the alert system:

1. **Let it run** - Leave it monitoring in background
2. **Check periodically** - `tail logs/alerts.log`
3. **Respond to alerts** - Act on CRITICAL immediately
4. **Review daily** - Check alert log for patterns

---

**Quick Launch:**
```bash
./start_alerts.sh
```

**Background Launch:**
```bash
nohup ./start_alerts.sh > logs/alert_system.log 2>&1 &
```

**Status Check:**
```bash
ps aux | grep alert_system
tail logs/alerts.log
```

---

**Status:** ✅ Ready
**Recommended:** Run 24/7 in background
**Check interval:** 5 minutes (configurable)
