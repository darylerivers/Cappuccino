# Paper Trading Alert System

**Status:** ✅ Ready to use
**Created:** January 17, 2026

---

## What It Does

Continuously monitors your paper traders and sends alerts when critical issues are detected:

### 🚨 Critical Alerts
- **Process Crashed** - Trader stopped running
- **Concentration Violated** - Position >30% in single asset
- **Portfolio Loss** - Down >10% from starting capital
- **Data Stale** - No updates for >6 hours

### ⚠️ Warning Alerts
- **Concentration Near Limit** - Position 28-30% (approaching)
- **No Trading Activity** - No trades in 24 hours
- **Portfolio Loss** - Down 5-10%
- **Performance Degradation** - Significantly worse than yesterday
- **Data Delayed** - No updates for 3-6 hours

---

## Quick Start

### Option 1: Auto-Detection (Easiest)
```bash
cd /opt/user-data/experiment/cappuccino
./start_alerts.sh
```

The alert system will automatically find:
- Latest CSV files for both traders
- Running process IDs
- Start monitoring every 5 minutes

### Option 2: Background Daemon
```bash
# Run in background
nohup ./start_alerts.sh > logs/alert_system.log 2>&1 &

# Check it's running
ps aux | grep alert_system
```

### Option 3: Custom Settings
```bash
python alert_system.py \
    --ensemble-csv paper_trades/ensemble_fixed_20260117.csv \
    --single-csv paper_trades/single_fixed_20260117.csv \
    --ensemble-pid 1105329 \
    --single-pid 1105587 \
    --check-interval 300  # 5 minutes
```

---

## Features

### 1. Process Monitoring
Checks if trader processes are still running:
- Uses process IDs to track traders
- Alerts if process dies or becomes zombie
- **Alert:** 🚨 CRITICAL - "Trader Process Crashed"

### 2. Concentration Limit Enforcement
Monitors position sizes:
- Checks every asset vs 30% limit
- **>30%:** 🚨 CRITICAL - "Concentration Limit Violated"
- **28-30%:** ⚠️ WARNING - "Concentration Near Limit"
- Helps verify the concentration fix is working

### 3. Trading Activity
Monitors for stuck traders:
- Counts trades in last 24 hours
- **0 trades:** ⚠️ WARNING - "No Trading Activity"
- Normal if markets are flat, concerning if extended

### 4. Stop-Loss Monitoring
Tracks portfolio losses:
- Compares current vs initial portfolio value
- **>10% loss:** 🚨 CRITICAL - "Portfolio Loss Exceeds Threshold"
- **5-10% loss:** ⚠️ WARNING - "Portfolio Loss Warning"

### 5. Performance Tracking
Compares recent vs past performance:
- Last 24h return vs previous 24h
- **>5% degradation:** ⚠️ WARNING - "Performance Degradation"

### 6. Staleness Detection
Ensures data is being updated:
- Checks timestamp of latest data point
- **>6 hours old:** 🚨 CRITICAL - "Data Stale - No Updates"
- **3-6 hours old:** ⚠️ WARNING - "Data Updates Delayed"

### 7. Smart Cooldowns
Prevents alert spam:
- **INFO:** 60 minute cooldown
- **WARNING:** 30 minute cooldown
- **CRITICAL:** 15 minute cooldown
- Same alert won't spam you repeatedly

---

## Alert Channels

### Terminal Output (Enabled by default)
Color-coded alerts printed to console:
- 🚨 **Red:** Critical issues
- ⚠️ **Yellow:** Warnings
- ℹ️ **Blue:** Informational

### Log File (Enabled by default)
All alerts written to `logs/alerts.log`:
```
[2026-01-17 10:15:23] [CRITICAL] [ensemble] Concentration Limit Violated: ...
[2026-01-17 10:20:45] [WARNING] [single] No Trading Activity: ...
```

### Email Notifications (Coming soon)
Placeholder for future email integration:
- SMTP support
- SendGrid/AWS SES
- Configurable recipients

---

## Configuration

### Check Interval
How often to run checks:
```bash
# Every 5 minutes (default)
python alert_system.py --check-interval 300

# Every 10 minutes (less frequent)
python alert_system.py --check-interval 600

# Every minute (very frequent, for testing)
python alert_system.py --check-interval 60
```

**Recommendation:** 5 minutes (300s) is good balance between responsiveness and resource usage.

### Auto-Detection
If you don't specify CSV files or PIDs:
- Finds latest CSV files in `paper_trades/`
- Prefers `*_fixed_*.csv` files
- Scans running processes for paper traders
- Reports what it found before starting

---

## Example Alerts

### Critical: Concentration Violation
```
🚨 [CRITICAL] Concentration Limit Violated
Trader: single
Time: 2026-01-17 10:15:23
LINK/USD: 48.6% of portfolio (limit: 30%)
Position value: $415.44
Total portfolio: $854.70
🚨 FIX MAY NOT BE WORKING!
```

### Warning: Near Concentration Limit
```
⚠️  [WARNING] Concentration Near Limit
Trader: ensemble
Time: 2026-01-17 10:20:45
BTC/USD: 29.2% of portfolio (limit: 30%)
Position value: $250.15
```

### Critical: Process Crashed
```
🚨 [CRITICAL] Trader Process Crashed
Trader: single
Time: 2026-01-17 10:25:12
Process 1105587 is not running!
```

### Warning: No Trading Activity
```
⚠️  [WARNING] No Trading Activity
Trader: ensemble
Time: 2026-01-17 10:30:00
No trades executed in the last 24 hours.
Data points: 24
This may be normal if market conditions don't trigger signals.
```

### Critical: Portfolio Loss
```
🚨 [CRITICAL] Portfolio Loss Exceeds Threshold
Trader: single
Time: 2026-01-17 10:35:18
Portfolio down 12.5% from start
Initial: $1000.00
Current: $875.00
Loss: $125.00
```

### Warning: Data Stale
```
⚠️  [WARNING] Data Updates Delayed
Trader: ensemble
Time: 2026-01-17 10:40:33
No data updates for 4.2 hours
Last update: 2026-01-17 06:20:00+00:00
```

---

## Monitoring the Alert System

### Check It's Running
```bash
ps aux | grep alert_system
```

### View Alert Log
```bash
# Live tail
tail -f logs/alerts.log

# Last 50 alerts
tail -50 logs/alerts.log

# Search for critical alerts
grep CRITICAL logs/alerts.log
```

### Stop Alert System
```bash
# Find PID
ps aux | grep alert_system

# Stop gracefully
kill -SIGTERM <PID>

# Or use Ctrl+C if running in foreground
```

---

## Integration with Other Tools

### With Dashboard
Run both simultaneously:
```bash
# Terminal 1: Dashboard (visual monitoring)
./start_dashboard.sh

# Terminal 2: Alert system (background monitoring)
./start_alerts.sh
```

### With tmux
Keep both running persistently:
```bash
# Create tmux session
tmux new -s monitoring

# Split window
Ctrl+B, then "

# Top pane: Dashboard
./start_dashboard.sh

# Bottom pane: Alert system
Ctrl+B, then down arrow
./start_alerts.sh

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t monitoring
```

---

## Troubleshooting

### "Auto-detecting CSV files... None found"
**Problem:** No CSV files in `paper_trades/`

**Solution:**
```bash
# Check if traders are running
ps aux | grep paper_trader

# Check if CSV files exist
ls -lh paper_trades/*.csv

# Specify manually
python alert_system.py \
    --ensemble-csv paper_trades/your_file.csv \
    --single-csv paper_trades/your_file.csv
```

### "Auto-detecting trader processes... None found"
**Problem:** Can't find running traders

**Solution:**
```bash
# Manually specify PIDs
ps aux | grep paper_trader
python alert_system.py --ensemble-pid 12345 --single-pid 67890
```

### "ModuleNotFoundError: No module named 'psutil'"
**Problem:** psutil not installed

**Solution:**
```bash
pip install psutil
```

### Alerts Not Appearing
**Possible causes:**
1. **Cooldown active** - Wait 15-60 minutes for same alert
2. **No issues detected** - Everything is working! ✅
3. **Check interval too long** - Use shorter interval

```bash
# Test with 1-minute interval
python alert_system.py --check-interval 60
```

---

## Advanced Usage

### Custom Alert Thresholds
Edit `alert_system.py` to customize:

**Concentration threshold** (line ~254):
```python
if concentration > 30:  # Change to 25 for stricter
    self._send_alert(...)
```

**Loss threshold** (line ~319):
```python
if loss_pct < -10:  # Change to -5 for earlier warning
    self._send_alert(...)
```

**Staleness threshold** (line ~363):
```python
if hours_since_update > 6:  # Change to 3 for faster alert
    self._send_alert(...)
```

### Add Email Notifications
Implement `_send_email()` method (line ~66):

```python
def _send_email(self, alert: Alert):
    import smtplib
    from email.message import EmailMessage

    msg = EmailMessage()
    msg['Subject'] = f"[{alert.level.value}] {alert.title}"
    msg['From'] = "alerts@yourdomain.com"
    msg['To'] = "your@email.com"
    msg.set_content(f"{alert.trader}: {alert.message}")

    smtp = smtplib.SMTP('smtp.gmail.com', 587)
    smtp.starttls()
    smtp.login('your@email.com', 'password')
    smtp.send_message(msg)
    smtp.quit()
```

Then enable:
```python
self.notifier = AlertNotifier(enable_email=True)
```

---

## Files Created

```
✅ alert_system.py          # Main alert system (500+ lines)
✅ start_alerts.sh           # Quick launcher
✅ ALERT_SYSTEM_GUIDE.md     # This file
✅ logs/alerts.log           # Alert log (created on first alert)
```

---

## What to Do When Alerts Fire

### 🚨 CRITICAL Alerts - Act Immediately

**"Trader Process Crashed"**
1. Check logs for error: `tail -100 logs/ensemble_fixed.log`
2. Restart trader: `./start_traders.sh`
3. Investigate root cause

**"Concentration Limit Violated"**
1. Check dashboard: `python dashboard_snapshot.py`
2. Verify concentration >30%
3. **If true:** Concentration fix not working! Report bug
4. **If false:** False positive, check alert logic

**"Portfolio Loss Exceeds Threshold"**
1. Review recent trades in CSV
2. Check if stop-loss triggered correctly
3. Consider pausing trader if losses continuing
4. Analyze what went wrong

**"Data Stale - No Updates"**
1. Check if process still running: `ps aux | grep paper_trader`
2. Check logs for errors
3. Restart if needed

### ⚠️ WARNING Alerts - Monitor Closely

**"Concentration Near Limit"**
- Watch next few trades
- Ensure doesn't cross 30%
- Normal if just under limit

**"No Trading Activity"**
- Check if markets are flat (no opportunities)
- Review model signals
- OK if extended sideways market

**"Performance Degradation"**
- Compare to benchmark
- Check if market conditions changed
- Monitor for improvement

---

## Testing the Alert System

### Test Process Monitoring
```bash
# Stop one trader temporarily
kill -STOP 1105587

# Wait for next check (up to 5 min)
# Should alert: "Trader Process Crashed" (actually just stopped)

# Resume trader
kill -CONT 1105587
```

### Test With Mock Data
Create fake CSV with concentration violation:
```python
# test_alert.py
import pandas as pd

df = pd.DataFrame({
    'timestamp': ['2026-01-17T10:00:00+00:00'],
    'cash': [500],
    'total_asset': [1000],
    'reward': [0],
    'holding_BTC/USD': [0],
    'holding_ETH/USD': [0],
    'holding_LINK/USD': [30],  # 30 shares
    'price_BTC/USD': [95000],
    'price_ETH/USD': [3300],
    'price_LINK/USD': [15],  # 30 * 15 = $450 = 45% of $1000
})

df.to_csv('paper_trades/test_alert.csv', index=False)
```

Then run:
```bash
python alert_system.py --ensemble-csv paper_trades/test_alert.csv --check-interval 10
# Should immediately alert about 45% concentration in LINK
```

---

## Summary

✅ **Monitors:** Processes, concentration, losses, performance, staleness
✅ **Alerts:** Terminal (color-coded) + Log file
✅ **Smart:** Auto-detection, cooldowns, severity levels
✅ **Easy:** One command to start
✅ **Reliable:** Runs continuously in background

**Launch command:**
```bash
./start_alerts.sh
```

**Run in background:**
```bash
nohup ./start_alerts.sh > logs/alert_system.log 2>&1 &
```

---

**Status:** ✅ Ready to use
**Dependencies:** psutil (installed ✓)
**Recommended:** Run 24/7 in background with 5-minute checks
