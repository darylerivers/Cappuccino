# Live Performance Degradation Monitor

## ✅ INSTALLED & RUNNING

**Status:** Active monitoring of Trial #250
**PID:** Check with `ps aux | grep live_performance_monitor`
**Logs:** `logs/live_performance_monitor.log`

---

## What It Does

Continuously monitors your paper trading performance and:
1. **Calculates live Sharpe ratio** from paper trading CSV
2. **Compares to backtest expectations** (Sharpe 0.1803)
3. **Sends Discord alerts** at degradation thresholds
4. **Auto-stops trading** if severe degradation persists

---

## Alert Thresholds

### ✅ OK (Green)
- Live Sharpe within 0.5 of backtest
- No alerts sent
- Trading continues normally

### ⚠️ WARNING (Orange)
- Live Sharpe < (Backtest - 0.5)
- Discord notification sent once
- Continue monitoring

### 🔴 CRITICAL (Red)
- Live Sharpe < -1.0
- Discord notification sent once
- **Current status as of startup!**

### 🚨 EMERGENCY (Dark Red)
- Live Sharpe < -2.0 for 24 hours
- **AUTOMATICALLY STOPS PAPER TRADER**
- Discord notification with action taken

---

## Current Status (as of startup)

```
Live Sharpe:     -5.3943
Backtest Sharpe:  0.1803
Gap:             -5.5746
Status:          🔴 CRITICAL
Bars:            28
Alert Sent:      Yes (CRITICAL level)
```

**What this means:**
- Your model is significantly underperforming backtest
- You've been notified via Discord
- Trading continues (not at emergency threshold yet)
- Monitor will check again in 1 hour

---

## Discord Notifications

You'll receive Discord messages for:

### Performance Alerts
- ⚠️ WARNING when degradation detected
- 🔴 CRITICAL when severe degradation
- 🚨 EMERGENCY when auto-stopping

### 6-Hour Summaries (if performing OK)
- Regular status updates
- Sharpe, return, bars processed
- Only sent when no alerts active

### Startup Notification
- Confirms monitor is active
- Shows configuration

---

## Monitor Configuration

| Setting | Value | Description |
|---------|-------|-------------|
| **CSV File** | `paper_trades/trial250_session.csv` | Paper trading log |
| **Backtest Sharpe** | 0.1803 | Expected performance |
| **Check Interval** | 3600s (1 hour) | How often to check |
| **Warning Threshold** | -0.5 from backtest | First alert level |
| **Critical Threshold** | -1.0 | Second alert level |
| **Emergency Threshold** | -2.0 | Auto-stop threshold |
| **Emergency Duration** | 24 hours | How long before auto-stop |
| **Min Bars** | 10 | Minimum data before calculating |

---

## How It Works

### Every Hour:
1. **Read CSV** - Load paper trading data
2. **Calculate Sharpe** - Compute rolling Sharpe ratio
3. **Compare** - Check against backtest expectation
4. **Alert** - Send Discord if thresholds crossed
5. **Log** - Record status to log file
6. **Sleep** - Wait 1 hour, repeat

### State Tracking:
- Remembers what alerts were sent (avoids spam)
- Tracks emergency timer (24-hour countdown)
- Logs alert history
- State saved to `monitoring/live_monitor_state.json`

### Auto-Stop Logic:
```
IF live_sharpe < -2.0:
    START emergency timer
    IF timer >= 24 hours:
        STOP paper trader
        SEND emergency alert
        EXIT monitor
```

---

## Managing the Monitor

### Check Status
```bash
# View live logs
tail -f logs/live_performance_monitor.log

# Check if running
ps aux | grep live_performance_monitor

# View state file
cat monitoring/live_monitor_state.json
```

### Stop Monitor
```bash
# Find PID
pgrep -f live_performance_monitor

# Kill it
pkill -f live_performance_monitor

# Or specific PID
kill <PID>
```

### Restart Monitor
```bash
# Stop old one
pkill -f live_performance_monitor

# Start new one
nohup python monitoring/live_performance_monitor.py \
    --paper-trader-csv paper_trades/trial250_session.csv \
    --backtest-sharpe 0.1803 \
    --check-interval 3600 \
    > logs/live_performance_monitor.log 2>&1 &
```

### Adjust Thresholds
```bash
# More sensitive (earlier warnings)
python monitoring/live_performance_monitor.py \
    --warning-threshold 0.3 \    # Alert at smaller gaps
    --critical-threshold -0.5 \  # Lower threshold
    --emergency-threshold -1.5   # Lower auto-stop

# Less sensitive (fewer alerts)
python monitoring/live_performance_monitor.py \
    --warning-threshold 1.0 \    # Larger gap before alert
    --critical-threshold -2.0 \  # Higher threshold
    --emergency-threshold -5.0   # Much lower auto-stop
```

---

## Alert Examples

### WARNING Alert
```
⚠️ WARNING: Live Sharpe -0.35 vs Backtest 0.18 (gap: -0.53)

📊 Live Sharpe: -0.3500
📈 Backtest Sharpe: 0.1803
📉 Gap: -0.5303
⏱️ Runtime: 50 bars
💰 Return: -0.15%
```

### CRITICAL Alert
```
🔴 CRITICAL: Live Sharpe -1.24 < -1.0! Model severely degraded.

📊 Live Sharpe: -1.2400
📈 Backtest Sharpe: 0.1803
📉 Gap: -1.4203
⏱️ Runtime: 100 bars
💰 Return: -0.50%
```

### EMERGENCY Alert
```
🚨 EMERGENCY: Live Sharpe -2.15 < -2.0 for 24.5h! Auto-stopping trading.

📊 Live Sharpe: -2.1500
📈 Backtest Sharpe: 0.1803
📉 Gap: -2.3303
⏱️ Runtime: 150 bars
💰 Return: -1.20%
🚨 Action: STOPPING PAPER TRADER
```

---

## Integration with Other Systems

### Works With:
- ✅ Paper trader (`paper_trader_alpaca_polling.py`)
- ✅ Discord notifications (uses same webhook)
- ✅ Multiple models (just point to different CSV)

### Doesn't Interfere With:
- Training campaigns (1hr, 5min)
- Other monitors
- Paper trader operation

### Can Run Multiple:
```bash
# Monitor Trial #250
python monitoring/live_performance_monitor.py \
    --paper-trader-csv paper_trades/trial250_session.csv \
    --backtest-sharpe 0.1803 &

# Monitor Trial #300 (when deployed)
python monitoring/live_performance_monitor.py \
    --paper-trader-csv paper_trades/trial300_session.csv \
    --backtest-sharpe 0.2500 &

# Monitor 5min model (when deployed)
python monitoring/live_performance_monitor.py \
    --paper-trader-csv paper_trades/trial_5min_session.csv \
    --backtest-sharpe 1.2000 &
```

---

## Troubleshooting

### No Discord Alerts?
- Check `DISCORD.ENABLED` in `constants.py`
- Verify webhook URL in `.env`
- Check logs for errors
- Test with: `python -c "from integrations.discord_notifier import DiscordNotifier; DiscordNotifier().send_message('Test')"`

### Monitor Not Starting?
- Check logs: `tail -f logs/live_performance_monitor.log`
- Verify CSV exists: `ls -lh paper_trades/trial250_session.csv`
- Check permissions: `ls -la monitoring/live_performance_monitor.py`

### Wrong Sharpe Calculation?
- Verify CSV has data: `wc -l paper_trades/trial250_session.csv`
- Check format: `head paper_trades/trial250_session.csv`
- Minimum 10 bars required
- Hourly data assumed for annualization

### False Alerts?
- Increase min-bars: `--min-bars 50`
- Widen thresholds: `--warning-threshold 1.0`
- Increase check interval: `--check-interval 7200` (2 hours)

---

## Advanced Features

### Custom Check Interval
```bash
# Check every 30 minutes
--check-interval 1800

# Check every 6 hours
--check-interval 21600

# Check every 15 minutes (frequent)
--check-interval 900
```

### Different Models
```bash
# Ensemble model (higher expected Sharpe)
--backtest-sharpe 0.8 \
--critical-threshold -0.5  # Stricter for better model

# Experimental model (lower expectations)
--backtest-sharpe 0.05 \
--critical-threshold -2.0  # More lenient
```

### Emergency Duration
```bash
# Auto-stop after 6 hours of degradation
--emergency-duration 6

# Auto-stop after 48 hours (very patient)
--emergency-duration 48

# Never auto-stop (manual intervention only)
--emergency-duration 999999
```

---

## Log Analysis

### View Recent Activity
```bash
tail -100 logs/live_performance_monitor.log
```

### Check Alert History
```bash
cat monitoring/live_monitor_state.json | jq '.alert_history'
```

### Count Alerts by Level
```bash
cat monitoring/live_monitor_state.json | \
  jq '.alert_history[] | .level' | \
  sort | uniq -c
```

### Find When Emergency Started
```bash
cat monitoring/live_monitor_state.json | \
  jq '.emergency_start_time'
```

---

## What Happens Next

### Scenario 1: Performance Improves
- Sharpe recovers above critical threshold
- Alerts stop being sent
- Emergency timer resets
- 6-hour summaries resume

### Scenario 2: Stays at Critical
- Alert sent once (no spam)
- Monitoring continues hourly
- Emergency timer only starts if drops below -2.0
- Trading continues (manual decision to stop)

### Scenario 3: Drops to Emergency
- Emergency timer starts
- Continues for 24 hours
- At 24 hours: auto-stops paper trader
- Sends final emergency alert
- Monitor exits

### Scenario 4: New Model Deployed
- Stop old monitor: `pkill -f live_performance_monitor`
- Update CSV path and backtest Sharpe
- Start new monitor with new parameters

---

## Best Practices

1. **Keep It Running** - Don't stop unless redeploying
2. **Check Discord** - Respond to alerts promptly
3. **Review Logs Weekly** - Look for patterns
4. **Adjust Thresholds** - Based on model characteristics
5. **Multiple Monitors** - One per paper trader
6. **Archive State** - Before redeploying models

---

## Summary

**What You Get:**
- ✅ Continuous performance monitoring
- ✅ Automatic degradation detection
- ✅ Discord alerts at thresholds
- ✅ Auto-stop protection on severe degradation
- ✅ Works with any paper trading CSV
- ✅ No interference with trading or training

**Current Status:**
- 🔴 CRITICAL alert already sent (Sharpe -5.39)
- ⏰ Next check in ~1 hour
- 🚨 Will auto-stop if drops to -2.0 for 24h
- 📊 Monitoring Trial #250 continuously

**You're now protected!** 🛡️
