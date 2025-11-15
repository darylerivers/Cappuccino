# Cappuccino Automation Systems Guide

Complete automation suite for training, deployment, monitoring, and paper trading.

## Overview

This automation system consists of 4 integrated components that work together to:
1. **Auto-deploy** best models to paper trading
2. **Monitor system health** and auto-restart crashed processes
3. **Track performance** and send alerts
4. **Manage paper trading** with the latest best models

---

## Quick Start

### Start All Automation
```bash
./start_automation.sh
```

This launches:
- Auto-Model Deployer (checks every hour for new best models)
- System Watchdog (checks every minute for crashed processes)
- Performance Monitor (checks every 5 minutes, sends desktop notifications)

### Check Status
```bash
./status_automation.sh
```

### Stop All Automation
```bash
./stop_automation.sh
```

---

## Components

### 1. Auto-Model Deployer (`auto_model_deployer.py`)

Automatically finds, validates, and deploys best models to paper trading.

**Features:**
- Monitors Optuna study for new best trials
- Validates models before deployment
- Auto-deploys to paper trading if improvement > threshold
- Maintains deployment history
- Rollback capability

**Configuration:**
```bash
python auto_model_deployer.py \
    --study cappuccino_3workers_20251102_2325 \
    --check-interval 3600 \          # Check every hour
    --min-improvement 1.0 \           # Min 1% improvement to deploy
    --daemon                          # Run continuously
```

**State Files:**
- `deployments/deployment_state.json` - Current deployment state
- `deployments/deployment_log.json` - Deployment history

**Logs:**
- `logs/auto_deployer.log` - Main log
- `logs/auto_deployer_console.log` - Console output

---

### 2. System Watchdog (`system_watchdog.py`)

Monitors all critical processes and auto-restarts them if they crash.

**Monitored Processes:**
- Training workers (3x)
- Paper trading
- Autonomous AI advisor
- GPU health
- Disk space
- Database integrity

**Features:**
- Auto-restart crashed processes
- Configurable restart limits
- Cooldown period between restarts
- Health check logging
- Alert tracking

**Configuration:**
```bash
python system_watchdog.py \
    --check-interval 60 \        # Check every minute
    --max-restarts 3 \           # Max 3 restarts per process
    --restart-cooldown 300       # 5 minute cooldown
```

**State Files:**
- `deployments/watchdog_state.json` - Restart counts and alerts

**Logs:**
- `logs/watchdog.log` - Main log
- `logs/watchdog_console.log` - Console output

---

### 3. Performance Monitor (`performance_monitor.py`)

Tracks system performance and sends desktop notifications for key events.

**Monitors:**
- New best trials found
- Paper trading activity (trades executed, P&L)
- Training progress (trials/hour)
- GPU utilization and temperature
- Database growth

**Alerts:**
- 🎯 New best trial found
- 📈 Trade executed
- ⚠️  High GPU temperature (>80°C)
- ⚠️  Low GPU utilization (<50%)
- ⚠️  No trials completed in last hour

**Configuration:**
```bash
python performance_monitor.py \
    --study cappuccino_3workers_20251102_2325 \
    --check-interval 300 \       # Check every 5 minutes
    --no-notifications           # Disable desktop notifications (optional)
```

**State Files:**
- `deployments/monitor_state.json` - Performance history

**Logs:**
- `logs/performance_monitor.log` - Main log with performance reports
- `logs/performance_monitor_console.log` - Console output

---

## Integration Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING WORKERS (3x)                    │
│          Exploring hyperparameter space continuously        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
          ┌──────────────────────┐
          │  Optuna Database     │
          │  (trials, params)    │
          └─────┬───────┬────────┘
                │       │
       ┌────────┘       └─────────┐
       ↓                          ↓
┌──────────────────┐      ┌──────────────────┐
│  AUTO-DEPLOYER   │      │  PERF MONITOR    │
│  - Finds best    │      │  - Tracks stats  │
│  - Validates     │      │  - Sends alerts  │
│  - Deploys       │      │  - Notifies user │
└────────┬─────────┘      └──────────────────┘
         │
         ↓
┌─────────────────────┐
│  PAPER TRADING      │
│  - Uses best model  │
│  - Polls market     │
│  - Executes trades  │
└─────────────────────┘
         ↑
         │
┌────────┴─────────┐
│  WATCHDOG        │
│  - Monitors all  │
│  - Auto-restarts │
│  - Health checks │
└──────────────────┘
```

---

## Manual Operations

### Deploy a Specific Model
```bash
# Dry run (no auto-deploy)
python auto_model_deployer.py --no-auto-deploy

# Force deploy best model
rm -f deployments/deployment_state.json
python auto_model_deployer.py
```

### Check Deployment History
```bash
cat deployments/deployment_log.json | python -m json.tool
```

### View Performance History
```bash
cat deployments/monitor_state.json | python -m json.tool
```

### Check Recent Alerts
```bash
python3 -c "
import json
with open('deployments/watchdog_state.json', 'r') as f:
    state = json.load(f)
    for alert in state['alerts'][-10:]:
        print(f\"{alert['timestamp']}: [{alert['severity']}] {alert['process']} - {alert['message']}\")
"
```

---

## Advanced Configuration

### Adjust Deployment Threshold
Edit minimum improvement threshold to be more/less aggressive:
```bash
# More aggressive (deploy on any improvement)
python auto_model_deployer.py --min-improvement 0.1

# More conservative (only deploy on significant improvements)
python auto_model_deployer.py --min-improvement 5.0
```

### Disable Auto-Restart for Specific Process
Edit `system_watchdog.py` and set `"critical": False` for the process you want to disable auto-restart.

### Change Notification Frequency
Edit `performance_monitor.py` check interval:
```bash
# Check more frequently (every minute)
python performance_monitor.py --check-interval 60

# Check less frequently (every hour)
python performance_monitor.py --check-interval 3600
```

---

## Monitoring Commands

### Live Logs
```bash
# Auto-deployer
tail -f logs/auto_deployer.log

# Watchdog
tail -f logs/watchdog.log

# Performance monitor
tail -f logs/performance_monitor.log

# Paper trading
tail -f logs/paper_trading.log

# Training workers
tail -f logs/parallel_training/worker_1.log
```

### System Status
```bash
# All automation systems
./status_automation.sh

# Training status
sqlite3 databases/optuna_cappuccino.db "
  SELECT COUNT(*) as total,
         SUM(CASE WHEN state='COMPLETE' THEN 1 ELSE 0 END) as complete
  FROM trials t
  JOIN studies s ON t.study_id = s.study_id
  WHERE s.study_name = 'cappuccino_3workers_20251102_2325'
"

# Best trials
sqlite3 databases/optuna_cappuccino.db "
  SELECT t.number, tv.value, t.datetime_complete
  FROM trials t
  JOIN studies s ON t.study_id = s.study_id
  JOIN trial_values tv ON t.trial_id = tv.trial_id
  WHERE s.study_name = 'cappuccino_3workers_20251102_2325'
  AND t.state = 'COMPLETE'
  ORDER BY tv.value ASC
  LIMIT 5
"

# GPU status
nvidia-smi
```

---

## Troubleshooting

### Auto-Deployer Not Deploying

**Issue:** New best model found but not deployed

**Check:**
1. Minimum improvement threshold: `--min-improvement` (default 1%)
2. Deployment state: `cat deployments/deployment_state.json`
3. Model validation: Check `logs/auto_deployer.log` for validation errors

**Solution:**
```bash
# Reset deployment state to force redeploy
rm -f deployments/deployment_state.json
python auto_model_deployer.py
```

### Watchdog Not Restarting Processes

**Issue:** Process crashed but watchdog didn't restart it

**Check:**
1. Restart count: `cat deployments/watchdog_state.json | grep restart_counts`
2. Cooldown period: Default 5 minutes between restarts
3. Max restarts: Default 3 per process

**Solution:**
```bash
# Reset restart counts
python3 -c "
import json
with open('deployments/watchdog_state.json', 'r') as f:
    state = json.load(f)
state['restart_counts'] = {}
state['last_restart_times'] = {}
with open('deployments/watchdog_state.json', 'w') as f:
    json.dump(state, f, indent=2)
"
```

### Paper Trading Not Executing Trades

**Issue:** Paper trading running but no trades

**Possible Causes:**
1. Model not finding profitable signals
2. Market conditions not favorable
3. Position sizing too conservative

**Check:**
```bash
# View paper trading activity
tail -50 logs/paper_trading.log

# Check CSV for actions
tail -20 paper_trades/session_stable.csv
```

### Desktop Notifications Not Working

**Issue:** No notifications appearing

**Check:**
1. `notify-send` installed: `which notify-send`
2. Notifications enabled: Check `--no-notifications` flag not used
3. Desktop environment supports notifications

**Solution:**
```bash
# Test notification
notify-send "Test" "If you see this, notifications work"

# Install notify-send (Arch Linux)
sudo pacman -S libnotify
```

---

## Performance Tips

1. **Adjust check intervals** based on your needs:
   - Auto-deployer: 1-6 hours (new best models are rare)
   - Watchdog: 30-120 seconds (balance responsiveness vs CPU)
   - Performance monitor: 3-10 minutes (balance alerts vs noise)

2. **Deployment threshold**:
   - Start conservative (5% improvement)
   - Relax threshold (1%) after initial stability

3. **Resource usage**:
   - All automation processes use <1% CPU when idle
   - Combined memory usage: ~200MB
   - Negligible GPU impact (CPU-only monitoring)

---

## File Structure

```
cappuccino/
├── auto_model_deployer.py       # Auto-deployment system
├── system_watchdog.py            # Health monitoring & auto-restart
├── performance_monitor.py        # Performance tracking & alerts
├── start_automation.sh           # Launch all automation
├── stop_automation.sh            # Stop all automation
├── status_automation.sh          # Check automation status
├── deployments/                  # State files
│   ├── deployment_state.json     # Current deployment
│   ├── deployment_log.json       # Deployment history
│   ├── watchdog_state.json       # Restart counts & alerts
│   ├── monitor_state.json        # Performance history
│   ├── auto_deployer.pid         # Process IDs
│   ├── watchdog.pid
│   └── performance_monitor.pid
└── logs/                         # Log files
    ├── auto_deployer.log
    ├── watchdog.log
    ├── performance_monitor.log
    └── paper_trading_auto_*.log  # Auto-deployed paper trading logs
```

---

## Future Enhancements

Potential improvements:
1. Email/Slack/Discord notifications
2. Web dashboard for real-time monitoring
3. Automatic backtesting before deployment
4. Multi-model ensemble trading
5. Performance regression detection
6. Automated hyperparameter search space adjustment
7. Cloud backup integration
8. Mobile app for alerts
9. Automated profit/loss reporting
10. Risk-adjusted deployment decisions

---

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review state files in `deployments/`
3. Run `./status_automation.sh` for diagnostics
4. Check this guide's Troubleshooting section

---

**Built with ❤️ for fully automated cryptocurrency trading research**
