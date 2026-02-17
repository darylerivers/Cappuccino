# Quick Diagnostic Reference Card

## First Response Checklist

When something goes wrong, run these commands in order:

```bash
# 1. Quick automated diagnostic
./diagnose.sh

# 2. Check configuration consistency
./verify_system.sh

# 3. Check component status
./status_automation.sh
```

---

## Common Problems & Solutions

### Problem: "Nothing is running"
```bash
# Check config exists
cat .env.training

# Start automation
./start_automation.sh

# Start training (if needed)
./start_training.sh

# Verify
./diagnose.sh
```

---

### Problem: "Training workers crashed"
```bash
# Check what happened
tail -100 logs/worker_*.log

# Check GPU
nvidia-smi

# Restart
pkill -f 1_optimize_unified.py
./start_training.sh
```

---

### Problem: "Paper trader keeps restarting"
```bash
# Check error log
tail -100 logs/paper_trading_live.log

# Common causes:
# - API authentication failed → Check .env file
# - Model not found → Check model directory exists
# - Network issues → Check internet connection

# Test API manually
python3 -c "
from alpaca.trading.client import TradingClient
import os
from dotenv import load_dotenv
load_dotenv()
client = TradingClient(os.getenv('APCA_API_KEY_ID'), os.getenv('APCA_API_SECRET_KEY'), paper=True)
print(client.get_account())
"
```

---

### Problem: "Components using wrong study"
```bash
# 1. Stop everything
./stop_automation.sh
pkill -f 1_optimize_unified.py

# 2. Check/fix config
cat .env.training
nano .env.training  # Edit if needed

# 3. Restart
./start_automation.sh
./start_training.sh

# 4. Verify
./verify_system.sh
```

---

### Problem: "Database locked"
```bash
# Find what's locking it
lsof databases/optuna_cappuccino.db

# Wait a moment and retry (usually resolves itself)
sleep 30

# If persistent, check for duplicate workers
ps aux | grep 1_optimize
```

---

### Problem: "Out of disk space"
```bash
# Check usage
df -h
du -sh train_results/ logs/ databases/

# Quick cleanup
find logs/ -name "*.log" -mtime +30 -delete
./archive_manager.py --archive-before 30

# Find space hogs
du -sh train_results/cwd_tests/trial_* | sort -h | tail -20
```

---

### Problem: "GPU not working"
```bash
# Check GPU status
nvidia-smi

# Check CUDA in Python
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# If hung, reset (requires root)
sudo nvidia-smi --gpu-reset

# If out of memory
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs kill
```

---

### Problem: "Ensemble not updating"
```bash
# Check updater is running
pgrep -f ensemble_auto_updater.py

# Check logs
tail -50 logs/ensemble_updater_console.log

# Check state
cat train_results/ensemble/ensemble_state.json | python3 -m json.tool

# Needs at least 20 completed trials
sqlite3 databases/optuna_cappuccino.db "
SELECT COUNT(*) FROM trials t
JOIN studies s ON t.study_id = s.study_id
WHERE s.study_name = '$(cat .env.training | grep ACTIVE_STUDY_NAME | cut -d= -f2)'
AND t.state = 'COMPLETE';
"
```

---

## Emergency Commands

### Nuclear Reset (stops and restarts everything)
```bash
# WARNING: This stops ALL processes

# Stop
./stop_automation.sh
./stop_autonomous_advisor.sh
pkill -f 1_optimize_unified.py
pkill -f paper_trader
sleep 5

# Force kill if needed
pkill -9 -f ".py"

# Clean PIDs
rm -f deployments/*.pid logs/*.pid

# Restart
./start_automation.sh
./start_training.sh

# Verify
./diagnose.sh
./verify_system.sh
```

---

### Backup Important Data
```bash
# Before major changes, backup:

# 1. Database
cp databases/optuna_cappuccino.db backups/optuna_backup_$(date +%Y%m%d_%H%M%S).db

# 2. Best models
mkdir -p backups/models_$(date +%Y%m%d)
cp -r train_results/ensemble backups/models_$(date +%Y%m%d)/

# 3. State files
cp -r deployments/ backups/deployments_$(date +%Y%m%d_%H%M%S)/
```

---

## Key Log Files

| Component | Log Location |
|-----------|-------------|
| Training | `logs/worker_*.log` |
| Auto-deployer | `logs/auto_deployer.log` |
| Watchdog | `logs/watchdog.log` |
| Performance monitor | `logs/performance_monitor.log` |
| Ensemble updater | `logs/ensemble_updater_console.log` |
| Paper trader | `logs/paper_trading_live.log` |
| Paper trading failsafe | `logs/paper_trading_failsafe.log` |
| AI Advisor | `logs/autonomous_advisor.log` |

## Key State Files

| Data | File Location |
|------|---------------|
| Deployment state | `deployments/deployment_state.json` |
| Watchdog state | `deployments/watchdog_state.json` |
| Performance history | `deployments/performance_history.json` |
| Ensemble state | `train_results/ensemble/ensemble_state.json` |
| AI Advisor state | `analysis_reports/advisor_state.json` |

---

## Monitoring Commands

```bash
# Watch all logs
tail -f logs/*.log

# Watch specific component
tail -f logs/paper_trading_live.log

# Watch automation status
watch -n 5 './status_automation.sh'

# Watch GPU
watch -n 2 nvidia-smi

# Search for errors
grep -i error logs/*.log | tail -20

# Check system health every 5 minutes
watch -n 300 './diagnose.sh'
```

---

## Health Check Schedule

### Every Day
- Run `./diagnose.sh`
- Check `nvidia-smi` for GPU health
- Check `df -h` for disk space

### Every Week
- Archive old results: `./archive_manager.py --archive-before 30`
- Clean old logs: `find logs/ -name "*.log" -mtime +30 -delete`
- Review performance trends

### Every Month
- Backup database
- Review and clean up old trials
- Update documentation if workflows changed

---

## Getting Help

1. Run diagnostics: `./diagnose.sh`
2. Check full guide: `DIAGNOSTIC_GUIDE.md`
3. Review logs in `logs/` directory
4. Check state files in `deployments/` directory

## Status Codes

| Code | Meaning |
|------|---------|
| ✓ (green) | Working correctly |
| ⚠ (yellow) | Warning - may need attention |
| ✗ (red) | Error - needs fixing |
| ✗✗ (red) | Critical - system won't work |
