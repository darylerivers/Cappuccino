# Cappuccino System Diagnostic Guide

Quick reference for troubleshooting when systems go offline or are misconfigured.

## Quick Health Check

```bash
# Run all checks at once
./verify_system.sh        # Check configuration consistency
./status_automation.sh    # Check automation components
./status_arena.sh         # Check model arena (if running)
```

---

## Component Checklist

### 1. Training System

**Check Status:**
```bash
ps aux | grep "1_optimize_unified.py" | grep -v grep
```

**Common Issues:**

| Symptom | Cause | Fix |
|---------|-------|-----|
| No workers running | Not started or crashed | `./start_training.sh` |
| Workers on wrong study | Study name mismatch | `pkill -f 1_optimize_unified.py && ./start_training.sh` |
| Workers consuming 100% CPU | Normal during training | Monitor GPU instead: `nvidia-smi` |
| Workers died silently | OOM, GPU error, or data issue | Check `logs/worker_*.log` |
| Database locked errors | Concurrent access issue | Wait 30s and retry, or check if duplicate workers exist |

**Diagnostic Commands:**
```bash
# Check which study workers are using
ps aux | grep "1_optimize" | grep -v grep | grep -oP -- '--study-name \K[^ ]+'

# Check worker logs for errors
tail -50 logs/worker_*.log

# Check GPU availability
nvidia-smi

# Check database connectivity
sqlite3 databases/optuna_cappuccino.db "SELECT COUNT(*) FROM trials LIMIT 1;"
```

---

### 2. Auto-Model Deployer

**Check Status:**
```bash
pgrep -f "auto_model_deployer.py"
cat deployments/auto_deployer.pid  # Check PID file
```

**Common Issues:**

| Symptom | Cause | Fix |
|---------|-------|-----|
| Not running | Stopped or crashed | `./start_automation.sh` |
| Running but not deploying | No improved trials | Check logs, this may be normal |
| Wrong study | Configuration mismatch | Stop and restart automation |
| Deployment failed | Model file not found | Check trial directory exists and has actor.pth |
| PID file exists but process dead | Unclean shutdown | Remove `deployments/auto_deployer.pid` |

**Diagnostic Commands:**
```bash
# Check logs
tail -50 logs/auto_deployer.log

# Check deployment state
cat deployments/deployment_state.json | python3 -m json.tool

# Check which study it's monitoring
ps aux | grep "auto_model_deployer.py" | grep -v grep | grep -oP -- '--study \K[^ ]+'

# Verify latest deployed model exists
ls -lh train_results/cwd_tests/trial_*/actor.pth 2>/dev/null | tail -1
```

---

### 3. System Watchdog

**Check Status:**
```bash
pgrep -f "system_watchdog.py"
cat deployments/watchdog.pid
```

**Common Issues:**

| Symptom | Cause | Fix |
|---------|-------|-----|
| Not running | Stopped or crashed | `./start_automation.sh` |
| Excessive restarts | Underlying component keeps crashing | Check watchdog logs and fix root cause |
| Not restarting failed components | Restart cooldown active | Wait 5 minutes or manually restart |
| High CPU usage | Checking too frequently | Normal if check_interval is low |

**Diagnostic Commands:**
```bash
# Check recent alerts
tail -50 logs/watchdog.log

# Check watchdog state
cat deployments/watchdog_state.json | python3 -m json.tool

# Check restart history
python3 -c "
import json
with open('deployments/watchdog_state.json') as f:
    state = json.load(f)
    restarts = state.get('restart_history', [])
    for r in restarts[-10:]:
        print(f\"{r['timestamp']}: {r['process']} - {r['reason']}\")
"
```

---

### 4. Performance Monitor

**Check Status:**
```bash
pgrep -f "performance_monitor.py"
cat deployments/performance_monitor.pid
```

**Common Issues:**

| Symptom | Cause | Fix |
|---------|-------|-----|
| Not running | Stopped or crashed | `./start_automation.sh` |
| No metrics logged | No completed trials yet | Wait for trials to complete |
| Wrong study | Configuration mismatch | Stop and restart automation |
| Old metrics | Check interval too long | Normal if set to 300s (5 min) |

**Diagnostic Commands:**
```bash
# Check logs
tail -50 logs/performance_monitor.log

# Check which study it's monitoring
ps aux | grep "performance_monitor.py" | grep -v grep | grep -oP -- '--study \K[^ ]+'

# Check latest performance data
ls -lh deployments/performance_history.json
cat deployments/performance_history.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f\"Total snapshots: {len(data)}\")
if data:
    latest = data[-1]
    print(f\"Latest: {latest['timestamp']}\")
    print(f\"Best value: {latest.get('best_value', 'N/A')}\")
"
```

---

### 5. Ensemble Auto-Updater

**Check Status:**
```bash
pgrep -f "ensemble_auto_updater.py"
cat deployments/ensemble_updater.pid
```

**Common Issues:**

| Symptom | Cause | Fix |
|---------|-------|-----|
| Not running | Stopped or crashed | `./start_automation.sh` |
| Ensemble not updating | Not enough trials yet | Need at least 20 completed trials |
| Wrong study | Configuration mismatch | Stop and restart automation |
| Old ensemble | Update interval too long | Normal if set to 600s (10 min) |
| Missing models | Trials pruned or failed | Check trial directories exist |

**Diagnostic Commands:**
```bash
# Check logs
tail -50 logs/ensemble_updater_console.log

# Check ensemble state
cat train_results/ensemble/ensemble_state.json | python3 -m json.tool

# Count ensemble members
ls train_results/ensemble/model_* 2>/dev/null | wc -l

# Check which study it's monitoring
ps aux | grep "ensemble_auto_updater.py" | grep -v grep | grep -oP -- '--study \K[^ ]+'

# Verify ensemble models exist
for i in {1..5}; do
    if [ -f "train_results/ensemble/model_${i}_actor.pth" ]; then
        echo "Model $i: OK"
    else
        echo "Model $i: MISSING"
    fi
done
```

---

### 6. Paper Trader

**Check Status:**
```bash
pgrep -f "paper_trader_alpaca_polling.py"
cat deployments/paper_trading.pid
```

**Common Issues:**

| Symptom | Cause | Fix |
|---------|-------|-----|
| Not running | Not started or failed | `./paper_trading_failsafe.sh` |
| Restarting frequently | API errors or network issues | Check `logs/paper_trading_live.log` |
| No trades executed | Low confidence or market closed | Normal, check market hours |
| API authentication failed | Invalid Alpaca keys | Check `.env` file has correct `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` |
| Model not found | Wrong model directory | Check model path in failsafe script |
| Out of buying power | Positions at max allocation | Reduce position sizes or close positions |

**Diagnostic Commands:**
```bash
# Check live trading log
tail -100 logs/paper_trading_live.log

# Check failsafe wrapper log
tail -50 logs/paper_trading_failsafe.log

# Check restart history
cat deployments/paper_trading_state.json | python3 -m json.tool

# Test Alpaca API connection
python3 -c "
from alpaca.trading.client import TradingClient
import os
from dotenv import load_dotenv
load_dotenv()
try:
    client = TradingClient(os.getenv('APCA_API_KEY_ID'), os.getenv('APCA_API_SECRET_KEY'), paper=True)
    account = client.get_account()
    print(f'✓ API Connected')
    print(f'  Buying Power: \${float(account.buying_power):,.2f}')
    print(f'  Portfolio Value: \${float(account.portfolio_value):,.2f}')
except Exception as e:
    print(f'✗ API Error: {e}')
"

# Check current positions
python3 -c "
from alpaca.trading.client import TradingClient
import os
from dotenv import load_dotenv
load_dotenv()
try:
    client = TradingClient(os.getenv('APCA_API_KEY_ID'), os.getenv('APCA_API_SECRET_KEY'), paper=True)
    positions = client.get_all_positions()
    print(f'Open positions: {len(positions)}')
    for p in positions:
        print(f'  {p.symbol}: {p.qty} @ \${float(p.avg_entry_price):.2f}')
except Exception as e:
    print(f'Error: {e}')
"
```

---

### 7. Autonomous AI Advisor

**Check Status:**
```bash
pgrep -f "ollama_autonomous_advisor.py"
cat logs/autonomous_advisor.pid
```

**Common Issues:**

| Symptom | Cause | Fix |
|---------|-------|-----|
| Not running | Not started | `./start_autonomous_advisor.sh` |
| No analysis generated | Not enough new trials | Wait for analysis interval threshold |
| Ollama connection failed | Ollama not running | `systemctl status ollama` or `ollama serve` |
| Model not found | Model not pulled | `ollama pull qwen2.5-coder:7b` |
| Suggestions not applied | Test trials still running | Wait for test completion |

**Diagnostic Commands:**
```bash
# Check logs
tail -100 logs/autonomous_advisor.log

# Check advisor state
cat analysis_reports/advisor_state.json | python3 -m json.tool

# Test Ollama connection
curl -s http://localhost:11434/api/tags | python3 -m json.tool

# Check available models
ollama list

# Check recent analysis reports
ls -lht analysis_reports/ollama_analysis_*.txt | head -5
```

---

### 8. Model Arena (Competition System)

**Check Status:**
```bash
./status_arena.sh
```

**Common Issues:**

| Symptom | Cause | Fix |
|---------|-------|-----|
| Not running | Not started | `./start_arena.sh` |
| No battles recorded | Not enough models | Need at least 2 models |
| Ratings not updating | No battles completed | Wait for evaluation cycles |
| High memory usage | Too many models loaded | Normal for arena, reduce model count |

**Diagnostic Commands:**
```bash
# Check arena logs
tail -50 logs/model_arena.log

# Check arena state
cat arena_state/arena_state.json | python3 -m json.tool

# Check leaderboard
python3 -c "
import json
with open('arena_state/arena_state.json') as f:
    state = json.load(f)
    models = state.get('models', {})
    sorted_models = sorted(models.items(), key=lambda x: x[1]['rating'], reverse=True)
    print('Model Arena Leaderboard:')
    for name, data in sorted_models[:10]:
        print(f\"  {data['rating']:.1f} - {name} (battles: {data['battles']})\")
"
```

---

## Configuration Issues

### Study Name Mismatch

**Symptom:** Components using different study names

**Check:**
```bash
./verify_system.sh
```

**Fix:**
```bash
# 1. Stop everything
./stop_automation.sh
pkill -f 1_optimize_unified.py

# 2. Verify .env.training has correct study
cat .env.training

# 3. Restart everything
./start_automation.sh
./start_training.sh
```

---

### Missing .env.training File

**Symptom:** `ERROR: .env.training not found!`

**Fix:**
```bash
# Create from template
cat > .env.training << 'EOF'
# Active Optuna Study Configuration
ACTIVE_STUDY_NAME="your_study_name_here"

# Auto-Deployer Settings
DEPLOYER_CHECK_INTERVAL=3600
DEPLOYER_MIN_IMPROVEMENT=1.0

# Ensemble Settings
ENSEMBLE_TOP_N=20
ENSEMBLE_UPDATE_INTERVAL=600
EOF

# Edit with your study name
nano .env.training
```

---

### Database Connectivity Issues

**Symptom:** `database is locked` or `unable to open database file`

**Diagnostic:**
```bash
# Check database exists
ls -lh databases/optuna_cappuccino.db

# Check permissions
stat databases/optuna_cappuccino.db

# Check for lock files
ls -la databases/*.lock 2>/dev/null

# Test database access
sqlite3 databases/optuna_cappuccino.db "SELECT COUNT(*) FROM studies;"
```

**Fix:**
```bash
# If locked, find and stop conflicting processes
lsof databases/optuna_cappuccino.db

# Remove stale lock files (only if no processes are using DB)
rm -f databases/*.lock

# Fix permissions if needed
chmod 664 databases/optuna_cappuccino.db
```

---

### Alpaca API Issues

**Symptom:** API authentication errors or connection failures

**Diagnostic:**
```bash
# Check .env file exists and has keys
grep "APCA_API_KEY_ID" .env
grep "APCA_API_SECRET_KEY" .env

# Test API connection (see Paper Trader section above)
```

**Fix:**
```bash
# Ensure .env has valid keys
cat > .env << 'EOF'
APCA_API_KEY_ID="your_key_id"
APCA_API_SECRET_KEY="your_secret_key"
EOF

# Test connection
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

## Resource Issues

### GPU Not Available

**Symptom:** Training fails with CUDA errors

**Diagnostic:**
```bash
# Check GPU status
nvidia-smi

# Check CUDA availability in Python
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check GPU processes
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
```

**Fix:**
```bash
# If GPU hung, reset it (requires root)
sudo nvidia-smi --gpu-reset

# If out of memory, kill GPU processes
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs kill

# Reduce batch size in training config if OOM persists
```

---

### Disk Space Issues

**Symptom:** Training crashes or logs can't be written

**Diagnostic:**
```bash
# Check disk usage
df -h

# Check directory sizes
du -sh train_results/ logs/ databases/

# Find large files
find . -type f -size +1G 2>/dev/null
```

**Fix:**
```bash
# Clean old logs
find logs/ -name "*.log" -mtime +30 -delete

# Archive old training results
./archive_manager.py --archive-before 30

# Remove failed trials
find train_results/cwd_tests/ -name "*.log" -size 0 -delete
```

---

### Memory Issues

**Symptom:** Processes killed by OOM killer

**Diagnostic:**
```bash
# Check memory usage
free -h

# Check process memory
ps aux --sort=-%mem | head -10

# Check system logs for OOM events
journalctl -k | grep -i "out of memory"
```

**Fix:**
```bash
# Reduce number of training workers
# Edit start_training.sh to reduce worker count

# Reduce replay buffer size in training config

# Add swap space if needed (requires root)
sudo dd if=/dev/zero of=/swapfile bs=1G count=8
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## Emergency Recovery

### Complete System Reset

```bash
# 1. Stop everything
./stop_automation.sh
./stop_autonomous_advisor.sh
./stop_arena.sh
pkill -f 1_optimize_unified.py
pkill -f paper_trader

# 2. Wait for processes to terminate
sleep 5

# 3. Force kill if needed
pkill -9 -f "auto_model_deployer.py"
pkill -9 -f "system_watchdog.py"
pkill -9 -f "performance_monitor.py"
pkill -9 -f "ensemble_auto_updater.py"

# 4. Clean PID files
rm -f deployments/*.pid logs/*.pid

# 5. Verify config
cat .env.training

# 6. Restart core systems
./start_automation.sh

# 7. Restart training (if needed)
./start_training.sh

# 8. Verify everything
./verify_system.sh
./status_automation.sh
```

---

### Recover from Corrupted State Files

```bash
# Backup first
mkdir -p backups
cp -r deployments/ backups/deployments_$(date +%Y%m%d_%H%M%S)
cp -r arena_state/ backups/arena_state_$(date +%Y%m%d_%H%M%S) 2>/dev/null

# Remove corrupted state files
rm -f deployments/*_state.json
rm -f arena_state/arena_state.json

# Restart systems (they will recreate state files)
./stop_automation.sh
./start_automation.sh
```

---

### Recover from Database Corruption

```bash
# Backup database
cp databases/optuna_cappuccino.db databases/optuna_cappuccino_backup_$(date +%Y%m%d_%H%M%S).db

# Check integrity
sqlite3 databases/optuna_cappuccino.db "PRAGMA integrity_check;"

# If corrupted, try to repair
sqlite3 databases/optuna_cappuccino.db ".recover" | sqlite3 databases/optuna_cappuccino_repaired.db

# If repair fails, may need to start new study
# (Contact support or check Optuna documentation)
```

---

## Monitoring Best Practices

### Daily Health Check
```bash
# Run these every day
./verify_system.sh
./status_automation.sh
nvidia-smi  # Check GPU health
df -h       # Check disk space
```

### Weekly Maintenance
```bash
# Archive old results
./archive_manager.py --archive-before 30

# Clean old logs
find logs/ -name "*.log" -mtime +30 -delete

# Check database size
du -sh databases/

# Review performance trends
python3 -c "
import json
with open('deployments/performance_history.json') as f:
    data = json.load(f)
    recent = data[-168:]  # Last week (hourly checks)
    values = [x.get('best_value', 0) for x in recent if 'best_value' in x]
    if values:
        print(f'Week best: {max(values):.6f}')
        print(f'Week avg:  {sum(values)/len(values):.6f}')
"
```

---

## Getting Help

### Log Locations
- Training workers: `logs/worker_*.log`
- Auto-deployer: `logs/auto_deployer.log`
- Watchdog: `logs/watchdog.log`
- Performance monitor: `logs/performance_monitor.log`
- Ensemble updater: `logs/ensemble_updater_console.log`
- Paper trader: `logs/paper_trading_live.log`
- AI Advisor: `logs/autonomous_advisor.log`
- Model Arena: `logs/model_arena.log`

### State Files
- Deployment: `deployments/deployment_state.json`
- Watchdog: `deployments/watchdog_state.json`
- Performance: `deployments/performance_history.json`
- Ensemble: `train_results/ensemble/ensemble_state.json`
- AI Advisor: `analysis_reports/advisor_state.json`
- Arena: `arena_state/arena_state.json`

### Useful Commands
```bash
# Watch all logs simultaneously
tail -f logs/*.log

# Monitor system in real-time
watch -n 5 './status_automation.sh'

# Monitor GPU in real-time
watch -n 2 nvidia-smi

# Check recent errors across all logs
grep -i error logs/*.log | tail -20
```
