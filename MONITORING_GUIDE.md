# Training Monitoring Guide

## 🎯 Quick Monitoring Commands

### 1. **Custom Dashboard** (Recommended)
```bash
cd /home/mrc/experiment/cappuccino

# One-time snapshot
python monitor.py

# Continuous monitoring (updates every 5 seconds)
python monitor.py --watch

# Specific study
python monitor.py --study-name my_study --watch
```

### 2. **GPU Monitoring**
```bash
# Continuous GPU monitoring (updates every 2 seconds)
watch -n 2 nvidia-smi

# Or more detailed
watch -n 2 'nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader'
```

### 3. **Database Queries**
```bash
# Count completed trials
sqlite3 databases/optuna_cappuccino.db "SELECT COUNT(*) FROM trials WHERE state='COMPLETE'"

# List all trials
sqlite3 databases/optuna_cappuccino.db "SELECT trial_id, state FROM trials ORDER BY trial_id"

# View latest 5 trials
sqlite3 databases/optuna_cappuccino.db "SELECT trial_id, state, datetime_start FROM trials ORDER BY trial_id DESC LIMIT 5"
```

### 4. **Process Monitoring**
```bash
# Check if training is running
ps aux | grep "1_optimize_unified.py"

# Watch CPU/Memory usage
top -p $(pgrep -f 1_optimize_unified.py)
```

### 5. **Log Files**
```bash
# View latest training output
ls -lt train_results/cwd_tests/

# Tail a specific trial's log
tail -f train_results/cwd_tests/trial_0_1h/recorder.log
```

## 📊 What to Monitor

### GPU Utilization
- **Target:** 60-90% during training steps
- **Normal:** 20-40% during evaluation/data loading
- **Problem:** <10% consistently (check for errors)

### Memory Usage
- **Normal:** 2000-4000 MiB
- **Maximum:** ~6000 MiB (leaves room for system)
- **Problem:** Approaching 8000 MiB (OOM risk)

### Temperature
- **Safe:** <75°C
- **Warning:** 75-85°C
- **Critical:** >85°C (check cooling)

### Training Progress
- **Each split:** ~1-2 minutes (120,000 steps)
- **Each trial:** ~10-15 minutes (6 splits)
- **100 trials:** ~20-30 hours total

## 🔍 Detailed Monitoring Methods

### Method 1: Real-Time Output (Most Immediate)

Since you're running with `python -u`, you can check the latest output:

```bash
# If running in terminal, just look at the output

# If in background, check process ID and tail output
# (This depends on how you started it)
```

### Method 2: Python Script for Analysis

Create a quick analysis script:
```python
import sqlite3
import optuna

# Load study
study = optuna.load_study(
    study_name="cappuccino_production",
    storage="sqlite:///databases/optuna_cappuccino.db"
)

# Print summary
print(f"Study: {study.study_name}")
print(f"Direction: {study.direction}")
print(f"Total trials: {len(study.trials)}")
print(f"Complete: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
print(f"Running: {len([t for t in study.trials if t in study.trials if t.state == optuna.trial.TrialState.RUNNING])}")

if len(study.trials) > 0:
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if complete_trials:
        best = max(complete_trials, key=lambda t: t.value)
        print(f"\nBest trial: #{best.number}")
        print(f"Best value: {best.value:.6f}")
        print(f"Parameters:")
        for k, v in best.params.items():
            print(f"  {k}: {v}")
```

### Method 3: tmux/screen Session

For persistent monitoring:
```bash
# Create a monitoring session
tmux new -s monitor

# Inside tmux, run continuous monitor
python monitor.py --watch

# Detach: Ctrl+B then D
# Reattach: tmux attach -t monitor
```

### Method 4: Grafana-style Dashboard (Advanced)

If you want web-based monitoring, use Optuna's dashboard:
```bash
# Install Optuna dashboard
pip install optuna-dashboard

# Start dashboard
optuna-dashboard sqlite:///databases/optuna_cappuccino.db

# Access at http://localhost:8080
```

## 📈 Performance Indicators

### Good Training Session
```
✅ GPU Utilization: 60-90% during training
✅ Memory: Stable around 2-4GB
✅ Temperature: <75°C
✅ Each split completes in 1-3 minutes
✅ No errors in logs
✅ Sharpe ratios being calculated
```

### Potential Issues
```
❌ GPU <20% for extended periods → Check for hangs
❌ Memory growing continuously → Memory leak
❌ Temperature >80°C → Cooling issue
❌ Splits failing → Data/parameter issues
❌ No new trials after 30 min → Hung process
```

## 🛠️ Troubleshooting Commands

### Check if process is alive
```bash
ps aux | grep "1_optimize_unified" | grep -v grep
```

### Check GPU is accessible
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Check database
```bash
sqlite3 databases/optuna_cappuccino.db ".tables"
sqlite3 databases/optuna_cappuccino.db "SELECT COUNT(*) FROM trials"
```

### Check disk space
```bash
df -h .
du -sh train_results/
du -sh databases/
```

## 📱 Remote Monitoring

If you need to monitor remotely:

### SSH + Watch
```bash
ssh your-server "cd /home/mrc/experiment/cappuccino && python monitor.py"
```

### Periodic Email Updates
Add to cron:
```bash
*/30 * * * * cd /home/mrc/experiment/cappuccino && python monitor.py | mail -s "Training Update" you@email.com
```

## 🎯 Current Training Status

Your training is currently:
- **Trial:** #0, Split 4/6
- **GPU:** 43% utilization
- **Memory:** 2124 MiB
- **Status:** Running normally ✅

## 📌 Quick Reference Card

```
┌─────────────────────────────────────────────────┐
│ MONITORING CHEAT SHEET                          │
├─────────────────────────────────────────────────┤
│ Dashboard:     python monitor.py --watch        │
│ GPU:           watch -n 2 nvidia-smi            │
│ Trials:        sqlite3 databases/...db "..."    │
│ Processes:     ps aux | grep python             │
│ Logs:          tail -f train_results/.../log    │
│ Output:        Check background process         │
└─────────────────────────────────────────────────┘
```

## 🔗 Related Files
- `monitor.py` - Custom monitoring dashboard
- `databases/optuna_cappuccino.db` - Training database
- `train_results/cwd_tests/` - Model checkpoints and logs
- `PRODUCTION_READY.md` - Complete technical docs
