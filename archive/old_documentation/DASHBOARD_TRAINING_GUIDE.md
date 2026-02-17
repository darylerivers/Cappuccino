# Dashboard & Training Guide

## Dashboard Updates

### ✓ Dynamic Study Detection
The dashboard now automatically detects which study is currently running and displays its name. No more hardcoded study names!

### ✓ Trailing Stop Loss Display
Top trials now show their trailing stop percentage:
```
Top 5 Trials:
  #  0: +0.015235 | TS: 8%
  #  1: +0.012626 | TS: 12%
  #  2: +0.012550 | TS: 15%
```

### ✓ New Page 4: Training Statistics

Access with the **→ Right Arrow** key (cycles through pages).

**Training Statistics Page Shows:**

1. **Trial Summary**
   - Total trials, completed, running, failed
   - Best value and top 10% cutoff

2. **Trailing Stop Loss Statistics**
   - Number of trials with trailing stop
   - Average trailing stop percentage
   - Min/max range

3. **Learning Rate Statistics**
   - Average learning rate
   - Range explored

4. **Trailing Stop vs Performance Correlation**
   - Table showing which trailing stop % performs best
   - Grouped by stop percentage with average values

5. **Top 10 Trials**
   - Extended list with trailing stop info

## Training Automation Script

### Quick Restart
```bash
./restart_training.sh
```

This will:
1. Stop all old training workers
2. Generate a new timestamped study name
3. Start 3 new workers with the latest code
4. Verify workers are running

### Advanced Options

**Continue Existing Study:**
```bash
./restart_training.sh --continue-study
```

**Custom Study Name:**
```bash
./restart_training.sh --study-name my_experiment_v2
```

**Different Number of Workers:**
```bash
./restart_training.sh --workers 5 --n-trials 200
```

**Use Different GPU:**
```bash
./restart_training.sh --gpu 1
```

**Combine Options:**
```bash
./restart_training.sh --study-name trailing_stop_test --workers 4 --n-trials 150 --gpu 0
```

### Get Help
```bash
./restart_training.sh --help
```

## Monitoring Workflow

### 1. Start Training
```bash
./restart_training.sh --study-name cappuccino_test_$(date +%Y%m%d)
```

### 2. Monitor with Dashboard
```bash
python dashboard.py
```

**Navigation:**
- `→` Right Arrow: Next page
- `←` Left Arrow: Previous page
- `q` or `Ctrl+C`: Exit

**Pages:**
1. **Main Dashboard** - Overview of all systems
2. **Ensemble Voting** - Model predictions and consensus
3. **Portfolio History** - Performance charts
4. **Training Statistics** - Detailed training metrics (NEW!)

### 3. Check Training Logs
```bash
# Live tail
tail -f logs/training_worker1_*.log

# Check recent errors
grep -i error logs/training_worker*.log
```

### 4. Query Database Directly
```bash
# Count trials
sqlite3 databases/optuna_cappuccino.db \
  "SELECT COUNT(*) FROM trials WHERE study_id = (SELECT study_id FROM studies WHERE study_name = 'YOUR_STUDY_NAME');"

# Get trailing stop distribution
sqlite3 databases/optuna_cappuccino.db \
  "SELECT param_value, COUNT(*) FROM trial_params WHERE param_name = 'trailing_stop_pct' GROUP BY param_value;"
```

## Trailing Stop Loss Feature

### What It Does
During training, each trial tests a different trailing stop percentage (2%-25%). The agent learns to:
- Protect profits by exiting when price drops X% from peak
- Let winning positions run to maximize gains
- Cut losing positions at optimal points

### How It's Optimized
Optuna explores different trailing stop percentages and finds the optimal value that maximizes the Sharpe ratio minus variance.

### Viewing Results
1. **Main Dashboard** - See trailing stop % for top 5 trials
2. **Training Statistics** (Page 4) - See correlation between trailing stop % and performance

### Example Output
```
TRAILING STOP vs PERFORMANCE

Stop %  | Avg Value  | Trials
─────────────────────────────────
    2%  | -0.002341 |   3
    5%  | +0.008234 |   5
    8%  | +0.015235 |   4
   10%  | +0.012250 |   6
   12%  | +0.012626 |   3
   15%  | +0.012550 |   2
```

This shows that 8% trailing stop had the best average performance!

## Best Practices

### 1. Fresh Start
Always restart training when you update the code:
```bash
./restart_training.sh
```

### 2. Monitor First Hour
Check dashboard frequently in the first hour to ensure:
- All workers are running
- Trials are completing successfully
- No errors in logs

### 3. Let It Run
Training needs time to explore the hyperparameter space. Let it run for at least 50-100 trials to see patterns.

### 4. Check Statistics Page
After 20+ trials with trailing stop, check Page 4 to see which percentages are working best.

### 5. Deploy Best Models
The auto-deployer will automatically use top-performing trials in the ensemble. Check the main dashboard to see which trials are deployed (* marker).

## Troubleshooting

### Workers Not Starting
```bash
# Check for errors
cat logs/training_worker1_*.log | tail -50

# Verify Python environment
which python
python --version
```

### Database Locked
```bash
# Stop all processes using the database
pkill -f "1_optimize_unified"
pkill -f "dashboard.py"

# Wait a moment, then restart
sleep 2
./restart_training.sh
```

### Out of Memory
```bash
# Reduce number of workers
./restart_training.sh --workers 2

# Or reduce batch size in the code (1_optimize_unified.py)
```

### GPU Not Being Used
```bash
# Check GPU availability
nvidia-smi

# Verify CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES

# Force specific GPU
./restart_training.sh --gpu 0
```

## Quick Reference Commands

```bash
# Start training
./restart_training.sh

# Monitor everything
python dashboard.py

# Stop training
pkill -f "1_optimize_unified"

# View logs
tail -f logs/training_worker1_*.log

# Check trials
sqlite3 databases/optuna_cappuccino.db "SELECT study_name FROM studies ORDER BY study_id DESC LIMIT 5;"

# Test dashboard once (no loop)
python dashboard.py --once
```

## What's Next?

1. Monitor training statistics on Page 4
2. Identify optimal trailing stop percentage
3. Best trials automatically added to ensemble
4. Paper trading uses ensemble with learned trailing stops
5. Review performance in portfolio history (Page 3)

Happy trading! 🚀
