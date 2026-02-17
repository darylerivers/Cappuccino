# Training Monitor - CGE Augmented Models

## ✅ Training Status: RUNNING

**Started:** January 25, 2026 at 19:50
**Process ID:** 1785754
**Study:** cappuccino_maxgpu
**Trials:** 100 total
**Current Trial:** #21 (in progress)

### Training Configuration

```
Data: 8,607 timesteps (70% real + 30% CGE synthetic bear markets)
GPU: NVIDIA GeForce RTX 3070 (430MB VRAM in use)
Validation: CombPurgedKFoldCV (6 splits per trial)
Training samples per split: ~4,299 timesteps
Test samples per split: ~4,298 timesteps
```

### Expected Performance

Based on stress tests with current data:
- **Overall Sharpe:** 11.5 → 13-14 (target +13-22%)
- **Bear Market Sharpe:** 4.3 → 5.5-6.5 (target +28-51%)
- **Max Drawdown:** -22% → -15-18% (improvement)

---

## 📊 How to Monitor Training

### Real-time Log Monitoring

```bash
# Watch training progress live
tail -f training_cge.log

# Check last 100 lines
tail -100 training_cge.log

# Monitor with color (if your terminal supports it)
less -R training_cge.log
```

### Check Training Status

```bash
# Verify process is running
ps aux | grep 1_optimize_unified.py

# Check GPU usage
nvidia-smi

# Watch GPU usage continuously
watch -n 1 nvidia-smi
```

### Database Monitoring

```bash
# Check best trial so far
python3 << 'EOF'
import optuna
storage = 'sqlite:///databases/optuna_cappuccino.db'
study = optuna.load_study(study_name='cappuccino_maxgpu', storage=storage)
print(f"Best trial so far: #{study.best_trial.number}")
print(f"Best Sharpe: {study.best_value:.4f}")
print(f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
print(f"Total trials: {len(study.trials)}")
EOF
```

### Training Dashboard (Optional)

If you want a visual dashboard:
```bash
# In a separate terminal
python3 dashboard_optimized.py
# Then open browser to http://localhost:8501
```

---

## ⏱️ Estimated Timeline

**Per Trial:** ~5-8 minutes (6 splits × ~60 seconds each)
**100 Trials:** 8-13 hours total
**Expected completion:** Tomorrow morning (Jan 26, 2026, 3am-8am)

**Progress Estimates:**
- 25% complete (~25 trials): ~2-3 hours from now
- 50% complete (~50 trials): ~4-6 hours from now
- 75% complete (~75 trials): ~6-10 hours from now
- 100% complete: ~8-13 hours from now

---

## 📈 What to Look For

### Good Signs ✅

- Episode returns > 0
- Bot performance > HODL performance
- Training completes all 6 splits per trial
- No errors in logs
- GPU usage stable (~400-500MB)
- Sharpe ratios improving over baseline

### Warning Signs ⚠️

- Repeated errors in logs
- GPU memory errors
- All trials failing
- Very low or negative episode returns
- Training stuck on same trial

---

## 🛠️ Common Commands

### Monitor Progress
```bash
# Live log tail
tail -f training_cge.log

# Count completed trials
grep "Trial #" training_cge.log | wc -l

# Check for errors
grep -i "error\|exception\|failed" training_cge.log

# GPU status
nvidia-smi
```

### Check Results
```bash
# Best trial info
python3 -c "
import optuna
study = optuna.load_study(
    study_name='cappuccino_maxgpu',
    storage='sqlite:///databases/optuna_cappuccino.db'
)
print(f'Best: Trial #{study.best_trial.number}, Sharpe {study.best_value:.4f}')
"

# Trial count
python3 -c "
import optuna
study = optuna.load_study(
    study_name='cappuccino_maxgpu',
    storage='sqlite:///databases/optuna_cappuccino.db'
)
completed = len([t for t in study.trials if t.state.name == 'COMPLETE'])
print(f'{completed}/100 trials complete')
"
```

### Stop Training (if needed)
```bash
# Stop gracefully (finish current trial)
kill 1785754

# Force stop (immediate)
kill -9 1785754

# Check if stopped
ps aux | grep 1_optimize_unified.py
```

---

## 📊 Trial Output Interpretation

### Example Trial Output

```
Trial #21: 1h timeframe
================================================================================

Hyperparameters:
  LR: 0.000019          # Learning rate
  Batch: 4096           # Batch size
  Gamma: 0.9800         # Discount factor
  Net dim: 1216         # Network dimensions
  Target step: 490      # Target steps
  Break step: 50000     # Max steps

  Split 1/6...
  episode_return: 195.17    ← Total return for this split
  Bot: 0.0039               ← Bot's Sharpe ratio
  HODL: 0.0004              ← Buy-and-hold Sharpe ratio
```

**Good trial:** Bot Sharpe > HODL Sharpe, positive episode returns
**Poor trial:** Bot Sharpe < HODL Sharpe, negative episode returns

---

## 🎯 When Training Completes

Training will automatically:
1. Save best models to database
2. Write final results to log
3. Exit process

After completion:
```bash
# 1. Analyze results
python3 analyze_training_results.py --study cappuccino_maxgpu

# 2. Run stress tests
cd /home/mrc/gempack_install
python3 cappuccino_stress_test.py

# 3. Compare to baseline
# Check if improvements match expectations:
#   - Overall Sharpe improved?
#   - Bear market Sharpe +20%+?
#   - Max drawdown reduced?

# 4. If successful, deploy to paper trading
cd /opt/user-data/experiment/cappuccino
python3 auto_model_deployer.py --auto-deploy
```

---

## 🔍 Troubleshooting

### Problem: Training stuck
```bash
# Check if process is running
ps aux | grep 1_optimize_unified.py

# Check GPU
nvidia-smi

# Look for errors in log
tail -100 training_cge.log
```

### Problem: Out of memory
```bash
# Check GPU memory
nvidia-smi

# If OOM, reduce batch size and restart
# Edit training params or use --use-best-ranges flag
```

### Problem: Poor performance
- Let it run through more trials (Optuna learns over time)
- Check if data loaded correctly (8,607 timesteps)
- Verify GPU is being used

---

## 📁 Output Files

```
databases/optuna_cappuccino.db   # Trial results database
training_cge.log                 # Training log
train_results/cwd_tests/trial_*/ # Model checkpoints
nohup.out                        # Background process output
```

---

## ✅ Current Status

```
Process: RUNNING (PID 1785754)
GPU: Active (430MB VRAM)
Trial: #21 in progress
Data: 8,607 timesteps (CGE augmented)
Expected completion: ~8-13 hours from start
```

**Training is progressing normally. Check back in a few hours!**

---

## 🚀 Next Steps After Training

1. **Evaluate results** (see FULL_PROCESS_GUIDE.md Phase 3)
2. **Run stress tests** to verify improvements
3. **Deploy to paper trading** if successful
4. **Monitor for 2-4 weeks** before live deployment

Training successfully started! 🎉
