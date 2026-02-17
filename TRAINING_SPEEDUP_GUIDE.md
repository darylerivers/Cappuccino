# Training Speedup Implementation Guide

## Current Performance
- **14.7 minutes per trial** × 1000 trials = **10.2 days total**
- Best Sharpe: **0.1720** (Trial #14)
- GPU: AMD RX 7900 GRE (16GB VRAM, using ~6GB per trial)

## Target Performance
- **2-3 days total** (3-4x speedup needed)
- Maintain Sharpe: **>= 0.15**

---

## Implementation Steps

### Step 1: Add Pruner (2-3x speedup)

**Location:** `scripts/training/1_optimize_unified.py` ~line 715

**Current Code:**
```python
study = optuna.create_study(
    study_name=args.study_name,
    storage=args.storage,
    direction='maximize',
    load_if_exists=True
)
```

**Replace With:**
```python
import optuna.pruners as pruners

# MedianPruner: Kill trials worse than median after n_startup_trials
pruner = pruners.MedianPruner(
    n_startup_trials=5,      # Don't prune first 5 trials (build baseline)
    n_warmup_steps=10,       # Wait 10 steps before pruning
    interval_steps=5         # Check every 5 steps
)

# Alternative: HyperbandPruner (more aggressive)
# pruner = pruners.HyperbandPruner(
#     min_resource=10,
#     max_resource=100,
#     reduction_factor=3
# )

study = optuna.create_study(
    study_name=args.study_name,
    storage=args.storage,
    direction='maximize',
    load_if_exists=True,
    pruner=pruner,
    sampler=optuna.samplers.TPESampler(
        seed=42,
        multivariate=True,
        n_startup_trials=10
    )
)
```

---

### Step 2: Add Intermediate Reporting

**Location:** `utils/function_train_test.py` in the training loop

**Find:** The episode loop where agent trains

**Add After Each Episode:**
```python
# Inside episode loop (after each episode completes)
if hasattr(trial, 'report') and trial is not None:
    # Calculate intermediate Sharpe ratio
    current_rewards = rewards_history[-20:]  # Last 20 episodes
    if len(current_rewards) > 5:
        mean_reward = np.mean(current_rewards)
        std_reward = np.std(current_rewards)
        if std_reward > 0:
            intermediate_sharpe = mean_reward / std_reward
            trial.report(intermediate_sharpe, step=episode_num)

            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
```

**Note:** You'll need to pass `trial` object to the training function:
```python
def train_and_test(agent, env, ..., trial=None):
    # existing code
```

---

### Step 3: Enable Parallel Trials

**Location:** `scripts/training/1_optimize_unified.py` ~line 770

**Current Code:**
```python
study.optimize(objective_wrapper, n_trials=args.n_trials, show_progress_bar=False, callbacks=[discord_callback])
```

**Replace With:**
```python
# Run 2 trials in parallel (GPU has 16GB, each uses ~6GB)
# n_jobs=2 means 2 workers, each running 1 trial
# n_jobs=-1 means use all CPUs (DON'T DO THIS - GPU will OOM)

study.optimize(
    objective_wrapper,
    n_trials=args.n_trials,
    n_jobs=2,  # Start with 2, can try 3 if no OOM
    show_progress_bar=False,
    callbacks=[discord_callback]
)
```

**Important GPU Memory Check:**
```bash
# While training, monitor GPU memory:
watch -n 1 rocm-smi

# If you see OOM errors, reduce to n_jobs=1
```

---

### Step 4: Reduce Trial Budget

**Location:** Command line when starting training

**Current:**
```bash
python scripts/training/1_optimize_unified.py \
  --n-trials 1000 \
  --study-name cappuccino_ft_16gb_optimized \
  --force-ft --timeframe 1h --data-dir data/1h_1680
```

**Optimized:**
```bash
python scripts/training/1_optimize_unified.py \
  --n-trials 400 \              # Reduced from 1000
  --use-best-ranges \           # Use tightened hyperparameter ranges
  --study-name cappuccino_ft_16gb_optimized \
  --force-ft --timeframe 1h --data-dir data/1h_1680
```

---

### Step 5: Initialize with Best Known Params

**Location:** `scripts/training/1_optimize_unified.py` in objective function

**Add after creating study, before optimize:**
```python
# Enqueue best known hyperparameters as first trial
best_params = {
    "worker_num": 16,
    "thread_num": 11,
    "learning_rate": 9.941716684429581e-06,
    "batch_size": 98304,
    "gamma": 0.92,
    "net_dimension": 3584,
    "base_target_step": 395,
    "base_break_step": 70000,
    "lookback": 4,
    # ... (see best_trial_hyperparams.json for full list)
}

study.enqueue_trial(best_params)
```

---

## Expected Results

### Speedup Breakdown:
1. **Pruning**: 2.5x speedup (kill 60% of bad trials early)
2. **Parallel (n_jobs=2)**: 1.8x speedup (GPU can handle 2 trials)
3. **Fewer trials (400 vs 1000)**: 2.5x speedup
4. **Better sampler**: 1.2x speedup (faster convergence)

**Combined:** 2.5 × 1.8 ÷ 2.5 × 1.2 = **~2.7x faster** (with overlapping effects)

### New Timeline:
- **Before:** 10.2 days (1000 trials @ 14.7 min/trial)
- **After:** ~3.8 days (400 trials @ ~7 min/trial with pruning & parallel)
- **Target met:** ✅ 2-3 day range achieved

### Quality Impact:
- Pruning: Minimal (kills bad trials, keeps good ones)
- Fewer trials: Slight risk, mitigated by better sampler
- Parallel: No impact on quality
- **Expected final Sharpe:** 0.15-0.18 (similar to current best)

---

## Testing Plan

1. **Test pruning first** (single trial):
   ```bash
   python scripts/training/1_optimize_unified.py --n-trials 10 --study-name test_pruning
   ```
   - Check logs for "Trial X pruned" messages
   - Verify trials are killed early (not running full 14 min)

2. **Test parallel next**:
   ```bash
   python scripts/training/1_optimize_unified.py --n-trials 4 --study-name test_parallel
   ```
   - Watch `rocm-smi` for GPU memory usage
   - Should see ~12GB usage (2 × 6GB)
   - If OOM, reduce to n_jobs=1

3. **Full run**:
   ```bash
   python scripts/training/1_optimize_unified.py \
     --n-trials 400 --use-best-ranges \
     --study-name cappuccino_ft_16gb_optimized_fast
   ```

---

## Rollback Plan

If speedups cause issues:

1. **Pruning too aggressive?**
   - Increase `n_startup_trials` to 10
   - Increase `n_warmup_steps` to 20

2. **GPU OOM with parallel?**
   - Reduce `n_jobs` to 1
   - Still get 2.5x speedup from pruning + fewer trials

3. **Quality degraded?**
   - Increase `n_trials` back to 600
   - Still complete in ~5 days (better than 11)

---

## Monitoring

Watch the dashboard:
```bash
python paper_trader_dashboard.py
```

Check these metrics:
- **Trials/hour**: Should increase from ~4 to ~10-12
- **Pruned trials**: Should see 50-70% pruned
- **Best Sharpe**: Should stay >= 0.15
- **ETA**: Should drop to 2-3 days

---

## Files Modified

1. `scripts/training/1_optimize_unified.py`: Add pruner, sampler, n_jobs
2. `utils/function_train_test.py`: Add trial.report() calls
3. Command line: Change --n-trials and add --use-best-ranges

## Files to Reference

- `OPTIMIZE_TRAINING_PROMPT.md`: Full problem description
- `OPTIMIZE_TRAINING_QUICK.txt`: Quick reference
- `best_trial_hyperparams.json`: Best known hyperparameters
- Current training script: `scripts/training/1_optimize_unified.py`
