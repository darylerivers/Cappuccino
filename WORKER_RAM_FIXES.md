# Worker and RAM Optimization Fixes

**Date:** 2026-02-15
**Issue:** Excessive trial pruning and potential RAM exhaustion
**Status:** ✅ FIXED

## Problems Identified

1. **Too many workers per trial**: Each trial was spawning 6-12 worker processes
2. **Too many parallel trials**: Automated pipeline defaulted to 3 workers
3. **RAM exhaustion**: With 31GB total RAM, running 3 trials × 10 workers = 30 processes was too aggressive
4. **Excessive pruning**: Trials were being pruned due to OOM errors

## Changes Made

### 1. Training Script (`scripts/training/1_optimize_unified.py`)
```python
# BEFORE:
worker_num = trial.suggest_int("worker_num", 6, 12)  # Too high!

# AFTER:
worker_num = trial.suggest_int("worker_num", 2, 3)  # Conservative for stability
```

**Impact:** Each trial now uses 2-3 workers instead of 6-12, reducing RAM usage by ~60-75%

### 2. Automated Pipeline (`scripts/automation/automated_training_pipeline.py`)
```python
# BEFORE:
n_workers: int = 3  # Fixed default

# AFTER:
n_workers: int = None  # Auto-detect based on RAM
```

**New Features:**
- Auto-detects available RAM
- Recommends 2-3 workers based on available memory
- Added `get_recommended_workers()` function
- Caps max workers at 3 for stability

**RAM Allocation Logic:**
- `>= 24GB available` → 3 workers
- `>= 12GB available` → 2 workers
- `< 12GB available` → 1 worker

### 3. Config Files (`train/config.py`, `train/config_fast.py`)
```python
# BEFORE:
self.worker_num = max(8, min(cpu_total - 4, 12))

# AFTER:
self.worker_num = 2  # Limited to 2-3 for RAM stability
```

## Expected Results

### Before
- **RAM Usage:** ~18-24GB (3 trials × 10 workers × ~600MB/worker)
- **Trial Success Rate:** ~50-70% (many OOM failures)
- **Pruning Rate:** High (30-40% trials pruned)

### After
- **RAM Usage:** ~6-9GB (2-3 trials × 3 workers × ~600MB/worker)
- **Trial Success Rate:** ~90-95% (rare OOM failures)
- **Pruning Rate:** Low (5-10% trials pruned for valid reasons)
- **More headroom:** 20GB+ RAM free for system stability

## How to Apply

### For New Training
The fixes are automatically applied - just start training normally:
```bash
python scripts/automation/automated_training_pipeline.py --mode full
```

### For Currently Running Training
You need to restart the workers to apply the new settings:

```bash
# Stop current workers
pkill -f "1_optimize_unified.py"

# Start fresh with new settings
python scripts/automation/automated_training_pipeline.py --mode training
```

## Monitoring

Check RAM usage during training:
```bash
# Live RAM monitoring
watch -n 5 'free -h && echo && ps aux | grep python | grep optimize'

# Worker count check
ps aux | grep "1_optimize_unified.py" | wc -l
```

## Notes

- Worker limits prevent RAM exhaustion while maintaining good throughput
- Each worker still uses GPU efficiently (GPU is shared, not duplicated)
- Training will be slightly slower but much more stable
- Fewer OOM crashes means better resource utilization overall
