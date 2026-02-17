# Fix GPU Idle Pattern - Keep GPU at 100%

## Problem Identified
GPU cycles between:
- **100% utilization** (24 seconds) - Neural network training
- **50% utilization** (rest of time) - Environment rollout/data collection

This is typical PPO training:
1. Collect experience from environment (CPU-bound, ~50% GPU)
2. Train neural network on that data (GPU-bound, 100% GPU)
3. Repeat

## Root Cause
During environment rollout collection:
- Running parallel environments (NumPy operations on CPU)
- GPU only used for inference (policy forward pass)
- Most GPU cores idle waiting for next training batch

## Solutions (Ranked by Impact)

### 1. Increase Training Intensity (EASIEST - HIGH IMPACT)
**Train more on each batch of collected data**

Add to your hyperparameters:
```python
# In 1_optimize_unified.py, in the erl_params dict:
"repeat_times": 16,  # Default is often 4-8, increase to 16-32
```

Effect:
- Longer training phase (more time at 100% GPU)
- Shorter rollout phase (less time at 50% GPU)
- Better sample efficiency

**Expected: 70-80% average GPU utilization**

### 2. Reduce Buffer Size, Increase Frequency
**Collect smaller batches more frequently**

```python
# Current:
base_target_step = trial.suggest_int("base_target_step", 98304, 196608, step=32768)

# Optimized for GPU utilization:
base_target_step = trial.suggest_int("base_target_step", 32768, 65536, step=16384)
# And increase repeat_times to compensate
```

Effect:
- Less time collecting (faster rollouts)
- More frequent training bursts
- Smoother GPU utilization curve

**Expected: 75-85% average GPU utilization**

### 3. Vectorized GPU Environments (COMPLEX - HIGHEST IMPACT)
**Run environments on GPU instead of CPU**

Use libraries like:
- `brax` (JAX-based physics on GPU)
- `isaacgym` (NVIDIA GPU environments)
- Custom vectorized envs with PyTorch

This is a major refactor but can achieve:
**Expected: 95-100% sustained GPU utilization**

### 4. Asynchronous Data Collection (MODERATE COMPLEXITY)
**Overlap rollout and training**

Implement double buffering:
```python
# Collect next batch while training on current batch
import threading

def collect_rollout_async(buffer):
    # Runs in background thread
    pass

# Training loop:
next_buffer = start_async_collection()
while training:
    train_on_current_buffer()  # GPU busy
    current_buffer = next_buffer
    next_buffer = start_async_collection()  # Overlap
```

**Expected: 85-95% average GPU utilization**

### 5. Increase Worker Count (EASY - MODERATE IMPACT)
**Faster rollout collection with more parallel environments**

```python
# Current: 16-24 workers
worker_num = trial.suggest_int("worker_num", 24, 40)  # More workers
```

Effect:
- Faster environment rollouts
- Shorter idle periods
- Requires more RAM (you have 19GB free, plenty of room)

**Expected: 60-70% average GPU utilization**

## Recommended Quick Fix

**Add to `1_optimize_unified.py`:**

```python
# In sample_hyperparams function, add to erl_params dict:
erl_params = {
    # ... existing params ...
    "repeat_times": 16,  # Train more on each batch
    "if_per_or_gae": True,  # Use GAE for better training
}

# Also reduce target_step to collect faster:
base_target_step = trial.suggest_int("base_target_step", 32768, 98304, step=16384)

# And increase workers:
worker_num = trial.suggest_int("worker_num", 20, 32)
```

## Expected Results

### Before:
```
Timeline: [100%][100%][50%][50%][50%][50%] → 62.5% avg
Power:    [208W][208W][125W][125W][125W][125W] → 155W avg
```

### After (with repeat_times=16):
```
Timeline: [100%][100%][100%][100%][50%][50%] → 83% avg
Power:    [208W][208W][208W][208W][125W][125W] → 180W avg
```

### After (with all optimizations):
```
Timeline: [100%][100%][100%][100%][100%][80%] → 95% avg
Power:    [208W][208W][208W][208W][208W][170W] → 200W avg
```

## Implementation Priority

1. **Start here:** Add `repeat_times: 16` (5 min change, big impact)
2. **Next:** Reduce target_step, increase workers (10 min)
3. **Advanced:** Async collection (1-2 hours)
4. **Expert:** GPU environments (major refactor)

The GPU idle time is not a GPU problem - it's a training algorithm structure problem. The GPU is doing its job; we need to keep feeding it more work.
