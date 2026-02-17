# Training Speed Fix - Break Steps Correction

**Date:** 2026-02-15
**Issue:** Trials completing in 1 minute - insufficient training time
**Status:** ✅ FIXED

## Problem

Trials were completing in ~1 minute with only **6,000 training steps**. This is far too short for a DRL agent to learn meaningful trading strategies.

### Root Cause

The "AGGRESSIVE SPEED" optimization mode was set too aggressively:

```python
# BEFORE (TOO FAST):
base_break_step = 5,000 - 12,000 steps  # Only ~1 minute per trial
base_target_step = 512 - 2,048 steps
```

**Result:** Agents had no time to learn, all trials converged to similar poor performance (~0.17 Sharpe)

## Solution

Increased training steps to allow meaningful learning while maintaining reasonable speed:

```python
# AFTER (BALANCED):
base_break_step = 30,000 - 60,000 steps  # ~5-10 minutes per trial
base_target_step = 256 - 1,024 steps
```

### Comparison Chart

| Mode | Break Steps | Trial Duration | Learning Quality |
|------|-------------|----------------|------------------|
| **Too Fast (OLD)** | 5k - 12k | ~1 min | ⚠️ Poor - No learning |
| **Balanced (NEW)** | 30k - 60k | ~5-10 min | ✅ Good - Learns strategies |
| **Exploitation** | 110k - 140k | ~20-30 min | 🏆 Best - Deep optimization |

## Expected Changes

### Before Fix
- Trial duration: **1 minute**
- Training steps: **6,000**
- Agent learning: **Minimal** (random behavior)
- Performance: **Poor convergence** (all ~0.17 Sharpe)
- Throughput: **60 trials/hour** (fast but useless)

### After Fix
- Trial duration: **5-10 minutes**
- Training steps: **30,000 - 60,000**
- Agent learning: **Meaningful** (learns patterns)
- Performance: **Better diversity** (0.10 - 0.30+ Sharpe expected)
- Throughput: **6-12 trials/hour** (slower but effective)

## When to Use Different Modes

### 1. Quick Exploration (Current - NEW Settings)
**Use when:** Initial hyperparameter search, testing new features
```bash
# Default mode (no flags)
python scripts/training/1_optimize_unified.py --n-trials 100
```
- Break steps: **30k - 60k**
- Trial time: **~5-10 min**
- Purpose: Find promising hyperparameter regions quickly

### 2. Deep Exploitation (Best for Final Models)
**Use when:** Refining best hyperparameters, production models
```bash
# Add --use-best-ranges flag
python scripts/training/1_optimize_unified.py --n-trials 50 --use-best-ranges
```
- Break steps: **110k - 140k**
- Trial time: **~20-30 min**
- Purpose: Extract maximum performance from known-good regions

## Apply the Fix

### For Currently Running Training

**Option 1: Let current trials finish, new trials use fix automatically**
- Current trials will still be fast (~1 min)
- New trials will be properly trained (~5-10 min)
- No interruption needed

**Option 2: Restart to apply immediately (Recommended)**
```bash
# Stop current workers
pkill -f "cappuccino_auto_20260215_2158"

# Start fresh with corrected settings
python scripts/automation/automated_training_pipeline.py --mode training
```

### For New Training

The fix is already applied - just start training normally:
```bash
python scripts/automation/automated_training_pipeline.py --mode full
```

## Monitoring

Check if trials are taking appropriate time:
```bash
# Watch trial durations in real-time
python monitor_training_dashboard.py \
  --study cappuccino_auto_20260215_2158 \
  --watch
```

**Good sign:** Trials taking 5-15 minutes each
**Bad sign:** Trials completing in < 2 minutes (still too fast)

## Performance Expectations

With proper training time, you should see:
- **Sharpe diversity:** 0.05 to 0.30+ (wide range as agents explore)
- **Learning curves:** Clear improvement during training
- **Strategy diversity:** Different agents try different approaches
- **Top performers:** Sharpe > 0.25 for well-tuned agents

## Notes

- The 1-minute trials were essentially "random agents" with no learning
- All getting ~0.17 Sharpe means they weren't exploring properly
- With 30k-60k steps, agents have time to discover profitable patterns
- This is still 5x faster than exploitation mode (110k-140k steps)
- Good balance of speed and learning quality
