# Memory Optimization Guide

**Date:** January 17, 2026
**Current Usage:** 7.6 GB VRAM (93%), ~13 GB RAM
**Goal:** Reduce usage while maintaining performance

---

## Current Resource Usage

### GPU VRAM (7.6 GB / 8.2 GB - 93%)
```
9 training workers @ ~850 MB each = 7.6 GB
```

### System RAM (~13 GB total)
```
9 training workers @ 1.2-1.3 GB each = 11.0 GB
2 paper traders @ 650-700 MB each   =  1.3 GB
Monitoring tools                     =  0.6 GB
                                    -------
                                     12.9 GB
```

**Bottleneck:** GPU VRAM at 93% (leaves only 600 MB buffer)

---

## Optimization Strategies

### 1. Reduce Training Workers ⭐ HIGH IMPACT

**Current:** 9 workers
**Recommended:** 6-7 workers

**Savings:**
- VRAM: 2-3 GB (brings down to 70-80%)
- RAM: 2.5-4 GB

**Trade-off:**
- Training speed: -22% to -33%
- Current: ~59 trials/hour
- After: ~40-47 trials/hour

**Implementation:**
```bash
# Stop 2-3 workers
ps aux | grep optimize_unified | grep -v grep
kill <PID_of_worker_7>
kill <PID_of_worker_8>
kill <PID_of_worker_9>  # Optional

# Check VRAM
nvidia-smi
```

**When to do this:**
- You're at 1,351 trials (already have good models)
- Target was 2,000-5,000 trials
- Can reach 2,000 in ~14 hours with 7 workers
- Can afford slower progress

---

### 2. Reduce Ensemble Size ⭐ MEDIUM IMPACT

**Current:** Ensemble trader loads 10 models
**Recommended:** 5 models

**Savings:**
- RAM: ~300-500 MB per trader
- Total: ~600-1000 MB

**Trade-off:**
- Less model diversity in voting
- May slightly reduce robustness
- Performance impact: Minimal (5 models still good)

**Implementation:**

**Option A: Change ensemble size globally**
```python
# Edit: train_results/ensemble/ensemble_params.json
{
  "top_n": 5,  # Change from 10 to 5
  ...
}
```

**Option B: Create separate lean ensemble**
```bash
cd train_results
cp -r ensemble ensemble_lean
# Edit ensemble_lean/ensemble_params.json - set top_n: 5

# Restart ensemble trader with lean version
kill <ensemble_trader_PID>
nohup python -u paper_trader_alpaca_polling.py \
    --model-dir train_results/ensemble_lean \
    ... # same args
```

**When to do this:**
- After confirming top-5 models are stable
- If paper trading performance is good with current ensemble
- Want to reduce memory without touching training

---

### 3. Reduce Historical Data Buffer 🔧 SMALL IMPACT

**Current:** 120 hours (5 days) of historical data
**Recommended:** 72 hours (3 days)

**Savings:**
- RAM: ~100-150 MB per trader
- Total: ~200-300 MB

**Trade-off:**
- Lookback window shorter
- Model gets less historical context
- Minimal impact if lookback < 72 hours

**Implementation:**
```bash
# Check current lookback in model
python -c "
import pickle
with open('train_results/ensemble/best_trial', 'rb') as f:
    trial = pickle.load(f)
print('Lookback:', trial.params.get('lookback', 'Unknown'))
"

# If lookback < 72, safe to reduce history
# Restart traders with --history-hours 72
```

**When to do this:**
- Model's actual lookback is < 72 hours
- Want minor memory savings
- Low risk optimization

---

### 4. PyTorch Memory Optimizations 🔧 SMALL IMPACT

**Strategies:**

**A. Use Mixed Precision (FP16)**
```python
# In paper trader or training script
torch.set_default_dtype(torch.float16)
# Savings: ~40% memory for model weights
```

**B. Clear CUDA Cache Periodically**
```python
# Add to paper trader main loop
import torch
torch.cuda.empty_cache()  # After each episode
```

**C. Gradient Checkpointing (Training Only)**
```python
# In model definition
torch.utils.checkpoint.checkpoint(model, inputs)
# Trades compute for memory
```

**Savings:**
- VRAM: 200-500 MB
- RAM: 100-300 MB

**Trade-off:**
- FP16: Slightly less precision (usually fine)
- Cache clearing: Small performance hit
- Checkpointing: Slower training

**Implementation:**
Complex - requires code changes in training scripts.

**When to do this:**
- After simpler optimizations
- If still need more memory
- Have time for testing

---

### 5. Reduce Paper Trader Polling 🔧 NEGLIGIBLE IMPACT

**Current:** Polls every 60 seconds
**Recommended:** Polls every 120 seconds

**Savings:**
- CPU: ~50% reduction in API calls
- RAM: Negligible
- Network: 50% less API traffic

**Trade-off:**
- Slower reaction to new hourly bars
- Minimal impact (hourly trading anyway)

**Implementation:**
```bash
# Restart traders with longer interval
--poll-interval 120  # Instead of 60
```

**When to do this:**
- Want to reduce API usage
- CPU usage is high
- Minimal memory benefit

---

## Recommended Optimization Plan

### Phase 1: Immediate (Low Risk)

**Stop 2 training workers**
```bash
# Reduces VRAM from 93% to ~75%
kill <worker_8_PID>
kill <worker_9_PID>

# Verify
nvidia-smi
```

**Savings:**
- VRAM: -2 GB (93% → 75%)
- RAM: -2.5 GB
- Training speed: -22% (acceptable with 1,351 trials)

**Risk:** Low
**Time:** 1 minute

---

### Phase 2: After Testing (Medium Risk)

**Reduce ensemble to 5 models**
```bash
# Create lean ensemble
cd train_results
cp -r ensemble ensemble_lean
# Edit ensemble_lean/ensemble_params.json: "top_n": 5

# Test with single trader first
# Monitor performance for 24h
# If good, apply to both traders
```

**Savings:**
- RAM: -600 MB per trader
- Total: -1.2 GB

**Risk:** Medium (may reduce performance)
**Time:** 1 day testing

---

### Phase 3: If Still Needed (Higher Risk)

**Reduce historical data to 72 hours**
```bash
# Restart with less history
--history-hours 72
```

**Savings:**
- RAM: -300 MB

**Risk:** Low-Medium
**Time:** Restart traders

---

## Memory Budget After Optimizations

### Phase 1 Only (Stop 2 Workers)
```
GPU VRAM:
  7 workers @ 850 MB = 6.0 GB / 8.2 GB (73%) ✅

System RAM:
  7 workers @ 1.3 GB  = 9.1 GB
  2 paper traders     = 1.3 GB
  Monitoring          = 0.6 GB
                      -------
  Total              = 11.0 GB ✅
```

### Phase 1 + 2 (Workers + Lean Ensemble)
```
GPU VRAM:
  7 workers @ 850 MB = 6.0 GB / 8.2 GB (73%) ✅

System RAM:
  7 workers @ 1.3 GB  = 9.1 GB
  2 lean traders      = 1.0 GB  (-300 MB each)
  Monitoring          = 0.6 GB
                      -------
  Total              = 10.7 GB ✅
```

### All Phases
```
GPU VRAM:
  7 workers @ 850 MB = 6.0 GB / 8.2 GB (73%) ✅

System RAM:
  7 workers @ 1.3 GB  = 9.1 GB
  2 lean traders      = 0.8 GB  (-150 MB each from less history)
  Monitoring          = 0.6 GB
                      -------
  Total              = 10.5 GB ✅
```

**Net savings:** 2.4 GB RAM, 1.6 GB VRAM

---

## Trade-offs Summary

| Optimization | VRAM Saved | RAM Saved | Risk | Impact on Performance |
|-------------|-----------|-----------|------|---------------------|
| Stop 2 workers | 2 GB | 2.5 GB | Low | -22% training speed |
| Lean ensemble (5 models) | 0 GB | 1.2 GB | Medium | Slightly less robust |
| Less history (72h) | 0 GB | 300 MB | Low | Minimal if lookback < 72h |
| PyTorch FP16 | 500 MB | 300 MB | Low | Slightly less precision |
| Longer polling | 0 GB | 0 GB | None | Negligible |

---

## What NOT to Optimize

### Keep These As-Is

**1. Number of Paper Traders (2)**
- Need both for comparison
- Memory cost is low (1.3 GB total)
- Critical for validation

**2. Alert System Check Interval (5 min)**
- Already efficient (300s)
- Memory: 120 MB (negligible)
- Need frequent checks for safety

**3. Training Workers Below 6**
- Already have good models at 1,351 trials
- But going < 6 workers too slow
- Need balance of speed vs resources

**4. Model Complexity**
- Don't reduce network size
- Don't change architecture
- Models are already optimized

---

## Monitoring After Optimization

### Check Resource Usage
```bash
# GPU VRAM
nvidia-smi

# System RAM
ps aux | grep python | grep -v grep | \
  awk '{sum+=$6} END {print "Total RAM:", int(sum/1024), "MB"}'

# Per-process
ps aux | grep -E "optimize|paper_trader" | grep -v grep
```

### Expected After Phase 1
- VRAM: 5.5-6.5 GB (70-80%)
- RAM: 10-11 GB
- Training: ~45-50 trials/hour
- 2000 trials in: ~15-18 hours

### Alert if:
- VRAM > 85% (check for leaks)
- RAM per worker > 1.5 GB (memory leak)
- Training speed < 35 trials/hour (too few workers)

---

## Quick Commands

### Stop Extra Workers
```bash
# List all workers
ps aux | grep optimize_unified | grep -v grep | \
  awk '{print "PID:", $2, "RAM:", int($6/1024), "MB"}'

# Stop highest RAM users (usually newest)
kill <PID>
```

### Check Current Usage
```bash
# Quick status
echo "=== GPU VRAM ===" && nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo "=== Training Workers ===" && ps aux | grep optimize_unified | grep -v grep | wc -l
echo "=== Paper Traders ===" && ps aux | grep paper_trader | grep -v grep | wc -l
```

### Restart Trader with Optimizations
```bash
# Stop current
kill <paper_trader_PID>

# Start with optimizations
nohup python -u paper_trader_alpaca_polling.py \
    --model-dir train_results/ensemble_lean \  # Lean ensemble
    --history-hours 72 \                       # Less history
    --poll-interval 120 \                      # Less polling
    ... # other args
    > logs/trader_optimized.log 2>&1 &
```

---

## Recommendations for Your Situation

Based on your current state:
- ✅ 1,351 trials completed (good progress)
- ✅ Paper traders working well
- ⚠️ VRAM at 93% (tight)
- ⚠️ RAM at ~13 GB (moderate)

### Immediate Action (Recommended)
**Stop 2 training workers**
- Brings VRAM to ~75% (comfortable)
- Frees 2.5 GB RAM
- Still reaches 2,000 trials in ~15 hours
- **Low risk, high reward**

```bash
# Do this now
ps aux | grep optimize_unified | grep -v grep | tail -2
# Kill the last 2 PIDs
kill <PID1> <PID2>
```

### Optional Follow-up
If still want more optimization after 24h:
- Test lean ensemble (5 models) on single trader
- Monitor performance vs full ensemble
- If comparable, apply to both traders

---

## Files for Reference

**Configuration Files:**
- Training workers: Check `logs/worker_*.log`
- Ensemble config: `train_results/ensemble/ensemble_params.json`
- Model params: `train_results/ensemble/best_trial`

**Memory Monitoring:**
```bash
# Create monitoring script
watch -n 5 'nvidia-smi && echo "---" && ps aux | grep -E "optimize|paper_trader" | grep -v grep | awk "{print \$2, \$3, \$4, \$6, \$11}"'
```

---

**Summary:**
- **Best bang for buck:** Stop 2 workers (-2 GB VRAM, -2.5 GB RAM)
- **Safe and easy:** Takes 1 minute, low risk
- **Still effective:** 7 workers = 45-50 trials/hour
- **Next level:** Lean ensemble if want more savings

**Start with Phase 1, evaluate, then decide on Phase 2.**
