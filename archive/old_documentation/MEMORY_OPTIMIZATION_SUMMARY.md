# Memory Optimization - Quick Summary

**Date:** January 17, 2026
**Current State:** 93% VRAM, ~12 GB RAM
**Recommendation:** Stop 2-3 training workers

---

## Current Usage

```
GPU VRAM: 7.6 GB / 8.2 GB (93%) ⚠️  TIGHT
System RAM: ~12 GB

Breakdown:
  10 training workers = 9.6 GB RAM, 7.6 GB VRAM
   2 paper traders   = 1.4 GB RAM, 0 GB VRAM
   Monitoring tools  = 0.8 GB RAM, 0 GB VRAM
```

**Problem:** GPU VRAM at 93% leaves only 600 MB buffer - risky!

---

## Recommended: Stop 2 Workers ⭐

**Quick action:**
```bash
./optimize_memory.sh
# Choose option 1
```

**What it does:**
- Stops 2 training workers (oldest/highest RAM)
- Frees ~2 GB VRAM, ~2.5 GB RAM
- Brings VRAM to ~75% (much safer)

**Impact:**
- Training speed: 59 → 47 trials/hour (-20%)
- Time to 2,000 trials: ~14 hours (still reasonable)
- Risk: LOW (you already have 1,351 good trials)

**After optimization:**
```
GPU VRAM: 6.0 GB / 8.2 GB (73%) ✅ SAFE
System RAM: ~9.5 GB ✅ PLENTY OF HEADROOM

  8 training workers = 7.6 GB RAM, 6.0 GB VRAM
  2 paper traders   = 1.4 GB RAM, 0 GB VRAM
  Monitoring tools  = 0.5 GB RAM, 0 GB VRAM
```

---

## Other Options (Optional)

### If You Want More Savings

**Stop 3 workers** (more aggressive)
```bash
./optimize_memory.sh
# Choose option 2
```
- VRAM: 5.0 GB / 8.2 GB (61%) ✅
- Training: 40 trials/hour (-33%)
- Time to 2,000: ~16 hours

**Create lean ensemble** (5 models instead of 10)
```bash
./optimize_memory.sh
# Choose option 3
```
- Saves: ~1 GB RAM (paper traders only)
- Requires testing for 24h
- Medium risk (less model diversity)

---

## Why This Helps

### Problem: GPU VRAM Bottleneck
Your GPU VRAM (8 GB) is the constraint, not system RAM (32 GB total).

**At 93% VRAM:**
- Only 600 MB free
- OOM errors if any spike
- Can't add more workers
- Limits experimentation

**At 75% VRAM:**
- 2 GB free
- Safe headroom
- Can add worker back if needed
- Stable operation

### Training Progress
You're at 1,351 trials:
- Already have excellent models (Sharpe 0.0140)
- Target was 2,000-5,000 trials
- Can afford slower training now
- Quality > speed at this stage

---

## Step-by-Step Guide

### 1. Check Current State
```bash
nvidia-smi  # Check VRAM
./optimize_memory.sh  # Choose option 4 for details
```

### 2. Run Optimization
```bash
./optimize_memory.sh
# Choose option 1 (recommended)
# Confirm when prompted
```

### 3. Verify Results
```bash
nvidia-smi  # Should show ~6 GB used (75%)
ps aux | grep optimize_unified | grep -v grep | wc -l  # Should show 8 workers
```

### 4. Monitor Training Speed
```bash
# Check trial rate after 30 minutes
# Should be ~45-50 trials/hour (down from 59, but still good)
```

---

## Trade-offs Explained

| Workers | VRAM | Trials/hr | Time to 2k | Notes |
|---------|------|-----------|------------|-------|
| 10 (current) | 93% | 59 | 11h | Too tight, risky |
| 8 (recommended) | 75% | 47 | 14h | ✅ Safe, still fast |
| 7 | 68% | 40 | 16h | Very safe, slower |
| 6 | 61% | 35 | 19h | Conservative |

**Sweet spot:** 8 workers (75% VRAM)

---

## What NOT to Change

**Keep these:**
- Paper traders (need both for comparison)
- Alert system (critical for safety)
- Dashboard (useful for monitoring)
- Training workers below 6 (too slow)

**Why:**
- Total memory cost: ~2 GB
- Critical for validation and safety
- Stopping won't meaningfully free VRAM
- VRAM bottleneck is training workers

---

## Files Created

```
✅ MEMORY_OPTIMIZATION_GUIDE.md       # Detailed guide
✅ MEMORY_OPTIMIZATION_SUMMARY.md     # This file
✅ optimize_memory.sh                 # Interactive helper script
```

---

## Quick Commands

**Run optimization helper:**
```bash
./optimize_memory.sh
```

**Manual worker stop:**
```bash
# List workers
ps aux | grep optimize_unified | grep -v grep

# Stop highest PID (newest worker)
kill <PID>
```

**Check VRAM:**
```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
```

**Monitor resource usage:**
```bash
watch -n 5 'nvidia-smi && echo "---" && ps aux | grep -E "optimize|paper_trader" | grep -v grep | head -12'
```

---

## Expected Results

### Before
```
VRAM: 7.6 GB / 8.2 GB (93%) ⚠️
Workers: 10
Speed: 59 trials/hour
Risk: HIGH (very little buffer)
```

### After (Stop 2 Workers)
```
VRAM: 6.0 GB / 8.2 GB (73%) ✅
Workers: 8
Speed: 47 trials/hour
Risk: LOW (plenty of buffer)
```

**Net result:**
- ✅ Safe VRAM usage
- ✅ Still good training speed
- ✅ Reach 2,000 trials in ~14 hours
- ✅ Reach 5,000 trials in ~4 days (if needed)

---

## Bottom Line

**Do this now:**
```bash
./optimize_memory.sh  # Choose option 1
```

**Benefits:**
- VRAM drops from 93% → 75% (much safer)
- Training still reasonably fast (47 trials/hr)
- Low risk (already have good models)
- Takes 1 minute

**Don't overthink it:**
- You're at 93% VRAM (dangerous)
- Stopping 2 workers = easy win
- Can always restart them if needed
- No downside at 1,351 trials

---

**TL;DR:** Stop 2 training workers → VRAM drops to 75% → Everyone happy! 🎉
