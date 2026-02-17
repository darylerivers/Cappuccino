# System Crash Analysis - Feb 11, 2026

## What Happened

Your desktop **froze completely** requiring a hard reboot due to **memory exhaustion**.

## Root Cause

**Two memory-intensive processes running simultaneously:**

### Before Crash:
- **Training:** 19GB RAM used
- **Swap:** 4GB/4GB (100% full)
- **GLM Model Loading:** Tried to allocate 20GB more
- **Total Needed:** ~39GB
- **Available:** 31GB RAM + 4GB swap = 35GB
- **Shortfall:** 4GB ❌

### Timeline:
1. `test_run_huge` training started (batch=131K, net_dim=2560)
2. Training consumed 19GB RAM
3. Swap filled to 100% (4GB)
4. GLM-4.7-Flash model started downloading/loading
5. Model tried to materialize 20GB of weights
6. **System ran out of memory**
7. Linux kernel froze trying to allocate memory
8. Desktop became unresponsive
9. Hard reboot required

## Evidence

### Memory State Before Crash:
```
RAM:  19GB used / 31GB total (61% full)
Swap: 4GB used / 4GB total (100% FULL)
Available: ~11GB
```

### Processes:
- Python (training): 2.6GB RSS + swapped memory
- GLM model loader: Tried to allocate ~20GB
- System couldn't allocate → freeze

### After Reboot (Current):
```
RAM:  4.1GB used / 31GB total (healthy)
Swap: 0GB used / 4GB total (cleared)
Available: 27GB
```

## Why Desktop Froze

When Linux runs out of both RAM and swap:
1. **OOM Killer** tries to kill processes
2. If it can't kill fast enough → system freezes
3. Desktop becomes unresponsive
4. Hard reboot is only option

This happened because **overcommit_memory=0** (heuristic) allowed processes to request more memory than available, then couldn't deliver it.

## Solutions

### Immediate Fix: Prevent Simultaneous Heavy Loads

**DON'T run these together:**
- ❌ Training (19GB) + GLM loading (20GB) = CRASH
- ✅ Training alone = OK
- ✅ GLM alone = OK (barely)

### Option 1: Increase Swap (Recommended)

```bash
# Increase swap from 4GB to 16GB
sudo swapoff /dev/zram0
sudo zramctl --reset /dev/zram0
sudo zramctl --size 16G /dev/zram0
sudo mkswap /dev/zram0
sudo swapon /dev/zram0
```

**Result:** 31GB RAM + 16GB swap = 47GB total (safe)

### Option 2: Reduce Training Memory Usage

```bash
# Use smaller batch sizes
# Edit scripts/training/1_optimize_unified.py
# Change: batch_size = [65536, 98304]  # Instead of 131K
# Change: net_dimension = (1024, 2048)  # Instead of 4096
```

**Result:** Training uses ~10GB instead of 19GB

### Option 3: Use Ollama for LLM (Best for GLM)

```bash
# Ollama manages memory efficiently
ollama pull glm4
# Uses ~5GB instead of 20GB
```

### Option 4: Set Memory Limits

```bash
# Limit training memory
ulimit -v 20000000  # 20GB limit
# Then run training
```

### Option 5: Enable OOM Killer Earlier

```bash
# Make OOM killer more aggressive
echo 1 | sudo tee /proc/sys/vm/overcommit_memory
# Prevents system freeze by killing processes sooner
```

## Prevention Checklist

Before running memory-intensive tasks:

```bash
# 1. Check available memory
free -h

# 2. Check what's running
ps aux | sort -k4 -rn | head -n 10

# 3. Ensure enough headroom
# Rule: Total needed < (RAM + Swap - 5GB safety margin)

# 4. Monitor during run
watch -n 1 'free -h; echo ""; ps aux | sort -k4 -rn | head -n 5'
```

## Recommended Action Plan

1. **Increase swap to 16GB** (prevents crashes)
2. **Use Ollama for GLM** (reduces memory to 5GB)
3. **Monitor memory** before starting tasks
4. **Don't run training + GLM loading simultaneously**

## Safe Memory Budget

With 31GB RAM + 16GB swap = 47GB total:

- Training (huge models): 15-20GB ✅
- GLM via Ollama: 5GB ✅
- System + Desktop: 3GB ✅
- Safety margin: 5GB ✅
- **Total: 33GB used / 47GB available** ✅

## Current Status

- ✅ System rebooted successfully
- ✅ Memory cleared (27GB available)
- ✅ No processes running
- ⚠️ Same issue will happen again without fixes

## Next Steps

1. Increase swap OR
2. Use Ollama for LLM OR
3. Reduce training batch sizes

**Pick one solution above to prevent future crashes.**
