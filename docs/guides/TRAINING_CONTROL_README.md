# Training Worker Control

**Quick GPU sharing for your training workflow**

This system lets you dynamically scale training workers up/down without stopping training completely. Perfect for when you need to use your GPU for other tasks while keeping training running in the background.

---

## Quick Start

### Fast Commands (Most Common)

**When you need GPU for other tasks:**
```bash
./training_minimal.sh
```
Scales down to 2 workers, frees ~7 GB VRAM

**When GPU is free again:**
```bash
./training_full.sh
```
Scales back up to 10 workers, maximum training speed

### Interactive Menu (All Options)

```bash
./training_control.sh
```

Provides menu with presets:
- **MINIMAL** (2 workers) - Use GPU for other tasks
- **LIGHT** (4 workers) - Light background training
- **HALF** (5 workers) - Balanced mode
- **RECOMMENDED** (7 workers) - Optimal (75% VRAM)
- **FULL** (10 workers) - Maximum speed (93% VRAM)
- **CUSTOM** - Specify exact worker count
- Plus status monitoring and detailed process view

---

## Usage Scenarios

### Scenario 1: Need GPU for Gaming/ML Experiments

**Before:**
```bash
# 10 workers running, 93% VRAM
nvidia-smi  # 7.6 GB / 8.2 GB used
```

**Scale down:**
```bash
./training_minimal.sh
# Stops 8 workers, keeps 2 running
# Frees ~7 GB VRAM
```

**Your GPU is now available:**
- Training continues at ~12 trials/hour (slow but running)
- ~7 GB VRAM free for other tasks
- Can game, run other models, etc.

**When done:**
```bash
./training_full.sh
# Starts 8 new workers, back to 10 total
# Training speed restored to ~60 trials/hour
```

---

### Scenario 2: Want Balanced Mode (Keep Some Headroom)

```bash
./training_control.sh
# Choose option 4: RECOMMENDED (7 workers)
```

**Result:**
- 7 workers running
- ~6 GB VRAM (75% of GPU) - safe headroom
- ~42 trials/hour - still good speed
- Can run smaller tasks on GPU if needed

---

### Scenario 3: Fine-Tune Exact Worker Count

```bash
./training_control.sh
# Choose option 6: CUSTOM
# Enter: 3
```

**Result:**
- Exactly 3 workers running
- ~2.5 GB VRAM used
- ~18 trials/hour
- ~5 GB VRAM free

---

## How It Works

### Worker Scaling

**Each training worker uses:**
- ~850 MB VRAM
- ~1.2 GB system RAM

**Scaling down (stop workers):**
- Stops newest workers first (highest PIDs)
- Graceful shutdown with kill (SIGTERM)
- Waits 3 seconds for cleanup
- Shows new resource usage

**Scaling up (start workers):**
- Starts new optimize_unified.py processes
- Logs to `logs/worker_N_TIMESTAMP.log`
- Staggers starts by 0.5s to avoid spikes
- Waits 3 seconds for initialization

**Continuous training:**
- Optuna database is shared across all workers
- Stopping/starting workers doesn't interrupt trials
- Each worker picks up from database state
- No data loss

---

## Resource Impact

| Mode | Workers | VRAM Used | VRAM % | Trials/hr | Use Case |
|------|---------|-----------|--------|-----------|----------|
| MINIMAL | 2 | ~1.7 GB | 21% | ~12 | GPU needed for other tasks |
| LIGHT | 4 | ~3.4 GB | 41% | ~24 | Light background training |
| HALF | 5 | ~4.2 GB | 51% | ~30 | Balanced mode |
| RECOMMENDED | 7 | ~6.0 GB | 73% | ~42 | Optimal (safe headroom) |
| FULL | 10 | ~7.6 GB | 93% | ~60 | Maximum speed |

**Training speed estimates based on current ~6 trials/hour per worker**

---

## Commands Reference

### Quick Scripts

```bash
# Scale to minimal (2 workers)
./training_minimal.sh

# Scale to full (10 workers)
./training_full.sh

# Interactive menu (all presets)
./training_control.sh
```

### Manual Control

```bash
# Check current worker count
ps aux | grep optimize_unified | grep -v grep | wc -l

# Check VRAM usage
nvidia-smi

# Manually stop a worker
kill <PID>

# Manually start a worker
nohup python -u optimize_unified.py > logs/worker_new.log 2>&1 &
```

---

## Examples

### Example 1: Quick GPU Switch

```bash
# Before gaming session
./training_minimal.sh
# Training drops to 2 workers, GPU mostly free

# After gaming
./training_full.sh
# Training back to full speed
```

**Timeline:**
- 9:00 AM - Training at 10 workers (60 trials/hr)
- 10:00 AM - Scale to 2 workers (12 trials/hr)
- 2:00 PM - Gaming for 4 hours (48 trials completed)
- 6:00 PM - Scale back to 10 workers (60 trials/hr)
- **Net result:** Kept training running, only lost ~50 trials vs stopping completely

---

### Example 2: Overnight Training Optimization

```bash
# Before bed, want stable overnight run
./training_control.sh
# Choose RECOMMENDED (7 workers)
```

**Why:**
- 75% VRAM is safer for long unattended runs
- Less heat/power consumption
- Still good training speed (~42 trials/hr)
- Wake up to ~340 trials instead of 480 (vs full)
- But safer and more stable

---

### Example 3: Testing Different Training Rates

```bash
# Try half speed
./training_control.sh  # Option 3: HALF (5 workers)
# Monitor for 1 hour

# If stable, go to recommended
./training_control.sh  # Option 4: RECOMMENDED (7 workers)
# Monitor for 1 hour

# If want more speed
./training_control.sh  # Option 5: FULL (10 workers)
```

---

## Safety Features

### Confirmation Prompts
All scaling operations require confirmation:
```bash
Will stop 8 workers (keeping 2 running)

Workers to stop:
  PID: 1234567, RAM: 1234 MB
  PID: 1234568, RAM: 1235 MB
  ...

Continue? (y/n):
```

### Status Display
After each operation:
```bash
New status:
  VRAM: 1700 MiB / 8192 MiB (21%)
  Workers: 2

✓ Training scaled to MINIMAL mode
  Training speed: ~12 trials/hour
```

### Detailed Process View
```bash
./training_control.sh
# Option 8: SHOW DETAILED STATUS

Training Workers:
  PID: 1105329  CPU: 45.2% RAM: 1234 MB  Started: 08:23
  PID: 1105587  CPU: 43.8% RAM: 1198 MB  Started: 08:23

Paper Traders:
  PID: 1234567  CPU:  2.1% RAM:  678 MB
  PID: 1234568  CPU:  1.9% RAM:  702 MB

GPU Details:
  VRAM Used: 1700 MiB
  VRAM Free: 6492 MiB
  GPU Util: 12%
  Temp: 52C
```

---

## Troubleshooting

### Workers Not Starting

**Problem:**
```bash
./training_full.sh
Error: optimize_unified.py not found
```

**Solution:**
```bash
# Run scripts from project directory
cd /opt/user-data/experiment/cappuccino
./training_full.sh
```

---

### VRAM Still High After Scaling Down

**Problem:**
- Scaled to 2 workers but VRAM still at 90%

**Cause:**
- CUDA cache not cleared
- Paper traders also use some VRAM (if using --gpu flag)

**Solution:**
```bash
# Check what's using GPU
nvidia-smi

# If paper traders using GPU, they don't use much (<100 MB each)
# The cache should clear automatically within a few minutes
```

---

### Training Speed Lower Than Expected

**Problem:**
- 7 workers but only getting 30 trials/hr (expected ~42)

**Possible causes:**
1. Trials are taking longer (harder hyperparameter space)
2. Database lock contention (many workers competing)
3. System RAM/CPU bottleneck

**Check:**
```bash
# Watch trial completion rate
tail -f logs/worker_*.log | grep "Trial.*finished"

# Check system resources
htop
```

---

### Too Many Workers After Script Exit

**Problem:**
- Accidentally started too many workers

**Solution:**
```bash
./training_control.sh
# Option 7: STOP ALL WORKERS
# Then start fresh with desired count
```

---

## Best Practices

### 1. Use Presets for Common Scenarios

**Don't:**
```bash
# Manually killing workers one by one
kill 1234567
kill 1234568
...
```

**Do:**
```bash
# Use preset
./training_minimal.sh
```

---

### 2. Check Status After Scaling

```bash
# After any scaling operation
nvidia-smi
ps aux | grep optimize_unified | grep -v grep | wc -l
```

---

### 3. Use RECOMMENDED (7 workers) for Unattended Runs

**Why:**
- 75% VRAM is safe margin
- Lower heat/power
- Stable overnight
- Still good speed

**When to use FULL (10 workers):**
- Actively monitoring
- Want maximum speed
- Short training bursts
- System has good cooling

---

### 4. Scale Gradually When Testing

```bash
# Start conservative
./training_control.sh  # HALF (5 workers)

# Increase if stable
./training_control.sh  # RECOMMENDED (7 workers)

# Max out if desired
./training_control.sh  # FULL (10 workers)
```

---

## Integration with Other Tools

### With Memory Optimization Script

```bash
# If you ran optimize_memory.sh earlier and stopped workers,
# you can use training_control.sh to precisely set worker count

./optimize_memory.sh  # Old way (stopped 2-3 workers)
./training_control.sh  # New way (precise control)
```

**Recommendation:** Use `training_control.sh` going forward - it's more flexible.

---

### With Dashboard

```bash
# Terminal 1: Dashboard
./start_dashboard.sh

# Terminal 2: Scale training
./training_minimal.sh

# Dashboard will show training speed decrease
# But paper traders continue unaffected
```

---

### With Alert System

```bash
# Alert system monitors paper traders (not training workers)
# Scaling training won't trigger any alerts
# Paper traders continue running normally
```

---

## Files Created

```
training_control.sh      # Interactive menu with all presets
training_minimal.sh      # Quick: scale to 2 workers
training_full.sh         # Quick: scale to 10 workers
TRAINING_CONTROL_README.md  # This file
```

---

## Summary

**Quick commands:**
- `./training_minimal.sh` - Free GPU (2 workers)
- `./training_full.sh` - Max speed (10 workers)
- `./training_control.sh` - Interactive menu

**Key benefits:**
- ✅ Share GPU without stopping training
- ✅ Easy presets for common scenarios
- ✅ Safe confirmation prompts
- ✅ No data loss (shared Optuna database)
- ✅ Instant resource feedback

**Typical workflow:**
1. Need GPU: `./training_minimal.sh`
2. Do your GPU task (game, train other model, etc.)
3. Done: `./training_full.sh`
4. Training continues uninterrupted

---

**Your training, your GPU, your schedule!** 🚀
