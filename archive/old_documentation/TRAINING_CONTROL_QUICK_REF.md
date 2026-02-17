# Training Control - Quick Reference

**Need GPU for other tasks? Keep training running!**

---

## Quick Commands

```bash
# Scale to MINIMAL (2 workers, frees ~7 GB VRAM)
./training_minimal.sh

# Scale to FULL (10 workers, max speed)
./training_full.sh

# Interactive menu (all options)
./training_control.sh
```

---

## What Each Mode Does

| Command | Workers | VRAM | Speed | Use When |
|---------|---------|------|-------|----------|
| `training_minimal.sh` | 2 | 21% | 12/hr | Need GPU for gaming/other tasks |
| Option 2: LIGHT | 4 | 41% | 24/hr | Light background training |
| Option 3: HALF | 5 | 51% | 30/hr | Balanced mode |
| **Option 4: RECOMMENDED** | **7** | **73%** | **42/hr** | **Overnight/unattended (safe)** |
| `training_full.sh` | 10 | 93% | 60/hr | Maximum speed |

---

## Typical Workflow

```bash
# 1. Before using GPU for other tasks
./training_minimal.sh
   # Training drops to 2 workers
   # ~7 GB VRAM freed
   # Training continues at 12 trials/hr

# 2. Do your GPU task
#    (gaming, ML experiments, video encoding, etc.)

# 3. When done
./training_full.sh
   # Training back to 10 workers
   # Speed restored to 60 trials/hr
```

---

## Status Checks

```bash
# Quick VRAM check
nvidia-smi

# Worker count
ps aux | grep optimize_unified | grep -v grep | wc -l

# Detailed status (from interactive menu)
./training_control.sh
# Choose option 8
```

---

## Examples

### Gaming Session
```bash
# Before gaming (5 PM)
./training_minimal.sh
# Game for 4 hours
# 48 trials completed in background
# After gaming (9 PM)
./training_full.sh
```

### Overnight Training
```bash
# Before bed
./training_control.sh
# Choose option 4: RECOMMENDED (7 workers)
# Safer for unattended operation (75% VRAM)
# Wake up to ~340 trials
```

### Quick ML Experiment
```bash
# Need GPU for 1 hour
./training_minimal.sh
# Run your experiment
# Back to training
./training_full.sh
```

---

## Safety Notes

- ✅ All operations require confirmation
- ✅ Shows which workers will be stopped
- ✅ Displays resource usage after changes
- ✅ Optuna database shared (no data loss)
- ✅ Training continues uninterrupted

---

## See Full Guide

```bash
cat TRAINING_CONTROL_README.md
```

---

**TL;DR:**
- `./training_minimal.sh` = Free GPU
- `./training_full.sh` = Max speed
- Training never stops, just scales up/down
