# AMD GPU Migration Guide: NVIDIA RTX 3070 → AMD RX 7900 XTX

**Current:** NVIDIA RTX 3070 (8GB VRAM, CUDA)
**Target:** AMD RX 7900 XTX (24GB VRAM, ROCm)
**Date:** January 16, 2026

---

## Executive Summary

**Good News:**
- ✅ PyTorch fully supports AMD GPUs via ROCm
- ✅ Your code is mostly GPU-agnostic (uses `torch.device`)
- ✅ 3x VRAM upgrade (8GB → 24GB) - can run more workers!
- ✅ Similar performance for ML workloads

**Changes Needed:**
- Replace CUDA with ROCm runtime
- Reinstall PyTorch for ROCm
- Update monitoring scripts (nvidia-smi → rocm-smi)
- Minor code adjustments (device selection)

**Effort:** Medium (2-3 hours)
**Risk:** Low (well-tested path)

---

## Part 1: System Changes

### 1.1 Install ROCm

**Remove NVIDIA drivers:**
```bash
# Not needed - you'll physically remove the GPU
# But clean up CUDA packages:
sudo apt remove --purge nvidia-* cuda-*
sudo apt autoremove
```

**Install ROCm (AMD's CUDA equivalent):**
```bash
# Add ROCm repository
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb
sudo dpkg -i amdgpu-install_6.0.60000-1_all.deb
sudo apt update

# Install ROCm
sudo amdgpu-install --usecase=rocm

# Add user to video/render groups
sudo usermod -a -G video,render $USER

# Reboot required
sudo reboot
```

**Verify ROCm installation:**
```bash
rocm-smi
# Should show your RX 7900 XTX with 24GB VRAM
```

### 1.2 Reinstall PyTorch for ROCm

**Uninstall CUDA PyTorch:**
```bash
pip uninstall torch torchvision torchaudio
```

**Install ROCm PyTorch:**
```bash
# For ROCm 6.0 (latest)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

**Verify PyTorch sees AMD GPU:**
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"ROCm available: {torch.cuda.is_available()}")  # Still returns True for ROCm!
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

Expected output:
```
PyTorch version: 2.8.0+rocm6.0
ROCm available: True
GPU: AMD Radeon RX 7900 XTX
VRAM: 24.00 GB
```

---

## Part 2: Code Changes

### 2.1 Monitoring Scripts (nvidia-smi → rocm-smi)

**Files to update:**
- `performance_monitor.py`
- `monitor.py`
- `dashboard.py`
- `system_watchdog.py`

**Find all nvidia-smi calls:**
```bash
grep -r "nvidia-smi" /opt/user-data/experiment/cappuccino --include="*.py"
```

**Changes needed:**

#### Before (NVIDIA):
```python
subprocess.run([
    'nvidia-smi',
    '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
    '--format=csv,noheader,nounits'
])
```

#### After (AMD):
```python
# Option 1: Use rocm-smi (similar to nvidia-smi)
subprocess.run(['rocm-smi', '--showuse', '--showmeminfo', '--showtemp', '--showpower'])

# Option 2: Parse rocm-smi output
import subprocess
result = subprocess.check_output(['rocm-smi', '--showuse', '--csv'], text=True)
# Parse CSV output
```

**ROCm-SMI command reference:**
```bash
rocm-smi                           # Overview
rocm-smi --showuse                 # GPU utilization
rocm-smi --showmeminfo vram       # VRAM usage
rocm-smi --showtemp               # Temperature
rocm-smi --showpower              # Power draw
rocm-smi --csv                    # CSV format
```

### 2.2 GPU Device Selection

**Your code already handles this well:**

```python
# This works for both NVIDIA and AMD!
self.device = torch.device("cpu" if gpu_id < 0 else f"cuda:{gpu_id}")
```

**Why it works:**
- PyTorch uses "cuda" for both NVIDIA CUDA and AMD ROCm
- The abstraction layer handles the differences
- No changes needed in most code!

**Files that use this pattern (already compatible):**
- ✅ `agent_ddqn.py`
- ✅ `simple_ensemble_agent.py`
- ✅ `paper_trader_alpaca_polling.py`
- ✅ `multi_timeframe_ensemble_agent.py`

### 2.3 Environment Variables

**Change CUDA environment variables:**

#### Before:
```bash
export CUDA_VISIBLE_DEVICES=0
```

#### After:
```bash
export HIP_VISIBLE_DEVICES=0  # HIP is AMD's CUDA equivalent
export ROCR_VISIBLE_DEVICES=0
```

**In Python code:**
```python
# Before
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

# After (works for both)
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)
```

### 2.4 Requirements File

**Update `/opt/user-data/experiment/cappuccino/requirements.txt`:**

#### Before:
```
# GPU-enabled PyTorch (CUDA 12.1)
torch>=2.6.0
torchvision>=0.21.0
```

#### After:
```
# GPU-enabled PyTorch (ROCm 6.0 for AMD GPUs)
# Install with: pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
torch>=2.6.0
torchvision>=0.21.0
```

---

## Part 3: Performance Optimizations

### 3.1 Worker Scaling

**Your current setup:**
- 9 workers on RTX 3070 (8GB VRAM)
- VRAM: 82% utilization (~6.7 GB)

**With RX 7900 XTX (24GB VRAM):**
- Theoretical: 9 × (24 / 8) = **27 workers**
- Recommended: **20-24 workers** (leave headroom)

**Calculate optimal workers:**
```python
# Current VRAM per worker
vram_per_worker = 6.7 / 9  # ~744 MB per worker

# Available VRAM on RX 7900 XTX
total_vram = 24000  # MB
safe_usage = 0.90   # Use 90%

# Maximum workers
max_workers = int((total_vram * safe_usage) / vram_per_worker)
print(f"Recommended workers: {max_workers}")  # ~29 workers
```

**Realistic target: 20 workers** (2.2x increase from current 9)

### 3.2 Batch Size Optimization

**With 3x more VRAM, you can:**
- Increase batch sizes (faster training)
- Use larger models (better performance)
- Run more trials simultaneously

**Current batch sizes:**
```python
# Check current batch size
grep -r "batch_size" /opt/user-data/experiment/cappuccino --include="*.py"
```

**Suggested increases:**
- 2048 → 4096 (2x)
- 3072 → 6144 (2x)

### 3.3 Expected Performance

**Training speed comparison:**

| Metric | RTX 3070 | RX 7900 XTX | Change |
|--------|----------|-------------|--------|
| **VRAM** | 8 GB | 24 GB | +3x |
| **Workers** | 9 | 20 | +2.2x |
| **Trials/hour** | 40-45 | **90-100** | +2.2x |
| **Time to 1000 trials** | 22 hours | **10 hours** | -54% |
| **Time to 5000 trials** | 111 hours | **50 hours** | -55% |

**Note:** Raw compute performance is similar, but VRAM allows more parallelism.

---

## Part 4: Migration Checklist

### Before GPU Swap

- [ ] Stop all training workers
  ```bash
  ps aux | grep "1_optimize_unified.py" | awk '{print $2}' | xargs kill
  ```

- [ ] Stop all automation
  ```bash
  cd /opt/user-data/experiment/cappuccino
  ./stop_automation.sh
  ```

- [ ] Backup current state
  ```bash
  cp paper_trades/positions_state.json paper_trades/positions_state_backup.json
  sqlite3 databases/optuna_cappuccino.db ".backup databases/optuna_backup_$(date +%Y%m%d).db"
  ```

- [ ] Note current trial count
  ```bash
  sqlite3 databases/optuna_cappuccino.db "SELECT COUNT(*) FROM trials WHERE study_id = (SELECT study_id FROM studies WHERE study_name = 'cappuccino_2year_20260112');"
  ```

### Physical GPU Swap

- [ ] Shutdown system
- [ ] Remove NVIDIA RTX 3070
- [ ] Install AMD RX 7900 XTX
- [ ] Boot into BIOS, verify GPU detected
- [ ] Boot Linux

### Software Installation

- [ ] Install ROCm (see Part 1.1)
- [ ] Verify ROCm: `rocm-smi`
- [ ] Reinstall PyTorch for ROCm (see Part 1.2)
- [ ] Test PyTorch GPU access

### Code Updates

- [ ] Update monitoring scripts (nvidia-smi → rocm-smi)
- [ ] Test monitor: `python monitor.py --study-name cappuccino_2year_20260112`
- [ ] Update requirements.txt
- [ ] Update any startup scripts

### Verification

- [ ] Run single trial test
  ```bash
  python 1_optimize_unified.py --n-trials 1 --gpu 0 --study-name test_rocm
  ```

- [ ] Check GPU utilization: `watch -n 1 rocm-smi`
- [ ] Verify model training works
- [ ] Check trial completes successfully

### Restart Production

- [ ] Start with 5 workers (conservative)
  ```bash
  for i in {1..5}; do
    nohup python -u 1_optimize_unified.py --n-trials 500 --gpu 0 --study-name cappuccino_2year_20260112 > logs/worker_$i.log 2>&1 &
    sleep 5
  done
  ```

- [ ] Monitor VRAM usage: `watch -n 1 rocm-smi --showmeminfo vram`
- [ ] If VRAM <70%, add more workers gradually
- [ ] Target: 20 workers for optimal utilization

- [ ] Restart automation
  ```bash
  ./start_automation.sh
  ```

- [ ] Restart paper trading
  - Ensemble trader should auto-restart via watchdog
  - Single model trader: check PID and restart if needed

---

## Part 5: Monitoring Script Updates

### Create GPU-agnostic wrapper

**File: `/opt/user-data/experiment/cappuccino/gpu_monitor.py`**

```python
#!/usr/bin/env python3
"""
GPU monitoring wrapper - works with both NVIDIA and AMD GPUs
"""
import subprocess
import re

def get_gpu_stats():
    """Get GPU statistics (vendor-agnostic)"""
    try:
        # Try ROCm first (AMD)
        result = subprocess.check_output(['rocm-smi', '--showuse', '--showmeminfo', 'vram', '--showtemp', '--showpower', '--csv'], text=True)
        return parse_rocm_output(result)
    except FileNotFoundError:
        try:
            # Fall back to nvidia-smi (NVIDIA)
            result = subprocess.check_output([
                'nvidia-smi',
                '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], text=True)
            return parse_nvidia_output(result)
        except FileNotFoundError:
            return None

def parse_rocm_output(output):
    """Parse rocm-smi CSV output"""
    # Implementation depends on rocm-smi CSV format
    return {
        'vendor': 'AMD',
        'utilization': 0,  # Parse from output
        'memory_used': 0,
        'memory_total': 24576,
        'temperature': 0,
        'power': 0
    }

def parse_nvidia_output(output):
    """Parse nvidia-smi output"""
    parts = output.strip().split(',')
    return {
        'vendor': 'NVIDIA',
        'utilization': int(parts[0]),
        'memory_used': int(parts[1]),
        'memory_total': int(parts[2]),
        'temperature': int(parts[3]),
        'power': float(parts[4])
    }

if __name__ == '__main__':
    stats = get_gpu_stats()
    if stats:
        print(f"GPU: {stats['vendor']}")
        print(f"Utilization: {stats['utilization']}%")
        print(f"Memory: {stats['memory_used']}/{stats['memory_total']} MB")
        print(f"Temperature: {stats['temperature']}°C")
        print(f"Power: {stats['power']}W")
    else:
        print("No GPU detected")
```

Then update your monitoring scripts to use this wrapper.

---

## Part 6: Troubleshooting

### Issue: PyTorch doesn't see GPU

**Check ROCm installation:**
```bash
rocm-smi
rocminfo | grep "Name:"
```

**Check PyTorch ROCm support:**
```python
import torch
print(torch.__version__)  # Should include +rocm6.0
print(torch.cuda.is_available())  # Should be True
```

**Fix:** Reinstall PyTorch with correct ROCm index:
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
```

### Issue: Training slower than expected

**Check GPU utilization:**
```bash
watch -n 1 rocm-smi --showuse
```

**If utilization <80%:**
- Add more workers
- Increase batch size
- Check for CPU bottlenecks

### Issue: Out of memory errors

**Check VRAM usage:**
```bash
rocm-smi --showmeminfo vram
```

**Solutions:**
- Reduce number of workers
- Reduce batch size
- Check for memory leaks

### Issue: ROCm commands not found

**Add ROCm to PATH:**
```bash
echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## Part 7: Expected Benefits

### Performance Improvements

**Training:**
- 2.2x more trials/hour (9 → 20 workers)
- Reach 1,000 trials: 22h → 10h (-54%)
- Reach 5,000 trials: 111h → 50h (-55%)

**Flexibility:**
- Can increase batch sizes for faster convergence
- Can train larger models (if needed)
- Can run more experiments simultaneously

**Paper Trading:**
- No impact (uses CPU/minimal GPU)
- Same behavior and performance

### Cost Considerations

**RX 7900 XTX (~$900-1000):**
- ✅ 3x VRAM (8GB → 24GB)
- ✅ Similar compute performance
- ✅ Lower power consumption (~300W vs 220W for 3070)
- ✅ Better Linux driver support (ROCm is open source)

**Alternative: RTX 4080 Super (~$1000):**
- VRAM: 16GB (2x, not 3x)
- Better CUDA support (more mature)
- Similar price point

---

## Part 8: Quick Reference

### Command Comparison

| Task | NVIDIA | AMD |
|------|--------|-----|
| **GPU Status** | `nvidia-smi` | `rocm-smi` |
| **Utilization** | `nvidia-smi --query-gpu=utilization.gpu --format=csv` | `rocm-smi --showuse` |
| **Memory** | `nvidia-smi --query-gpu=memory.used,memory.total --format=csv` | `rocm-smi --showmeminfo vram` |
| **Temperature** | `nvidia-smi --query-gpu=temperature.gpu --format=csv` | `rocm-smi --showtemp` |
| **Power** | `nvidia-smi --query-gpu=power.draw --format=csv` | `rocm-smi --showpower` |
| **Watch Mode** | `watch -n 1 nvidia-smi` | `watch -n 1 rocm-smi` |

### PyTorch Code (Works for Both!)

```python
import torch

# Check if GPU available (works for CUDA and ROCm)
has_gpu = torch.cuda.is_available()

# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get GPU name
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Using: {gpu_name}")

# Check VRAM
if torch.cuda.is_available():
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {total_mem:.1f} GB")
```

---

## Part 9: Timeline

**Estimated migration time:** 2-3 hours

1. **Preparation (15 min):** Stop processes, backup data
2. **Physical swap (30 min):** Remove NVIDIA, install AMD, boot
3. **ROCm installation (30 min):** Install drivers and runtime
4. **PyTorch reinstall (15 min):** Install ROCm-enabled PyTorch
5. **Code updates (45 min):** Update monitoring scripts
6. **Testing (30 min):** Verify GPU works, run test trial
7. **Restart production (15 min):** Start workers and automation

---

## Part 10: Recommendation

**Should you upgrade?**

**Yes, if:**
- ✅ You want to train 2x faster (20 vs 9 workers)
- ✅ You're comfortable with 2-3 hours of downtime
- ✅ You want to explore larger models/batch sizes
- ✅ You prefer open-source drivers (ROCm)

**Wait, if:**
- ⏸️ Currently in critical training phase (wait for milestone)
- ⏸️ Not comfortable with potential compatibility issues
- ⏸️ Need maximum stability (CUDA more mature)

**My recommendation:** **Go for it!** The RX 7900 XTX is a great upgrade, PyTorch's ROCm support is mature, and you'll get ~2x training speed. Just do the migration during a non-critical time (like after hitting 1,000 trials).

---

**Migration Guide Version:** 1.0
**Date:** January 16, 2026
**Status:** Ready for implementation
