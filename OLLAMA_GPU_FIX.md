# Fix Ollama to Use AMD GPU

## Problem

Ollama is currently running GLM-4 on **CPU only**:

```bash
$ ollama ps
NAME           PROCESSOR
glm4:latest    100% CPU     ❌ Should be GPU!
```

## Root Cause

Ollama was installed when you had an **NVIDIA RTX 3070** (CUDA), but you now have an **AMD RX 7900 GRE** (ROCm). Ollama needs ROCm environment variables to detect AMD GPUs:

**Missing variables:**
- `HSA_OVERRIDE_GFX_VERSION` - Tells ROCm which GPU architecture (gfx1100 for RX 7900)
- `HIP_VISIBLE_DEVICES` - Which GPU to use
- `ROCR_VISIBLE_DEVICES` - ROCm device visibility

**Current log shows:**
```
HIP_VISIBLE_DEVICES: (empty)
ROCR_VISIBLE_DEVICES: (empty)
GPULayers:[] (no GPU layers!)
```

## Solution

### Option 1: Quick Fix (Run as User)

**Pros:** No sudo needed, test immediately
**Cons:** Only works for current session

```bash
# Stop current Ollama models
ollama stop glm4

# Set ROCm environment variables
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0

# Kill and restart Ollama manually
pkill ollama
OLLAMA_DEBUG=1 ollama serve > /tmp/ollama.log 2>&1 &

# Wait for startup
sleep 5

# Load GLM with GPU
ollama run glm4:latest "Test GPU"

# Check if using GPU
ollama ps
# Should show: PROCESSOR: 100% GPU (or X% GPU)
```

### Option 2: Permanent Fix (Systemd Service)

**Pros:** Permanent, survives reboots
**Cons:** Requires sudo

```bash
# Run the fix script (requires sudo password)
chmod +x fix_ollama_gpu.sh
./fix_ollama_gpu.sh

# This will:
# 1. Create /etc/systemd/system/ollama.service.d/rocm.conf
# 2. Set HSA_OVERRIDE_GFX_VERSION=11.0.0
# 3. Set HIP_VISIBLE_DEVICES=0
# 4. Restart Ollama service
# 5. Verify GPU detection
```

### Option 3: Alternative Install (ROCm-enabled Ollama)

If neither works, reinstall Ollama with ROCm support:

```bash
# Download ROCm-enabled Ollama (if available for Arch)
# OR build from source with ROCm support

# Check if current Ollama supports ROCm
ldd $(which ollama) | grep -i hip

# If no HIP libraries found, Ollama was built for CUDA only
```

## Verification

After applying the fix:

```bash
# 1. Check Ollama process status
ollama ps
# Expected: glm4:latest    5.4 GB    100% GPU ✅

# 2. Check GPU usage while running GLM
rocm-smi
# Expected: GPU% should be >0% when GLM is responding

# 3. Test inference speed
time ollama run glm4:latest "Write a Python function to calculate Sharpe ratio"
# Expected: Should be MUCH faster than CPU
```

## Why This Matters

**CPU Inference:**
- GLM-4 response time: 30-60+ seconds
- Blocks CPU for other tasks
- High CPU usage (100%+)
- Slower training if running simultaneously

**GPU Inference:**
- GLM-4 response time: 3-5 seconds (10x faster!)
- Frees up CPU for training
- Uses idle GPU capacity
- Can run alongside training (different GPU memory)

## Expected Results

### Before (CPU):
```
$ ollama ps
NAME           SIZE     PROCESSOR
glm4:latest    5.4 GB   100% CPU

$ time ollama run glm4 "Hello"
# Takes 30-60 seconds ❌
```

### After (GPU):
```
$ ollama ps
NAME           SIZE     PROCESSOR
glm4:latest    5.4 GB   100% GPU

$ time ollama run glm4 "Hello"
# Takes 3-5 seconds ✅

$ rocm-smi
GPU%: 15-30% (when GLM responding)
VRAM: +5GB (GLM loaded)
```

## Architecture Details

**RX 7900 GRE GPU:**
- Architecture: RDNA 3 (gfx1100)
- Compute Units: 80
- VRAM: 16GB
- ROCm Support: ✅ Yes (ROCm 5.4+)

**HSA_OVERRIDE_GFX_VERSION=11.0.0:**
- Tells ROCm to treat GPU as gfx1100
- Required for RDNA 3 GPUs
- Without it, ROCm won't recognize the GPU

## Troubleshooting

### "Still showing CPU after fix"

```bash
# Check if environment variables are set
journalctl -u ollama -n 50 | grep HSA_OVERRIDE

# Should see: HSA_OVERRIDE_GFX_VERSION=11.0.0
# If not, systemd override didn't work
```

### "Ollama won't start after changes"

```bash
# Check Ollama logs
journalctl -u ollama -n 100 --no-pager

# Common issue: Wrong GFX version
# Try: HSA_OVERRIDE_GFX_VERSION=11.0.1
# Or:  HSA_OVERRIDE_GFX_VERSION=11.0.2
```

### "GPU shows 0% usage"

```bash
# Verify ROCm can see the GPU
rocminfo | grep -A 5 "Name.*gfx"

# Should show gfx1100 device
# If not, ROCm driver issue
```

### "Out of memory error"

```bash
# GLM-4 needs ~5-6GB VRAM
# If training is using >10GB, not enough room

# Solution: Stop training temporarily
pkill -f 1_optimize_unified.py

# Or: Use smaller model
ollama pull qwen2.5-coder:3b  # Only 2GB VRAM
```

## Running Alongside Training

Your RX 7900 GRE (16GB VRAM) can handle both:

| Process | VRAM Usage |
|---------|------------|
| DRL Training | 6-8GB |
| GLM-4 Inference | 5-6GB |
| **Total** | **11-14GB** ✅ Fits in 16GB |

Just ensure training doesn't use >10GB (use `--batch-size` to limit).

## Next Steps

1. **Run the fix:** `./fix_ollama_gpu.sh`
2. **Verify GPU:** `ollama ps` should show GPU
3. **Test speed:** GLM responses should be 3-5 seconds
4. **Monitor:** `watch rocm-smi` while using GLM

---

**Run this now:**
```bash
chmod +x fix_ollama_gpu.sh
./fix_ollama_gpu.sh
```
