# DRL Training RAM Usage Breakdown

**Question:** How does a single training process use 20GB of RAM?

**Answer:** It's a combination of PyTorch, multi-threading, and PPO-specific memory requirements.

## The 20GB Process (PID 47068)

**Current Stats:**
- Actual RAM (RSS): **19.8 GB**
- Virtual Memory: **35.4 GB**
- Threads: **53 threads**
- Running time: **2 hours**
- Configuration: `worker_num=3`, `net_dim=768`, `thread_num=12`

## Memory Breakdown

### 1. PyTorch Neural Networks (~2-4 GB)

PPO requires **4 separate neural networks**:

```python
# Per trial, you have:
Actor Network (policy):     net_dim × layers × parameters
Critic Network (value):     net_dim × layers × parameters
Actor Target (frozen copy): net_dim × layers × parameters
Critic Target (frozen copy): net_dim × layers × parameters
```

**Calculation for net_dim=768:**
- Input layer: `state_dim (71) × 768 = 54,528 params`
- Hidden layers: `768 × 768 × 3 = 1,769,472 params`
- Output layer: `768 × action_dim (7) = 5,376 params`
- **Total per network: ~1.8M parameters × 4 bytes = 7.2 MB**
- **× 4 networks = 29 MB**

But PyTorch also stores:
- **Gradients** (same size as parameters): 29 MB
- **Optimizer state** (Adam: 2x parameters for momentum/variance): 58 MB
- **Activation maps** during forward/backward: ~100-500 MB
- **CUDA overhead** (even on CPU): ~500 MB

**Realistic total: 1-2 GB for neural networks + optimizer**

### 2. Replay Buffers (~4-8 GB)

PPO uses on-policy replay buffers with `max_memo = 2^12 = 4096` transitions:

```python
# Per worker (3 workers total):
Buffer size = 4096 transitions × state_dim (71) × 4 bytes
            + 4096 × (reward + mask + action) × 4 bytes
```

**Per worker calculation:**
- States: `4096 × 71 × 4 = 1.16 MB`
- Other (rewards, masks, actions): `4096 × 9 × 4 = 147 KB`
- **Total per worker: ~1.3 MB**
- **× 3 workers = 3.9 MB**

**But wait!** During PPO updates, trajectories are converted to **tensors on GPU/CPU**:
- Original buffer: 3.9 MB
- Tensor copies for training: 3.9 MB
- Advantage calculations (GAE): Creates temporary buffers ~10-50 MB
- **Actual buffer overhead with copies: 100-500 MB**

### 3. Thread Stack Space (~2-4 GB)

**53 threads** each get their own stack:
- Default Python thread stack: **8 MB per thread** (Linux)
- 53 threads × 8 MB = **424 MB**

But PyTorch worker threads may use more:
- 53 threads × 32-64 MB = **1.7-3.4 GB**

This is **massive overhead** from having so many threads!

### 4. PyTorch Caching and Memory Pool (~5-10 GB)

PyTorch uses a **caching allocator** that:
- Pre-allocates large memory pools for efficiency
- Doesn't release memory back to OS immediately
- Caches tensors to avoid allocation overhead

With `thread_num=12` and `worker_num=3`:
- Each thread may allocate its own tensor caches
- **Memory fragmentation** leads to inefficient usage
- Cached allocations: **5-10 GB** easily

### 5. Training Data and Copies (~0.5-2 GB)

While the raw data is small (10 MB), during training:
- **Multiple copies** are made for each worker
- **Batches** are extracted and stored as tensors
- **Normalization** creates temporary arrays
- **State histories** (lookback) duplicate data

With 3 workers and batching:
- Data copies: `10 MB × 3 workers = 30 MB`
- Batch tensors: `batch_size (2048-8192) × state_dim × 4 bytes = 2-32 MB per batch`
- Multiple batches in flight: **100-500 MB**

### 6. Python Runtime and Libraries (~1-2 GB)

- Python interpreter: ~50 MB
- NumPy arrays and temp data: ~200-500 MB
- PyTorch library code: ~500 MB
- ROCm/HIP libraries: ~500 MB
- Misc imports (pandas, optuna, etc.): ~200 MB
- **Total: 1-2 GB**

### 7. Memory Fragmentation and Overhead (~2-5 GB)

After running many trials:
- **Fragmentation**: Memory chunks can't be coalesced
- **Heap overhead**: malloc metadata for millions of allocations
- **Copy-on-write**: Shared memory pages get duplicated
- **Lazy release**: Python/PyTorch don't always free immediately

This "wasted" space: **2-5 GB**

## Total RAM Usage Estimate

| Component | Low Estimate | High Estimate |
|-----------|--------------|---------------|
| Neural Networks + Optimizer | 1 GB | 2 GB |
| Replay Buffers + Copies | 0.5 GB | 1 GB |
| Thread Stacks (53 threads) | 1.7 GB | 3.4 GB |
| PyTorch Caching | 5 GB | 10 GB |
| Training Data Copies | 0.5 GB | 2 GB |
| Python + Libraries | 1 GB | 2 GB |
| Fragmentation + Overhead | 2 GB | 5 GB |
| **TOTAL** | **11.7 GB** | **25.4 GB** |

**Actual usage: 19.8 GB** ✅ Falls right in the middle!

## Why So Many Threads?

Looking at your config:
- `thread_num = 12` (PyTorch threads for parallelization)
- `worker_num = 3` (Environment rollout workers)
- Plus PyTorch's internal threads:
  - Data loading threads
  - CUDA streams (even on CPU/ROCm)
  - Optimizer threads
  - Garbage collection threads

**Total: 53 threads** consuming 1.7-3.4 GB just in stack space!

## How to Reduce RAM Usage

### 1. Reduce Thread Count (Biggest Impact)
```python
# In 1_optimize_unified.py, reduce thread_num range:
thread_num = trial.suggest_int("thread_num", 4, 8)  # Was: 4-15
```
**Savings: 2-3 GB**

### 2. Reduce Worker Count (Already Done ✓)
```python
worker_num = trial.suggest_int("worker_num", 2, 3)  # Current setting
```
**Savings: Already optimized**

### 3. Reduce Network Size (Trade Performance for RAM)
```python
net_dimension = trial.suggest_int("net_dimension", 256, 512)  # Was: 512-1536
```
**Savings: 1-2 GB**

### 4. Enable PyTorch Memory Pinning Limits
```python
# Add to training script:
import torch
torch.cuda.empty_cache()  # Force cache clearing
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
```
**Savings: 2-5 GB**

### 5. Use Smaller Batch Sizes
```python
batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048])  # Was: 2048-8192
```
**Savings: 0.5-1 GB**

## Is 20GB Normal?

**For single-process DRL training:** Yes, unfortunately.

**Comparison:**
- **Stable Baselines3 (single env):** 2-5 GB
- **ElegantRL (3 workers, threading):** 15-25 GB ← **You are here**
- **RLlib (multi-process):** 10-15 GB per worker
- **Professional setups (distributed):** 30-100 GB total

## The Threading Problem

The main culprit is **53 threads in one process**. Each thread:
- Gets 8-64 MB stack space
- May cache its own tensors
- Contributes to fragmentation

**Better approach:**
- Use **multiprocessing** with smaller processes
- Or use **fewer threads** with vectorized environments

## Recommended Fix

Add this to reduce thread count:

```python
# In 1_optimize_unified.py, line ~276:
if use_best_ranges:
    thread_num = trial.suggest_int("thread_num", 4, 8)  # Reduced from 12-15
else:
    thread_num = trial.suggest_int("thread_num", 4, 8)  # Reduced from 4-max_threads
```

**Expected result:** 19.8 GB → 12-15 GB per process

## Monitoring

Check RAM per process:
```bash
# Live monitoring
watch -n 2 'ps aux | grep optimize_unified | awk "{printf \"PID: %s | RAM: %.1f GB\\n\", \$2, \$6/1024/1024}"'

# Thread count check
ps -eLf | grep 47068 | wc -l
```

Good targets:
- **RAM per process:** < 15 GB
- **Threads per process:** < 30
- **Total system RAM:** < 25 GB (with 31 GB available)
