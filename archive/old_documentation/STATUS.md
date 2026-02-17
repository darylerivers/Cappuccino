# LUDICROUS MODE - System Status

## Current Configuration

### Training Processes (4 Active)
1. **Study: maxvram_v2** (PID 166195) - Main study with 6.5GB VRAM preallocation
2. **Study: maxvram_parallel_2** (PID 168755) - Parallel study
3. **Study: maxvram_parallel_3** (PID 171358) - Parallel study
4. **Study: maxvram_parallel_4** (PID 171493) - Parallel study

### Resource Usage
- **GPU VRAM**: 7.7GB / 8.2GB (94% utilized)
- **GPU Compute**: 86%
- **GPU Memory Bandwidth**: 96%
- **RAM**: 10GB / 31GB (32% utilized)
- **CPU**: 400%+ (4 processes at 100% each)

### Training Parameters (LUDICROUS)
- **Batch Sizes**: 131,072 - 262,144
- **Network Dimensions**: 8,192 - 10,240
- **Workers per study**: 128-256 parallel environments
- **Target Steps**: 32,768 - 131,072
- **Dataset**: 2-year (17,246 samples)

## Kill Switch Status

**Active**: Yes (PID in logs/kill_switch.log)
**Mode**: both (monitors GPU and RAM)
**Thresholds**:
- GPU VRAM: 98%
- RAM: 95%

### Manual Kill Switch Usage

```bash
# Kill by resource type
./kill_switch.sh gpu 95     # Kill if GPU > 95%
./kill_switch.sh ram 90     # Kill if RAM > 90%
./kill_switch.sh both 98 95 # Kill if GPU>98% OR RAM>95% (current)

# Emergency manual kill
pkill -f 1_optimize_unified

# Graceful shutdown
pkill -TERM -f 1_optimize_unified
```

## Monitoring Commands

```bash
# Real-time status
watch -n 2 './summary_status.sh'

# GPU detailed monitoring
nvidia-smi dmon -s mu

# Individual study logs
tail -f logs/training_maxvram.log
tail -f logs/training_parallel_2.log
tail -f logs/training_parallel_3.log
tail -f logs/training_parallel_4.log

# Kill switch log
tail -f logs/kill_switch.log

# All training output
tail -f logs/training_*.log
```

## Study Databases

- `databases/optuna_maxvram_v2.db` - Main study
- `databases/optuna_maxvram_parallel_2.db` - Parallel 2
- `databases/optuna_maxvram_parallel_3.db` - Parallel 3
- `databases/optuna_maxvram_parallel_4.db` - Parallel 4

## Scripts

- `./summary_status.sh` - Quick status overview
- `./kill_switch.sh` - Automatic resource monitoring kill switch
- `./launch_parallel.sh` - Launch parallel studies
- `./launch_parallel_no_prealloc.sh` - Launch parallel without VRAM prealloc

## Performance Notes

- GPU is **MAXED OUT** at 94% VRAM, 86% compute, 96% memory bandwidth
- RAM has 21GB available for even more parallel studies if needed
- Each study is exploring massive batch sizes (up to 262k) and huge networks (10k+ dimensions)
- Training will continue until hitting 2000 trials (main) or 500 trials (parallel studies)

## Expected Behavior

- GPU memory will stay at 93-95% (kill switch triggers at 98%)
- RAM will gradually increase as PyTorch caches grow
- If OOM occurs, kill switch will activate and stop all training
- Models are saved continuously in `train_results/`
