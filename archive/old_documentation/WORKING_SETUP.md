# WORKING TRAINING SETUP

## What Was Wrong

1. **Batch sizes too large**: 98k-262k batches don't fit in 8GB VRAM
2. **Network too large**: 8k-10k dimensions are massive overkill
3. **Too many workers**: 128-256 workers is insane for single GPU
4. **VRAM preallocation**: Locking 2-6GB before training doesn't help
5. **Pruning enabled**: All trials were being pruned, none completed

## Working Configuration

### Realistic Parameters
- **Batch sizes**: 8,192 - 32,768 (fits in VRAM)
- **Network dimensions**: 512 - 2,048 (reasonable)
- **Workers**: 8 - 16 (realistic parallelism)
- **VRAM preallocation**: DISABLED
- **Pruning**: DISABLED (trials complete)

### Current Setup
- **Training PID**: Check with `ps aux | grep 1_optimize_unified`
- **Database**: `/tmp/optuna_working.db`
- **Study**: `working`
- **Trials**: 100 total
- **Data**: 2-year dataset (17,246 samples)

### Components Running
1. **Training**: `1_optimize_unified.py` - runs trials
2. **Orchestrator**: `pipeline_orchestrator.py` - processes complete trials
3. **Kill switch**: `kill_switch.sh` - monitors GPU/RAM

## Monitoring

```bash
# Watch training
tail -f logs/training_working.log

# Check completed trials
sqlite3 /tmp/optuna_working.db "SELECT number, state, value FROM trials ORDER BY number DESC LIMIT 10"

# GPU status
nvidia-smi

# Orchestrator log
tail -f logs/pipeline_working.log
```

## Expected Behavior

1. Each trial takes ~15-30 minutes
2. Trials complete (not pruned)
3. Orchestrator processes them every 30 minutes
4. GPU stays at 70-90% utilization
5. VRAM ~2-4GB (not maxed)

## If OOM Happens Again

The batch sizes might still be too large. Edit:
```bash
# Reduce batch sizes further
vim 1_optimize_unified.py
# Change: batch_size = trial.suggest_categorical("batch_size", [4096, 8192, 16384])
```

## Key Files
- Training script: `1_optimize_unified.py`
- Config: `config/pipeline_config.json`
- Logs: `logs/training_working.log`, `logs/pipeline_working.log`
- Database: `/tmp/optuna_working.db`
