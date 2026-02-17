# Pipeline Troubleshooting Guide

## Issue: Backtest Gate Failed

### Symptoms
```
[WARNING] Trial XXX FAILED backtest gate: Sharpe 0.000 below threshold 0.30
```

### Possible Causes

#### 1. Most Common: Old trials have weights cleaned up

**Check:**
```bash
ls -la train_results/cwd_tests/trial_XXX_1h/
```

If you see `.cleanup_done` file but no `actor.pth` or `critic.pth`, the weights were removed to save disk space.

**Solution:**
- Use recently trained trials that still have weights
- Or add specific trials to manual list in config:
  ```json
  {
    "pipeline": {
      "use_manual_trials": true,
      "manual_trials": [965, 1000, 1500]
    }
  }
  ```

#### 2. Validation Data Missing

**Check:**
```bash
ls -la data/1h_1680/val/
```

Should contain:
- `price_array.npy`
- `tech_array.npy`

**Solution:**
```bash
# If val/ doesn't exist, pipeline will fall back to full dataset
# Ensure data/1h_1680/ exists with data files
```

#### 3. Model Loading Issues

**Check logs:**
```bash
grep "Error\|Failed" logs/pipeline_orchestrator.log | tail -20
```

**Common issues:**
- Missing dependencies (torch, numpy, pandas)
- GPU/CPU device mismatch
- Incompatible model format

#### 4. Zero Returns (Model Not Trading)

If the model loads but returns 0.0 for all metrics, it might:
- Not be making any trades
- Have incorrect environment parameters
- Be using stale/incompatible data format

**Debug:**
```bash
# Run backtest manually to see detailed output
python 4_backtest.py --results train_results/cwd_tests/trial_965_rerun_1h --gpu -1
```

## Issue: No Trials Found

### Symptoms
```
[WARNING] Skipping trial XXX - model directory not found
```

All top trials skipped, none added to pipeline.

### Solution

**Option 1: Use Manual Trials**

Edit `config/pipeline_config.json`:
```json
{
  "pipeline": {
    "use_manual_trials": true,
    "manual_trials": [965]
  }
}
```

**Option 2: Find Trials with Weights**

```bash
# Find all trials with saved weights
for dir in train_results/cwd_tests/trial_*; do
  if [ -f "$dir/actor.pth" ] && [ -f "$dir/critic.pth" ]; then
    echo "$dir has weights"
  fi
done
```

**Option 3: Preserve Weights for New Trials**

When training new trials, ensure weights are saved:
- Don't run cleanup scripts on recent trials
- Keep weights for top N performing trials
- Or save to separate backup location

## Issue: CGE Stress Test Fails

### Symptoms
```
[ERROR] CGE stress test failed for trial XXX
```

### Possible Causes

1. **Missing CGE Data**
   ```bash
   ls data/cge_synthetic/
   # Should have synthetic_XXXX.npy files
   ```

2. **Insufficient Disk Space**
   ```bash
   df -h .
   # Need ~500MB for 200 scenarios
   ```

3. **Timeout**
   - CGE tests take 30-60 minutes
   - Increase timeout in `pipeline/cge_runner.py` if needed

## Testing the Pipeline

### Demo Mode (Trial 965)

```bash
./demo_pipeline.sh
```

This tests with trial 965 which has saved weights.

### Manual Test with Specific Trial

```bash
# 1. Edit config
vim config/pipeline_config.json
# Set: "use_manual_trials": true, "manual_trials": [YOUR_TRIAL]

# 2. Clear state
rm deployments/pipeline_state.json

# 3. Run once
python pipeline_orchestrator.py --once

# 4. Check results
cat deployments/pipeline_state.json | jq
```

### Dry Run Mode

Test without making actual deployments:

```bash
python pipeline_orchestrator.py --dry-run --once
```

## Adjusting Gate Thresholds

If trials are failing gates too aggressively, adjust thresholds in `config/pipeline_config.json`:

```json
{
  "gates": {
    "backtest": {
      "thresholds": {
        "mature": {
          "max_loss": -0.05,     // More lenient
          "min_sharpe": 0.1,     // Lower requirement
          "max_drawdown": 0.25   // Higher tolerance
        }
      }
    }
  }
}
```

## Common Workflow Issues

### Pipeline Not Processing New Trials

**Check:**
1. Study name in config matches `.env.training`
2. Trials have model weights saved
3. No emergency stop file exists:
   ```bash
   rm deployments/pipeline_emergency_stop
   ```

### Backtest Takes Too Long

**Solutions:**
1. Use smaller validation dataset
2. Run on CPU with `gpu_id=-1` (faster for small models)
3. Reduce retry count in config

### Models Stuck at Paper Trading Stage

**Check:**
1. Model Arena is running
2. Trials have been evaluated for 7+ days
3. Performance grades are A or B

## Getting Help

**Check Logs:**
```bash
tail -100 logs/pipeline_orchestrator.log
tail -50 logs/pipeline_notifications.log
```

**Check State:**
```bash
cat deployments/pipeline_state.json | jq '.trials'
```

**Verify Configuration:**
```bash
python test_pipeline.py
```

## Known Limitations

1. **Old Trials**: Most trials have weights cleaned up - pipeline can only process trials with saved weights
2. **Validation Data**: Requires proper validation dataset in `data/1h_1680/val/`
3. **CGE Data**: CGE stress testing requires synthetic scenario data
4. **Single Trial**: Pipeline processes one trial at a time through paper trading

## Success Criteria

A properly working pipeline should:
- ✅ Find trials with saved weights
- ✅ Run backtests successfully
- ✅ Pass/fail trials based on criteria
- ✅ Update pipeline state correctly
- ✅ Send desktop notifications
- ✅ Log all actions

If you see these, the pipeline is working!
