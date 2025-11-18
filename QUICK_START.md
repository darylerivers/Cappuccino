# Quick Start Guide

## Start Training (One Command)

```bash
cd /home/mrc/experiment/cappuccino
python -u 1_optimize_unified.py --n-trials 100 --gpu 0 --study-name production_run
```

## Monitor GPU

```bash
watch -n 2 nvidia-smi
```

Expected: **60-90% GPU utilization**

## Check Progress

```bash
# Count completed trials
sqlite3 databases/optuna_cappuccino.db "SELECT COUNT(*) FROM trials WHERE state='COMPLETE'"

# View best trial
sqlite3 databases/optuna_cappuccino.db "SELECT number, value FROM trials WHERE state='COMPLETE' ORDER BY value DESC LIMIT 1"
```

## Important Notes

- ✅ Always use `python -u` for unbuffered output
- ✅ Training will run in background
- ✅ Each trial takes ~10-20 minutes (6 splits)
- ✅ 100 trials = ~20-30 hours total
- ✅ Safe to Ctrl+C and resume (Optuna persists state)

## Resume Training

If interrupted, just run the same command again:
```bash
python -u 1_optimize_unified.py --n-trials 100 --gpu 0 --study-name production_run
```

Optuna automatically continues from where it left off.

## View Results

```python
import optuna

study = optuna.load_study(
    study_name="production_run",
    storage="sqlite:///databases/optuna_cappuccino.db"
)

print(f"Best value: {study.best_value}")
print(f"Best params: {study.best_params}")
print(f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
```

## Troubleshooting

**Problem:** Training seems slow
**Solution:** Check GPU utilization with `nvidia-smi`. Should be 60-90%.

**Problem:** No output visible
**Solution:** Make sure to use `python -u` flag for unbuffered output.

**Problem:** Out of memory
**Solution:** Reduce `--n-trials` or lower batch size in code.

## Next: Analyze Results

After training completes:
```bash
python analyze_training.py --study-name production_run
```

This will show you the best hyperparameters and generate visualizations.
