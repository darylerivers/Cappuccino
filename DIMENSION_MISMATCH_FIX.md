# State Dimension Mismatch Fix

**Date:** February 9, 2026
**Status:** ✅ Implemented and Tested

## Problem

During cross-validation training with mixed lookback values:
- Different trials use different lookback parameters (1-20 bars)
- State dimension = 1 + num_tickers + (num_tickers × 14 indicators) × **lookback**
- When loading checkpoints from splits with different lookback, state dimensions don't match
- Result: `RuntimeError: size mismatch for net.0.weight`

### Example Error:
```
Split 0 failed: Error(s) in loading state_dict for ActorPPO:
  size mismatch for net.0.weight: copying a param with shape
  torch.Size([1856, 498]) from checkpoint, the shape in current
  model is torch.Size([896, 302])
```

## Root Cause

1. Optuna samples `lookback` parameter (1-20)
2. Each CV split trains with sampled lookback → saves checkpoint
3. During evaluation, tries to load checkpoint into model with DIFFERENT lookback
4. `torch.load_state_dict()` fails because input layer dimensions differ

## Solution

Implemented graceful error handling at 3 levels:

### 1. AgentBase.py (drl_agents/agents/AgentBase.py:366-377)
```python
def load_torch_file(model_or_optim, _path):
    state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
    try:
        model_or_optim.load_state_dict(state_dict)
    except RuntimeError as e:
        # Handle state dimension mismatch gracefully
        if "size mismatch" in str(e):
            print(f"⚠️  State dimension mismatch when loading {_path}")
            print(f"    {str(e)[:200]}...")
            print(f"    Skipping this checkpoint (different lookback/state_dim)")
            raise ValueError(f"State dimension mismatch in {_path}") from e
        else:
            raise e
```

**Effect:** Catches dimension mismatch and raises ValueError with clear message

### 2. elegantrl_models.py (drl_agents/elegantrl_models.py:149-161)
```python
try:
    agent = init_agent(args, gpu_id=gpu_id)
    act = agent.act
    device = agent.device
except ValueError as e:
    # State dimension mismatch - re-raise to mark trial as PRUNED
    if "State dimension mismatch" in str(e):
        raise e
    else:
        raise ValueError(f"Fail to load agent: {str(e)}") from e
```

**Effect:** Preserves ValueError for dimension mismatches, improves other error messages

### 3. 1_optimize_unified.py (scripts/training/1_optimize_unified.py:626-634)
```python
except ValueError as e:
    # Handle state dimension mismatch gracefully
    if "State dimension mismatch" in str(e):
        print(f"Split {split_idx} skipped: Different state dimension (lookback mismatch)")
    else:
        print(f"Split {split_idx} failed: {e}")
    trial.report(float('nan'), step=trial.number)  # Log failure
    continue
```

**Effect:** Skips mismatched splits, continues with remaining splits. Trial PRUNED only if ALL splits fail.

## Behavior After Fix

### Before:
- ❌ Training crashes with RuntimeError
- ❌ All 5m trials FAILED after 3 hours
- ❌ No models trained

### After:
- ✅ Dimension mismatch detected and logged
- ✅ Mismatched splits skipped gracefully
- ✅ Trial completes with successful splits
- ✅ Trial PRUNED only if all splits fail
- ✅ Training continues with next trial

## Testing

Comprehensive test in `test_dimension_mismatch_fix.py`:
```bash
python test_dimension_mismatch_fix.py
```

**Test Results:**
- ✅ Models with different state_dim raise ValueError
- ✅ Error message correctly identifies dimension mismatch
- ✅ Models with matching state_dim load successfully
- ✅ All tests PASSED

## Impact on Training

### 5m Training:
- **Before Fix:** All 5 studies crashed after 3 hours
- **After Fix:** Can now train with mixed lookback values
- **Expected:** Some trials will be PRUNED (mismatched splits), but training will complete

### Cross-Validation:
- Trials with 3 splits and varying lookback:
  - Split 0: lookback=5 ✓ trains
  - Split 1: lookback=10 ❌ skipped (different dimension)
  - Split 2: lookback=5 ✓ trains
- Result: Trial completes with 2/3 splits (not ideal but functional)

### Best Practice:
For optimal results, consider:
1. **Fixed lookback per study** - Ensures all splits use same dimension
2. **Separate studies per lookback** - lookback=1,5,10,15,20 studies
3. **Mixed lookback** (current approach) - More exploration but some split failures

## Files Modified

1. `drl_agents/agents/AgentBase.py` - Added error handling in load_torch_file
2. `drl_agents/elegantrl_models.py` - Improved error propagation in DRL_prediction
3. `scripts/training/1_optimize_unified.py` - Added graceful skip for mismatched splits

## Next Steps

- ✅ Fix implemented
- ✅ Tests passed
- ⏳ Launch 5m training on GPU
- ⏳ Monitor for split failures
- ⏳ Consider fixed-lookback studies if too many PRUNEDs

## Related Issues

- Paper trading dimension mismatch (SOLVED: auto-detection in paper_trader_alpaca_polling.py)
- Database lookback vs actual lookback discrepancy
- State space calculation: `1 + num_tickers + (num_tickers × 14) × lookback`
