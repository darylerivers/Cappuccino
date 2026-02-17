# URGENT: Optimize DRL Training to Complete in 2-3 Days Instead of 11 Days

## Problem Statement

Our cryptocurrency DRL trading model training is taking **11 days** to complete 1000 trials. This is problematic because:
- By the time training completes, the model will be **outdated by over a week** of market data
- Crypto markets move fast - a week-old model has significantly reduced predictive value
- We need to iterate faster to stay competitive

**Current Stats:**
- Average trial duration: **14.7 minutes**
- Target trials: **1000**
- Estimated completion: **10.2 days**
- Current best Sharpe: **0.1720** (Trial #14, after 12 trials)
- GPU: AMD RX 7900 GRE (16GB VRAM) at 100% utilization
- Training script: `scripts/training/1_optimize_unified.py`

## Current Setup

**Hardware:**
- GPU: AMD RX 7900 GRE (16GB VRAM, 100% utilized, 124W/244W)
- CPU: Multi-core (enough for parallel workers)
- RAM: 31GB total (19GB used by current training)

**Training Configuration:**
```bash
python scripts/training/1_optimize_unified.py \
  --study-name cappuccino_ft_16gb_optimized \
  --n-trials 1000 \
  --force-ft \
  --timeframe 1h \
  --data-dir data/1h_1680
```

**Key Details:**
- Using FT-Transformer architecture (--force-ft)
- 1-hour timeframe data
- Optuna hyperparameter optimization
- NO pruner currently enabled (trials run to completion)
- NO parallel workers (single trial at a time)
- NO early stopping based on performance
- Database: `databases/optuna_cappuccino.db`

## Your Task

Analyze the training code and provide **specific, actionable modifications** to reduce training time from **11 days to 2-3 days** while maintaining model quality.

## Files to Examine

1. **Main training script:**
   - `/opt/user-data/experiment/cappuccino/scripts/training/1_optimize_unified.py`
   - Contains objective function, hyperparameter ranges, and study creation

2. **Training utilities:**
   - `/opt/user-data/experiment/cappuccino/utils/function_train_test.py`
   - Contains the actual PPO training loop

3. **Environment:**
   - `/opt/user-data/experiment/cappuccino/environment_Alpaca.py`
   - The DRL trading environment

## Optimization Strategies to Consider

### 1. **Optuna Pruning** (Kill bad trials early)
- Add MedianPruner or HyperbandPruner
- Report intermediate Sharpe ratios during training
- Prune trials that perform worse than median after N episodes
- **Expected speedup: 2-3x** (by killing ~60-70% of trials early)

### 2. **Parallel Trial Execution**
- Use `study.optimize(..., n_jobs=2)` for 2 parallel trials
- Check GPU memory - we have 16GB, currently using ~6GB per trial
- **Expected speedup: 1.5-2x** (if GPU memory allows)

### 3. **Reduce Trial Budget Intelligently**
- Use TPE sampler with good priors from best trial (#14)
- Reduce from 1000 to 300-500 trials
- Use `--use-best-ranges` mode to narrow search space
- **Expected speedup: 2-3x** (fewer trials, faster convergence)

### 4. **Reduce Episode Length**
- Current episodes may be too long
- Test with 80% of current episode length
- Validate that shorter episodes still converge
- **Expected speedup: 1.2x**

### 5. **Smart Sampling**
- Use CmaEsSampler or TPESampler with multivariate=True
- Initialize with known good hyperparameters from trial #14
- **Expected speedup: 1.3-1.5x** (faster convergence)

### 6. **Early Stopping Within Episodes**
- Stop training episodes if Sharpe ratio plateaus
- Monitor rolling Sharpe every N steps
- **Expected speedup: 1.2-1.5x**

### 7. **Reduce Validation/Test Overhead**
- Streamline the backtesting phase
- Use fewer test windows if currently using multiple
- **Expected speedup: 1.1-1.2x**

## Output Required

Provide the following:

### 1. **Modified Study Creation Code**
```python
# Show the updated optuna.create_study() with:
# - Pruner configuration
# - Sampler configuration
# - Any other optimizations
```

### 2. **Modified Optimization Call**
```python
# Show the updated study.optimize() with:
# - n_jobs for parallel execution
# - Callbacks for pruning
# - Progress reporting
```

### 3. **Code for Intermediate Reporting**
```python
# Show how to report intermediate values for pruning
# e.g., trial.report(intermediate_sharpe, step=episode_num)
```

### 4. **Recommended Hyperparameters**
- Should we reduce n_trials from 1000? To what?
- Which pruner is best for this use case?
- How many parallel jobs can we safely run?

### 5. **Expected Results**
- Estimated new training duration
- Expected impact on model quality
- Any trade-offs or risks

## Constraints

- **MUST maintain model quality** - don't sacrifice Sharpe ratio significantly
- **MUST use existing GPU** - no hardware changes
- **MUST preserve FT-Transformer architecture** - this is critical
- **PREFER code changes over architectural changes** - optimize process, not model
- **AVOID reducing data quality** - keep full 1h timeframe dataset

## Success Criteria

- Training completes in **2-3 days** instead of 11
- Final best Sharpe ratio >= **0.15** (current best is 0.1720)
- Code changes are minimal and focused
- No GPU OOM errors
- Solution is production-ready (not experimental hacks)

## Example Code Structure

The training script likely has this structure:
```python
def objective(trial):
    # Suggest hyperparameters
    net_dim = trial.suggest_int("net_dim", 256, 2048)
    batch_size = trial.suggest_int("batch_size", 512, 32768)
    # ... more hyperparameters

    # Train the agent
    sharpe_ratio = train_and_test(...)

    return sharpe_ratio

# Current (slow) approach:
study = optuna.create_study(direction='maximize', load_if_exists=True)
study.optimize(objective, n_trials=1000)  # Takes 11 days!
```

## Questions to Answer

1. What's the single biggest bottleneck causing slow training?
2. Can we run 2+ trials in parallel without GPU OOM?
3. Which pruner (Median, Hyperband, Successive Halving) is best for DRL?
4. Should we reduce n_trials? If so, to what number?
5. What's the optimal combination of speedups to achieve 2-3 day completion?

## Additional Context

- We're training crypto trading models that need to stay current
- Model drift is a serious concern - fresher data = better predictions
- We want to retrain weekly, so 2-3 day training time enables continuous improvement
- Current training uses rolling windows for realistic backtesting
- PPO algorithm with FT-Transformer encoder for market state representation

---

**Please provide specific code changes, configuration updates, and your reasoning for each optimization. Focus on practical, production-ready solutions that will work with our existing AMD GPU setup.**
