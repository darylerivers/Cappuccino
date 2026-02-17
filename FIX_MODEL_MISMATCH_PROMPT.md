# Model State Dimension Mismatch - Fix Request for GLM-4.7-Flash

## Problem Summary

Paper trading models in `deployments/model_0`, `deployments/model_1`, and `deployments/trial_250_live` are failing to load with error:

```
ValueError: State dimension mismatch in deployments/model_X/actor.pth
```

## Root Cause Analysis

**Old Models:**
- Trained with environment state_dim = 85
- Located in: `deployments/model_0/actor.pth`, `deployments/model_1/actor.pth`
- Trial numbers: 191, 192

**Current Environment:**
- State dimension = 141-211 (depending on lookback configuration)
- Changed when FT-Transformer was integrated
- Increased lookback window from old configuration

**Why This Happened:**
1. Environment evolved over time (added features, changed lookback)
2. Old models were never retrained
3. Model checkpoints reference old state dimensions

## Project Context

**Location:** `/opt/user-data/experiment/cappuccino`

**Key Files:**
- `environment_Alpaca_phase2.py` - Trading environment (defines state dimension)
- `constants.py` - Configuration including NORMALIZATION.lookback
- `scripts/training/1_optimize_unified.py` - Training script
- `drl_agents/agents/AgentPPO_FT.py` - PPO agent with FT-Transformer
- `deployments/model_*/actor.pth` - Old incompatible models

**Environment:**
- Python 3.11.9 in pyenv virtualenv `cappuccino-rocm`
- PyTorch 2.10.0+rocm7.1
- AMD RX 7900 GRE GPU (16GB VRAM)
- Activate with: `source activate_rocm_env.sh`

## Task: Fix the Model Mismatch

Please provide a solution that does ONE of the following:

### Approach A: Model Conversion (Preferred if possible)

Create a script that:
1. Loads old models (state_dim=85)
2. Determines what the old environment configuration was
3. Creates a compatibility layer that converts old state → new state
4. Saves converted models that work with current environment

**Key Questions to Answer:**
- What was the old lookback value that gave state_dim=85?
- What features were different?
- Can we create a padding/transformation layer?

### Approach B: Clean Retrain (Simpler, but loses old models)

Create a script that:
1. Trains 2-3 new models with current environment
2. Uses best hyperparameters from current successful training
3. Deploys them to replace old incompatible models
4. Provides clear commands to start paper trading

**Use these hyperparameters (from current successful training):**
```python
# From study: cappuccino_ft_16gb_optimized
batch_size = [65536, 98304, 131072]
net_dimension = 2048-4096
learning_rate = 1e-6 to 1e-3
gamma = 0.88-0.99
# Use FT-Transformer (--force-ft flag)
```

### Approach C: Backwards Compatibility

Modify the agent loading code to:
1. Detect state dimension mismatch
2. Automatically adapt old models to new state dimension
3. Pad/truncate state as needed
4. Add compatibility warnings

**File to modify:** `drl_agents/agents/AgentBase.py` line 376

## Expected Deliverables

1. **Analysis Report:** Explain exactly what changed and why models broke
2. **Fix Script:** Runnable bash/python script to fix the issue
3. **Deployment Commands:** Clear steps to deploy and test fixed models
4. **Prevention:** How to avoid this in future

## Testing the Fix

After implementing, test with:

```bash
cd /opt/user-data/experiment/cappuccino
source activate_rocm_env.sh

# Test loading model
python3 << EOF
from scripts.deployment.paper_trader_alpaca_polling import AlpacaPaperTraderPolling
trader = AlpacaPaperTraderPolling(
    model_dir='deployments/model_0',
    tickers=['BTC/USD', 'ETH/USD'],
    poll_interval=60
)
print("✓ Model loaded successfully!")
EOF
```

Should complete without "State dimension mismatch" error.

## Additional Context

**Why This Matters:**
- These models were trained for hours and have good performance
- Retraining costs GPU time and electricity
- Users expect deployed models to keep working
- This same issue will recur unless we add compatibility handling

**Current Workaround:**
- Trial #250 works (unclear why - maybe matches current state_dim?)
- Trial #965 works (recently trained with current environment)
- But we need trials 91 and 100 working for the dashboard

## Files You Can Read/Modify

- `/opt/user-data/experiment/cappuccino/environment_Alpaca_phase2.py`
- `/opt/user-data/experiment/cappuccino/constants.py`
- `/opt/user-data/experiment/cappuccino/drl_agents/agents/AgentBase.py`
- `/opt/user-data/experiment/cappuccino/drl_agents/agents/AgentPPO_FT.py`
- `/opt/user-data/experiment/cappuccino/scripts/training/1_optimize_unified.py`

## Success Criteria

- [ ] Old models load without "State dimension mismatch" error
- [ ] Paper traders (trials 91, 100) run successfully
- [ ] Dashboard shows all traders as active (not overdue)
- [ ] Solution is documented for future reference

## Constraints

- Must use existing environment (`cappuccino-rocm`)
- Cannot break currently working models (trial 250, 965)
- Prefer solution that preserves old trained weights if possible
- If retraining, use GPU efficiently (batch_size 65K+)

Please provide a complete, working solution with clear step-by-step instructions.
