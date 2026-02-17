# Two-Phase Training System

**A systematic approach to optimizing cryptocurrency trading agents with time-frame constraints and progressive fee modeling.**

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Phase 1: Time-Frame Optimization](#phase-1-time-frame-optimization)
7. [Phase 2: Feature-Enhanced Training](#phase-2-feature-enhanced-training)
8. [Command Reference](#command-reference)
9. [Understanding Results](#understanding-results)
10. [Troubleshooting](#troubleshooting)
11. [Advanced Usage](#advanced-usage)
12. [File Structure](#file-structure)

---

## Overview

The two-phase training system addresses key limitations in traditional cryptocurrency trading agent optimization:

### Problems Solved

1. **Time-Frame Ambiguity**: Models could hold positions indefinitely without constraint
2. **Fee Mismatch**: Training used static 0.25% fees vs. realistic 0.6%+ fees for new traders
3. **Missing Temporal Context**: Short-term lookback missed longer-term trends
4. **Large Search Space**: Optimizing 26 hyperparameters across multiple time-frames was computationally prohibitive

### Solution: Two-Phase Approach

**Phase 1: Time-Frame Optimization** (500 trials)
- Systematically tests 25 time-frame/interval combinations
- Uses simplified hyperparameter search (10 parameters)
- Enforces time-frame constraints that force liquidation at deadline
- **Output**: Winning time-frame and interval combination

**Phase 2: Feature-Enhanced Training** (400 trials)
- Uses winning parameters from Phase 1
- Adds 7-day and 30-day rolling means (91 state dimensions)
- Implements progressive Coinbase fee tiers (0.6% → 0.25%)
- Compares PPO vs. DDQN algorithms
- **Output**: Production-ready trading agent

**Total**: 900 trials yielding an optimized, realistic trading strategy

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Two-Phase Training System                     │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────────┐         ┌──────────────────────────┐
│      Phase 1 Input       │         │      Phase 2 Input       │
├──────────────────────────┤         ├──────────────────────────┤
│ • 12 months of 1h data   │         │ • Phase 1 winner params  │
│ • 5 time-frames          │    ┌───>│ • 12 months enhanced     │
│ • 5 intervals            │    │    │ • 7-day rolling means    │
│ • 25 combinations        │    │    │ • 30-day rolling means   │
│ • 20 trials per combo    │    │    │ • Progressive fee tiers  │
└──────────────┬───────────┘    │    └──────────┬───────────────┘
               │                │               │
               ▼                │               ▼
┌──────────────────────────┐   │    ┌──────────────────────────┐
│   Phase 1 Optimization   │   │    │   Phase 2 Optimization   │
├──────────────────────────┤   │    ├──────────────────────────┤
│ • Simplified hyperparam  │   │    │ • Full hyperparameter    │
│   search (10 params)     │   │    │   search (26 params)     │
│ • Time-frame constraints │   │    │ • Dynamic fee tiers      │
│ • PPO only               │   │    │ • PPO + DDQN comparison  │
│ • CPCV evaluation        │   │    │ • Enhanced state space   │
│ • 500 total trials       │   │    │ • 400 total trials       │
└──────────────┬───────────┘   │    └──────────┬───────────────┘
               │                │               │
               ▼                │               ▼
┌──────────────────────────┐   │    ┌──────────────────────────┐
│      Phase 1 Output      │───┘    │      Phase 2 Output      │
├──────────────────────────┤        ├──────────────────────────┤
│ • Winning time-frame     │        │ • PPO best model         │
│ • Winning interval       │        │ • DDQN best model        │
│ • Best Sharpe ratio      │        │ • Algorithm comparison   │
│ • phase1_winner.json     │        │ • Production-ready agent │
└──────────────────────────┘        └──────────────────────────┘
```

### Core Components

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| `config_two_phase.py` | Central configuration | Time-frame definitions, fee tiers, hyperparameter ranges |
| `timeframe_constraint.py` | Enforces trading deadlines | Calculates max candles, forces liquidation |
| `fee_tier_manager.py` | Progressive fee modeling | 30-day volume tracking, 0.6% → 0.25% progression |
| `environment_Alpaca_phase2.py` | Enhanced trading environment | 91-dim state space with rolling means |
| `phase1_timeframe_optimizer.py` | Phase 1 orchestrator | Tests 25 time-frame combinations |
| `phase2_feature_maximizer.py` | Phase 2 orchestrator | PPO/DDQN comparison with full features |
| `agent_ddqn.py` | DDQN implementation | Discrete action space Q-learning |
| `run_two_phase_training.py` | Master orchestrator | Runs both phases sequentially |

---

## Prerequisites

### System Requirements

- **GPU**: CUDA-compatible GPU recommended (training will use CPU if unavailable)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 10GB free space for data and model checkpoints
- **OS**: Linux (tested on Arch Linux), macOS, or Windows with WSL2

### Software Requirements

```bash
# Python 3.8+
python --version

# Required packages
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install torch>=1.10.0
pip install optuna>=3.0.0
pip install alpaca-trade-api>=2.3.0
pip install elegantrl>=0.3.5
pip install scikit-learn>=1.0.0
pip install matplotlib>=3.4.0
```

### Data Requirements

```bash
# Ensure data directory exists with required files
ls data/

# Required files (generated by prepare_multi_timeframe_data.py):
# - price_array_1h_12mo.npy
# - tech_array_1h_12mo.npy
# - time_array_1h_12mo.npy

# Phase 2 enhanced data (optional, generated by prepare_phase2_data.py):
# - price_array_1h_12mo.npy
# - tech_array_enhanced_1h_12mo.npy (with rolling means)
# - time_array_1h_12mo.npy
```

---

## Installation

### 1. Clone Repository

```bash
cd /home/mrc/experiment/cappuccino
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Data

```bash
# Download and prepare 12 months of data
python prepare_multi_timeframe_data.py --months 12 --interval 1h

# Generate Phase 2 enhanced data with rolling means
python prepare_phase2_data.py --interval 1h --months 12
```

### 4. Verify Installation

```bash
# Run prerequisite check
python run_two_phase_training.py --skip-prerequisites

# Or test components individually
python -c "from config_two_phase import PHASE1, PHASE2; print('Config OK')"
python -c "from timeframe_constraint import TimeFrameConstraint; print('Constraint OK')"
python -c "from fee_tier_manager import FeeTierManager; print('Fee Manager OK')"
```

---

## Quick Start

### Mini Test (Recommended for First Run)

Test the entire pipeline with minimal compute time:

```bash
# Run mini test: Phase 1 (10 trials) + Phase 2 (10 trials) = ~30 minutes
python run_two_phase_training.py --mini-test
```

**Expected output:**
```
==================================================================================================
                              TWO-PHASE TRAINING ORCHESTRATOR
==================================================================================================

Checking Prerequisites
----------------------
  ✓ numpy installed
  ✓ optuna installed
  ✓ torch installed
  ...

==================================================================================================
                              PHASE 1: TIME-FRAME OPTIMIZATION
==================================================================================================

▶ Running Phase 1 Optimization
  Command: python phase1_timeframe_optimizer.py --mini-test
✓ Success

Phase 1 Results
---------------
  Time-frame: 7d
  Interval: 1h
  Best value: 0.3245
  Best Sharpe (bot): 1.4521
  ...
```

### Full Production Run

Run the complete 900-trial optimization:

```bash
# Full run: Phase 1 (500 trials) + Phase 2 (400 trials) = ~48-72 hours
python run_two_phase_training.py
```

### Individual Phases

```bash
# Run only Phase 1 (500 trials)
python run_two_phase_training.py --phase1-only

# Run only Phase 2 (400 trials, requires Phase 1 winner)
python run_two_phase_training.py --phase2-only
```

---

## Phase 1: Time-Frame Optimization

### Objective

Find the optimal time-frame and interval combination by testing 25 configurations.

### Time-Frame Combinations

| Time-Frame | Intervals Tested | Description |
|------------|------------------|-------------|
| 3d | 5m, 15m, 30m, 1h, 4h | Short-term scalping (3 days max hold) |
| 5d | 5m, 15m, 30m, 1h, 4h | Medium-short swing (5 days max hold) |
| 7d | 5m, 15m, 30m, 1h, 4h | Weekly strategy (7 days max hold) |
| 10d | 5m, 15m, 30m, 1h, 4h | Extended swing (10 days max hold) |
| 14d | 5m, 15m, 30m, 1h, 4h | Bi-weekly position (14 days max hold) |

**Total**: 5 time-frames × 5 intervals = 25 combinations × 20 trials = **500 trials**

### Simplified Hyperparameter Search

Phase 1 optimizes only 10 critical parameters to focus on time-frame validation:

**ElegantRL Parameters (4)**:
- `learning_rate`: 1e-5 to 1e-4 (log scale)
- `batch_size`: {256, 512, 1024, 2048}
- `gamma`: 0.95 to 0.999 (step 0.005)
- `net_dimension`: 512 to 2048 (step 128)

**Environment Parameters (6)**:
- `lookback`: 1 to 10 candles
- `trailing_stop_pct`: 0.03 to 0.15 (3% to 15%)
- Plus 4 fixed normalization parameters

**Fixed Parameters**:
- `target_step`: 256
- `break_step`: 50,000
- `worker_num`: 2
- `thread_num`: 4

### Time-Frame Constraint Enforcement

Each trial enforces a deadline based on the time-frame:

```python
# Example: 7d time-frame with 1h interval
max_candles = 7 days × 24 hours = 168 candles
deadline = lookback + max_candles

# If agent holds position past deadline:
# → Force liquidation at current prices
# → Cap reward at boundary
# → Episode ends
```

### Running Phase 1 Standalone

```bash
# Full Phase 1 (500 trials)
python phase1_timeframe_optimizer.py

# Mini test (10 trials: 2 combos × 5 trials)
python phase1_timeframe_optimizer.py --mini-test

# Custom trial count
python phase1_timeframe_optimizer.py --trials-per-combo 10

# Custom CPCV configuration
python phase1_timeframe_optimizer.py --num-paths 5 --k-test-groups 3
```

### Phase 1 Outputs

```bash
# Winner configuration
phase1_winner.json

# Full results for all combinations
phase1_all_results.json

# Checkpoint (for resume)
phase1_checkpoint.json

# Optuna database
databases/phase1_optuna.db

# Training artifacts
train_results/phase1/
  ├── phase1_trial_0_3d_5m/
  ├── phase1_trial_1_3d_15m/
  └── ...
```

**Example `phase1_winner.json`:**
```json
{
  "timeframe": "7d",
  "interval": "1h",
  "best_trial_number": 87,
  "best_value": 0.3245,
  "best_sharpe_bot": 1.4521,
  "best_sharpe_hodl": 1.1892,
  "n_trials": 500,
  "best_params": {
    "learning_rate": 3.24e-05,
    "batch_size": 1024,
    "gamma": 0.985,
    "net_dimension": 1408,
    "lookback": 5,
    "trailing_stop_pct": 0.08
  }
}
```

---

## Phase 2: Feature-Enhanced Training

### Objective

Train production-ready agents using Phase 1's winning time-frame with:
- Full hyperparameter optimization (26 parameters)
- Enhanced state space (91 dimensions with rolling means)
- Progressive Coinbase fee tiers
- PPO vs. DDQN algorithm comparison

### Enhanced State Space

**Base Features (63 dimensions)**:
- Close prices (7 assets)
- MACD, MACD signal, MACD histogram (7 × 3)
- RSI (7 assets)
- CCI (7 assets)
- ADX/DX (7 assets)
- Holdings (7 assets)
- Cash position
- Account value

**Rolling Mean Features (28 dimensions)**:
- 7-day rolling mean of close prices (7 assets)
- 30-day rolling mean of close prices (7 assets)
- 7-day rolling mean of volume (7 assets)
- 30-day rolling mean of volume (7 assets)

**Total**: 63 + 28 = **91 state dimensions**

### Progressive Fee Tiers

Simulates realistic Coinbase fee progression based on 30-day rolling volume:

| Tier | 30-Day Volume | Maker Fee | Taker Fee |
|------|---------------|-----------|-----------|
| Default | $0 - $10K | 0.60% | 1.20% |
| Tier 1 | $10K - $25K | 0.40% | 0.80% |
| Tier 2 | $25K - $50K | 0.25% | 0.50% |
| Tier 3 | $50K - $100K | 0.15% | 0.35% |
| VIP | $100K+ | 0.10% | 0.25% |

**Key Features**:
- 30-day rolling window (variable length depending on interval)
- Progressive tier upgrades as volume increases
- Realistic simulation of new trader → experienced trader progression

### Full Hyperparameter Search

Phase 2 optimizes all 26 parameters:

**ElegantRL Parameters (15)**:
- `learning_rate`, `batch_size`, `gamma`, `net_dimension`
- `target_step`, `repeat_times`, `reward_scale`
- `clip_range`, `entropy_coef`, `value_loss_coef`
- `max_grad_norm`, `gae_lambda`, `ppo_epochs`
- `kl_target`, `adam_epsilon`

**Environment Parameters (11)**:
- `lookback`, `norm_cash`, `norm_stocks`, `norm_tech`
- `norm_reward`, `norm_action`, `time_decay_floor`
- `min_cash_reserve`, `concentration_penalty`
- `trailing_stop_pct`, fee-related parameters

### Algorithm Comparison: PPO vs. DDQN

**PPO (Proximal Policy Optimization)**:
- Continuous action space
- Well-tested with 11,160+ trials
- Best for smooth trading decisions

**DDQN (Double Deep Q-Network)**:
- Discrete action space (70 actions: 7 assets × 10 bins)
- Sequential action selection
- Alternative approach for comparison

**Trials**: 200 PPO + 200 DDQN = **400 total trials**

### Running Phase 2 Standalone

```bash
# Full Phase 2 (400 trials: 200 PPO + 200 DDQN)
python phase2_feature_maximizer.py

# Mini test (10 trials: 5 PPO + 5 DDQN)
python phase2_feature_maximizer.py --mini-test

# PPO only
python phase2_feature_maximizer.py --algorithm ppo --trials-ppo 200

# DDQN only
python phase2_feature_maximizer.py --algorithm ddqn --trials-ddqn 200

# Custom data configuration
python phase2_feature_maximizer.py --data-dir data/phase2 --months 12
```

### Phase 2 Outputs

```bash
# Best PPO model
phase2_ppo_best.json

# Best DDQN model
phase2_ddqn_best.json

# Algorithm comparison
phase2_comparison.json

# Optuna databases
databases/phase2_ppo_optuna.db
databases/phase2_ddqn_optuna.db

# Training artifacts
train_results/phase2_ppo/
  ├── phase2_ppo_trial_0/
  ├── phase2_ppo_trial_1/
  └── ...

train_results/phase2_ddqn/
  ├── phase2_ddqn_trial_0/
  └── ...
```

**Example `phase2_comparison.json`:**
```json
{
  "winner": "ppo",
  "results": {
    "ppo": {
      "algorithm": "ppo",
      "best_trial_number": 142,
      "best_value": 0.4812,
      "best_sharpe_bot": 1.7234,
      "best_sharpe_hodl": 1.2145,
      "n_trials": 200
    },
    "ddqn": {
      "algorithm": "ddqn",
      "best_trial_number": 78,
      "best_value": 0.3621,
      "best_sharpe_bot": 1.5431,
      "best_sharpe_hodl": 1.2034,
      "n_trials": 200
    }
  },
  "phase1_timeframe": "7d",
  "phase1_interval": "1h",
  "timestamp": "2025-12-16T10:30:45.123456"
}
```

---

## Command Reference

### Master Orchestrator

```bash
python run_two_phase_training.py [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--mini-test` | - | Run mini test (20 trials total) |
| `--phase1-only` | - | Run only Phase 1 |
| `--phase2-only` | - | Run only Phase 2 (requires Phase 1 winner) |
| `--resume FILE` | - | Resume from checkpoint file |
| `--skip-prerequisites` | - | Skip prerequisite checks |
| `--data-dir DIR` | `data` | Data directory |
| `--months N` | `12` | Data months for Phase 2 |
| `--phase1-trials N` | `20` | Trials per combination in Phase 1 |
| `--algorithm ALG` | `both` | Algorithm for Phase 2 (`ppo`, `ddqn`, `both`) |
| `--phase2-ppo-trials N` | `200` | PPO trials in Phase 2 |
| `--phase2-ddqn-trials N` | `200` | DDQN trials in Phase 2 |
| `--num-paths N` | `3` | CPCV paths |
| `--k-test-groups N` | `2` | CPCV test groups |

**Examples:**

```bash
# Full run
python run_two_phase_training.py

# Mini test
python run_two_phase_training.py --mini-test

# Only Phase 1
python run_two_phase_training.py --phase1-only

# Only Phase 2 with PPO
python run_two_phase_training.py --phase2-only --algorithm ppo

# Resume from checkpoint
python run_two_phase_training.py --resume two_phase_checkpoint.json

# Custom configuration
python run_two_phase_training.py \
  --phase1-trials 10 \
  --phase2-ppo-trials 100 \
  --phase2-ddqn-trials 100 \
  --num-paths 5
```

### Phase 1 Standalone

```bash
python phase1_timeframe_optimizer.py [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir DIR` | `data` | Data directory |
| `--trials-per-combo N` | `20` | Trials per combination |
| `--mini-test` | - | Run mini test (2 combos × 5 trials) |
| `--num-paths N` | `3` | CPCV paths |
| `--k-test-groups N` | `2` | CPCV test groups |

### Phase 2 Standalone

```bash
python phase2_feature_maximizer.py [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir DIR` | `data/phase2` | Phase 2 data directory |
| `--algorithm ALG` | `both` | Algorithm (`ppo`, `ddqn`, `both`) |
| `--trials-ppo N` | `200` | PPO trials |
| `--trials-ddqn N` | `200` | DDQN trials |
| `--mini-test` | - | Run mini test (5 PPO + 5 DDQN) |
| `--months N` | `12` | Data months |
| `--num-paths N` | `3` | CPCV paths |
| `--k-test-groups N` | `2` | CPCV test groups |

---

## Understanding Results

### Metrics Explained

**Sharpe Ratio (Bot)**:
- Risk-adjusted return of the trading agent
- Higher is better
- **Good**: > 1.0, **Excellent**: > 2.0

**Sharpe Ratio (HODL)**:
- Risk-adjusted return of equal-weight buy-and-hold
- Baseline for comparison

**Objective Value**:
- Composite metric: `mean_sharpe_bot - mean_sharpe_hodl - 0.1 × std_sharpe_bot`
- Rewards outperformance vs. HODL
- Penalizes inconsistency across CPCV splits

**Best Value**:
- Maximum objective value achieved
- Higher indicates better risk-adjusted outperformance

### Reading Phase 1 Results

```json
{
  "timeframe": "7d",          // Winner: 7-day time-frame
  "interval": "1h",            // Winner: 1-hour intervals
  "best_value": 0.3245,        // Objective value
  "best_sharpe_bot": 1.4521,   // Agent achieved 1.45 Sharpe
  "best_sharpe_hodl": 1.1892,  // HODL baseline: 1.19 Sharpe
  "n_trials": 500              // 500 trials tested
}
```

**Interpretation**:
- Agent outperformed HODL by ~0.26 in Sharpe ratio
- 7-day time-frame with 1-hour data yielded best results
- This configuration will be used in Phase 2

### Reading Phase 2 Results

```json
{
  "winner": "ppo",
  "results": {
    "ppo": {
      "best_value": 0.4812,        // PPO objective value
      "best_sharpe_bot": 1.7234,   // PPO Sharpe ratio
      "best_sharpe_hodl": 1.2145   // HODL baseline
    },
    "ddqn": {
      "best_value": 0.3621,        // DDQN objective value
      "best_sharpe_bot": 1.5431    // DDQN Sharpe ratio
    }
  }
}
```

**Interpretation**:
- PPO outperformed DDQN (0.48 vs. 0.36 objective value)
- PPO achieved 1.72 Sharpe ratio (excellent)
- PPO model recommended for production deployment

### Comparing Phases

| Metric | Phase 1 | Phase 2 | Improvement |
|--------|---------|---------|-------------|
| Sharpe (Bot) | 1.4521 | 1.7234 | +18.7% |
| Sharpe vs. HODL | +0.26 | +0.51 | +96% |
| State Dimensions | 63 | 91 | +44% |
| Hyperparameters | 10 | 26 | +160% |
| Trials | 500 | 400 | - |

**Key Insight**: Phase 2's enhanced features and full hyperparameter search significantly improved performance.

---

## Troubleshooting

### Common Issues

#### 1. Phase 1 Winner File Not Found

**Error:**
```
Phase 1 winner file not found: phase1_winner.json
```

**Solution:**
```bash
# Run Phase 1 first
python phase1_timeframe_optimizer.py --mini-test

# Or check if file exists
ls -lh phase1_winner.json

# Resume from checkpoint if interrupted
python run_two_phase_training.py --resume two_phase_checkpoint.json
```

#### 2. Data Files Missing

**Error:**
```
FileNotFoundError: data/price_array_1h_12mo.npy
```

**Solution:**
```bash
# Generate data files
python prepare_multi_timeframe_data.py --months 12 --interval 1h

# For Phase 2 enhanced data
python prepare_phase2_data.py --interval 1h --months 12

# Verify files exist
ls -lh data/*.npy
```

#### 3. GPU Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```python
# Reduce batch size in config_two_phase.py
PHASE1.BATCH_SIZES = (256, 512)  # Instead of (256, 512, 1024, 2048)

# Or reduce net_dimension
PHASE1.NET_DIM_MAX = 1024  # Instead of 2048

# Or use CPU (slower)
export CUDA_VISIBLE_DEVICES=""
```

#### 4. Optuna Database Locked

**Error:**
```
sqlite3.OperationalError: database is locked
```

**Solution:**
```bash
# Kill any hanging processes
pkill -f phase1_timeframe_optimizer
pkill -f phase2_feature_maximizer

# Check for lock files
find databases/ -name "*.db-wal" -o -name "*.db-shm"

# Remove lock files if safe
rm databases/*.db-wal databases/*.db-shm
```

#### 5. Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'config_two_phase'
```

**Solution:**
```bash
# Ensure you're in the correct directory
cd /home/mrc/experiment/cappuccino

# Verify file exists
ls -lh config_two_phase.py

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Add current directory to PYTHONPATH if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Performance Issues

#### Slow Training

**Problem**: Training is taking too long

**Solutions:**
1. **Use mini-test mode** for validation: `--mini-test`
2. **Reduce trials**: `--phase1-trials 10 --phase2-ppo-trials 50`
3. **Use fewer CPCV paths**: `--num-paths 2`
4. **Enable GPU**: Ensure CUDA is available
5. **Reduce break_step** in config (careful: may undertrain)

#### High Memory Usage

**Problem**: System running out of RAM

**Solutions:**
1. **Reduce lookback window** in config
2. **Reduce batch size** options
3. **Use smaller net_dimension** ranges
4. **Close other applications**
5. **Monitor with**: `watch -n 1 free -h`

### Validation Checks

```bash
# Test individual components
python timeframe_constraint.py        # Should print test results
python fee_tier_manager.py            # Should print test results
python agent_ddqn.py                  # Should test DDQN agent

# Test data pipeline
python prepare_phase2_data.py --interval 1h --months 1 --test

# Dry run Phase 1
python phase1_timeframe_optimizer.py --mini-test

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'Device: {torch.cuda.get_device_name(0)}')"
```

---

## Advanced Usage

### Custom Time-Frame Combinations

Edit `config_two_phase.py`:

```python
@dataclass(frozen=True)
class Phase1Config:
    # Custom time-frames
    TIME_FRAMES: Tuple[str, ...] = ('1d', '3d', '5d')  # Only 3 time-frames
    INTERVALS: Tuple[str, ...] = ('15m', '1h')         # Only 2 intervals

    # Result: 3 × 2 = 6 combinations instead of 25
```

### Custom Fee Tiers

Edit `fee_tier_manager.py`:

```python
class FeeTierManager:
    # Custom fee structure
    FEE_TIERS = [
        {'volume_threshold': 0, 'maker': 0.005, 'taker': 0.010},      # 0.5% / 1.0%
        {'volume_threshold': 5000, 'maker': 0.003, 'taker': 0.006},   # 0.3% / 0.6%
        {'volume_threshold': 15000, 'maker': 0.002, 'taker': 0.004},  # 0.2% / 0.4%
    ]
```

### Parallel Optimization

Run multiple Phase 1 combinations in parallel:

```bash
# Terminal 1: Time-frames 3d, 5d
python phase1_timeframe_optimizer.py --custom-timeframes 3d 5d &

# Terminal 2: Time-frames 7d, 10d
python phase1_timeframe_optimizer.py --custom-timeframes 7d 10d &

# Terminal 3: Time-frame 14d
python phase1_timeframe_optimizer.py --custom-timeframes 14d &
```

### Integration with Existing System

Deploy Phase 2 winner to live trading:

```bash
# 1. Find best model
cat phase2_comparison.json

# 2. Copy to deployment directory
cp train_results/phase2_ppo/phase2_ppo_trial_142/actor.pth deployments/

# 3. Deploy to paper trading
python auto_model_deployer.py --model deployments/actor.pth

# 4. Start automation
./start_automation.sh

# 5. Monitor performance
python dashboard.py
```

---

## File Structure

```
cappuccino/
├── run_two_phase_training.py          # Master orchestrator
├── phase1_timeframe_optimizer.py      # Phase 1 orchestrator
├── phase2_feature_maximizer.py        # Phase 2 orchestrator
├── config_two_phase.py                # Central configuration
├── timeframe_constraint.py            # Time-frame enforcement
├── fee_tier_manager.py                # Progressive fee modeling
├── environment_Alpaca.py              # Base environment
├── environment_Alpaca_phase2.py       # Enhanced Phase 2 environment
├── agent_ddqn.py                      # DDQN implementation
├── function_train_test.py             # Training/testing utilities
├── function_CPCV.py                   # Cross-validation setup
├── prepare_multi_timeframe_data.py    # Data preparation
├── prepare_phase2_data.py             # Phase 2 data with rolling means
│
├── data/                              # Data storage
│   ├── price_array_1h_12mo.npy
│   ├── tech_array_1h_12mo.npy
│   ├── tech_array_enhanced_1h_12mo.npy
│   └── time_array_1h_12mo.npy
│
├── databases/                         # Optuna databases
│   ├── phase1_optuna.db
│   ├── phase2_ppo_optuna.db
│   └── phase2_ddqn_optuna.db
│
├── train_results/                     # Training artifacts
│   ├── phase1/
│   │   └── phase1_trial_N_TF_INT/
│   ├── phase2_ppo/
│   │   └── phase2_ppo_trial_N/
│   └── phase2_ddqn/
│       └── phase2_ddqn_trial_N/
│
├── phase1_winner.json                 # Phase 1 winner
├── phase1_all_results.json            # Phase 1 all results
├── phase1_checkpoint.json             # Phase 1 checkpoint
├── phase2_ppo_best.json               # Phase 2 PPO winner
├── phase2_ddqn_best.json              # Phase 2 DDQN winner
├── phase2_comparison.json             # Phase 2 comparison
├── two_phase_checkpoint.json          # Master checkpoint
├── two_phase_training_report.json     # Final report
│
└── README_TWO_PHASE_TRAINING.md       # This file
```

---

## Summary

The two-phase training system provides a systematic, realistic approach to optimizing cryptocurrency trading agents:

**Phase 1** identifies the optimal time-frame and interval through exhaustive search with simplified hyperparameters.

**Phase 2** maximizes performance using Phase 1's winner, enhanced features, progressive fees, and full hyperparameter optimization.

**Result**: A production-ready trading agent optimized for realistic market conditions with proper time-frame constraints and fee modeling.

### Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                  Two-Phase Training Quick Reference          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  MINI TEST:   python run_two_phase_training.py --mini-test  │
│  Duration:    ~30 minutes                                    │
│  Trials:      20 (10 Phase 1 + 10 Phase 2)                  │
│                                                              │
│  FULL RUN:    python run_two_phase_training.py              │
│  Duration:    ~48-72 hours                                   │
│  Trials:      900 (500 Phase 1 + 400 Phase 2)               │
│                                                              │
│  RESUME:      python run_two_phase_training.py --resume     │
│               two_phase_checkpoint.json                      │
│                                                              │
│  PHASE 1 ONLY: --phase1-only                                │
│  PHASE 2 ONLY: --phase2-only                                │
│                                                              │
│  OUTPUTS:                                                    │
│    • phase1_winner.json                                      │
│    • phase2_comparison.json                                  │
│    • two_phase_training_report.json                          │
│    • train_results/phase2_*/best_model/                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

**Need Help?**

- Review the [Troubleshooting](#troubleshooting) section
- Check existing GitHub issues
- Run `python run_two_phase_training.py --help`
- Test individual components with mini-test mode

**Ready to Begin?**

```bash
# Start with mini test
python run_two_phase_training.py --mini-test

# Then run full optimization
python run_two_phase_training.py
```

Happy training! 🚀
