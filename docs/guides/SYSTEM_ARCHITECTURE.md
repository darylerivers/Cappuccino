# Cappuccino Trading System - Complete Architecture

**Purpose**: This document provides a complete understanding of the Cappuccino DRL crypto trading system's architecture, data flow, file organization, and operational procedures.

**Date**: December 18, 2025
**Status**: Authoritative Reference

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Data Flow](#data-flow)
3. [Directory Structure](#directory-structure)
4. [Database Schema](#database-schema)
5. [Model Lifecycle](#model-lifecycle)
6. [Trading Systems](#trading-systems)
7. [Validation Checkpoints](#validation-checkpoints)
8. [Process Flows](#process-flows)
9. [Configuration Files](#configuration-files)
10. [Monitoring and Control](#monitoring-and-control)

---

## System Overview

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Cappuccino System                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
│  │   Training   │ ───> │   Database   │ ───> │  Models   │ │
│  │   (Optuna)   │      │   (SQLite)   │      │   (PTH)   │ │
│  └──────────────┘      └──────────────┘      └───────────┘ │
│         │                      │                     │       │
│         │                      │                     │       │
│         v                      v                     v       │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
│  │ Validation   │      │    Arena     │      │  Ensemble │ │
│  │   Scripts    │      │  (Compete)   │      │   (Vote)  │ │
│  └──────────────┘      └──────────────┘      └───────────┘ │
│         │                      │                     │       │
│         │                      └─────────┬───────────┘       │
│         │                                │                   │
│         v                                v                   │
│  ┌──────────────────────────────────────────────────┐       │
│  │        Paper Trading (Alpaca Polling)            │       │
│  └──────────────────────────────────────────────────┘       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Technologies

- **DRL Framework**: ElegantRL (PPO agent)
- **Optimization**: Optuna (distributed hyperparameter optimization)
- **Database**: SQLite (study/trial metadata)
- **Trading API**: Alpaca Markets (paper trading)
- **Data Source**: Alpaca historical/live data
- **Monitoring**: Custom dashboard (CLI-based)

---

## Data Flow

### 1. Training → Database

```
[Training Worker] ──────> [Optuna Study] ──────> [SQLite Database]
                  Creates                 Stores
                  Trial                   Metadata

Flow:
1. Worker samples hyperparameters from Optuna
2. Worker trains PPO agent with those parameters
3. Worker evaluates on validation data (Sharpe ratio)
4. Worker reports value back to Optuna
5. Optuna stores trial in database:
   - studies table: study metadata
   - trials table: trial state, timestamps
   - trial_values table: Sharpe ratio
   - trial_params table: hyperparameters
   - trial_user_attrs table: custom metrics
```

**File**: `1_optimize_unified.py`
**Database**: `databases/optuna_cappuccino.db`

### 2. Database → Models

```
[Training Worker] ──────> [File System]
                  Saves
                  Model Files

Flow:
1. During training, agent saves checkpoints:
   - actor.pth: Policy network (state → actions)
   - critic.pth: Value network (state → value estimate)

2. At trial completion, saves final model to:
   train_results/cwd_tests/trial_{number}_{timeframe}/

3. OPTIONALLY creates best_trial pickle:
   - Serialized Optuna trial object
   - Contains all metadata (params, value, timestamps)
   - Used for loading without database query
```

**Directory**: `train_results/cwd_tests/`
**File Format**: PyTorch state dict (`.pth`), Python pickle (`.pkl`)

### 3. Models → Validation

```
[Model Files] ──────> [Validation Script] ──────> [Report]
              Checks                       Generates

Flow:
1. Validation script queries database for trial metadata
2. Checks model directory exists
3. Verifies required files:
   - actor.pth (required)
   - critic.pth (optional, not used for inference)
   - best_trial (recommended, can be regenerated)
4. Validates Sharpe values match between:
   - Database trial_values table
   - best_trial pickle file
   - Original training intent
5. Checks for trial number ambiguity across studies
6. Generates validation report
```

**Script**: `validate_models.py`
**Output**: Console report + ModelInfo objects

### 4. Models → Deployment

```
[Validated Models] ──────> [Deployment Directory] ──────> [Trading System]
                   Copies                         Loads

Flow - Arena:
1. setup_arena_clean.py selects top N models
2. Validates each model
3. Copies to deployments/model_{i}/
4. Creates metadata.json with trial info
5. Arena loads each model independently
6. Each model gets its own paper trading portfolio

Flow - Ensemble:
1. Select top N models from study
2. Copy to train_results/ensemble_best/model_{i}/
3. Create ensemble_manifest.json with:
   - trial_numbers: List of trial numbers
   - trial_values: List of Sharpe ratios
   - study_name: Source study (CRITICAL!)
4. Ensemble loads all models
5. On each decision, all models vote
6. Single unified portfolio executes majority decision
```

**Arena Dir**: `deployments/`
**Ensemble Dir**: `train_results/ensemble_best/`

### 5. Trading System → Live Data

```
[Paper Trader] <──────> [Alpaca API] <──────> [Market]
               Polls                  Real-time

Flow:
1. Paper trader polls Alpaca every N seconds
2. Gets latest prices for all tickers
3. Constructs state vector (prices, holdings, indicators)
4. Passes to model(s) for decision
5. Model outputs: position weights for each asset
6. Trader calculates trade orders (buy/sell/hold)
7. Submits orders to Alpaca paper trading
8. Logs all trades and portfolio states
```

**Script**: `paper_trader_alpaca_polling.py`
**Logs**: `logs/paper_trading_{name}.log`, `paper_trades/positions_state.json`

---

## Directory Structure

### Overview

```
cappuccino/
├── 1_optimize_unified.py          # Main training script
├── validate_models.py              # Model validation (NEW)
├── setup_arena_clean.py            # Arena deployment (NEW)
│
├── databases/                      # Optuna study databases
│   ├── optuna_cappuccino.db       # Main database (all studies)
│   ├── phase1_optuna.db           # Two-phase Phase 1
│   └── phase2_optuna.db           # Two-phase Phase 2
│
├── train_results/                  # All training outputs
│   ├── cwd_tests/                 # Individual trial directories
│   │   ├── trial_686_1h/          # Trial 686, 1-hour timeframe
│   │   │   ├── actor.pth          # Policy network weights
│   │   │   ├── critic.pth         # Value network weights (optional)
│   │   │   ├── best_trial         # Optuna trial object (optional)
│   │   │   └── stored_agent/      # Some trials use subdirectory
│   │   │       └── actor.pth
│   │   └── ...
│   │
│   └── ensemble_best/              # Ensemble deployment
│       ├── ensemble_manifest.json # Manifest with trial list
│       ├── model_0/
│       │   ├── actor.pth
│       │   └── best_trial
│       └── ...
│
├── deployments/                    # Arena deployment
│   ├── model_0/
│   │   ├── actor.pth
│   │   ├── best_trial
│   │   └── metadata.json          # Validation info
│   └── ...
│
├── arena_state/                    # Arena runtime state
│   ├── arena_config.json          # Arena configuration
│   └── arena_manifest.json        # Active models
│
├── paper_trades/                   # Paper trading state
│   ├── positions_state.json       # Current positions
│   └── trades_history.json        # Trade log
│
├── logs/                           # All system logs
│   ├── paper_trading_BEST.log     # Ensemble trading log
│   ├── paper_trading_trial191_console.log  # Trial 191 stderr
│   ├── arena.log                  # Arena system log
│   └── training_worker_{i}.log    # Training worker logs
│
├── drl_agents/                     # Agent implementations
│   ├── agents/
│   │   ├── AgentBase.py           # Base agent class
│   │   └── AgentPPO.py            # PPO implementation
│   └── elegantrl_models.py        # Neural network architectures
│
├── environment_Alpaca.py           # Trading environment
├── ultra_simple_ensemble.py        # Ensemble voting system
├── arena_runner.py                 # Arena competition system
├── paper_trader_alpaca_polling.py  # Paper trading engine
├── dashboard.py                    # Monitoring dashboard
│
└── *.sh                            # Control scripts
    ├── start_automation.sh
    ├── stop_automation.sh
    ├── status_automation.sh
    ├── start_arena.sh
    ├── stop_arena.sh
    └── status_arena.sh
```

### Critical Directories

#### `train_results/cwd_tests/`

**Purpose**: Stores all trained models
**Naming**: `trial_{number}_{timeframe}/`
**Files**:
- `actor.pth`: **REQUIRED** - Policy network (46KB - 2MB typical)
- `critic.pth`: Optional - Value network (not used in production)
- `best_trial`: Optional - Pickled Optuna trial object
- `stored_agent/`: Some trials have subdirectory with models

**Note**: Trial numbers are NOT unique across studies!

#### `databases/`

**Purpose**: Optuna study persistence
**Main DB**: `optuna_cappuccino.db` (SQLite)
**Size**: ~50MB (10,000+ trials)

**Schema**:
- `studies`: Study metadata (name, direction=maximize)
- `trials`: Trial records (number, state, timestamps)
- `trial_values`: Objective values (Sharpe ratio)
- `trial_params`: Hyperparameters (JSON stored as text)
- `trial_user_attrs`: Custom metrics (mean_sharpe_bot, etc.)

**CRITICAL**: Trial numbers are unique WITHIN a study, not globally!

#### `deployments/`

**Purpose**: Arena model deployment
**Structure**: One directory per model
**Files**:
- `actor.pth`: Policy network
- `best_trial`: Trial metadata (created if missing)
- `metadata.json`: Validation status

**Lifecycle**: Cleaned and rebuilt on each Arena setup

#### `paper_trades/`

**Purpose**: Paper trading runtime state
**Files**:
- `positions_state.json`: Current portfolio (updated every poll)
- `trades_history.json`: All executed trades

**Format**:
```json
{
  "portfolio_value": 100000.0,
  "cash": 50000.0,
  "positions": [
    {
      "ticker": "BTC/USD",
      "quantity": 0.5,
      "avg_entry": 40000.0,
      "current_price": 42000.0,
      "position_value": 21000.0,
      "pnl_pct": 5.0
    }
  ],
  "timestamp": "2025-12-18T10:30:00"
}
```

---

## Database Schema

### Studies Table

```sql
CREATE TABLE studies (
    study_id INTEGER PRIMARY KEY,
    study_name TEXT UNIQUE NOT NULL,
    direction TEXT NOT NULL  -- 'maximize' or 'minimize'
);
```

**Example Studies**:
- `cappuccino_alpaca_v2`: Best performing (Sharpe 0.14-0.15)
- `cappuccino_week_20251206`: Weekly training (Sharpe 0.01-0.02)
- `cappuccino_3workers_20251102_2325`: Old long training
- `two_phase_phase1`: Multi-timeframe optimization
- `two_phase_phase2`: Feature maximization

### Trials Table

```sql
CREATE TABLE trials (
    trial_id INTEGER PRIMARY KEY,     -- GLOBALLY UNIQUE
    number INTEGER NOT NULL,          -- Unique within study only!
    study_id INTEGER NOT NULL,
    state TEXT NOT NULL,              -- COMPLETE, RUNNING, PRUNED, FAIL
    datetime_start TIMESTAMP,
    datetime_complete TIMESTAMP,
    FOREIGN KEY (study_id) REFERENCES studies(study_id)
);
```

**CRITICAL**: `number` is NOT globally unique! Always use `(study_name, number)` or `trial_id`.

### Trial Values Table

```sql
CREATE TABLE trial_values (
    trial_value_id INTEGER PRIMARY KEY,
    trial_id INTEGER NOT NULL,
    objective INTEGER DEFAULT 0,
    value REAL NOT NULL,              -- Sharpe ratio
    FOREIGN KEY (trial_id) REFERENCES trials(trial_id)
);
```

**Value**: Mean Sharpe ratio across validation episodes (typically 6 episodes).

### Trial Params Table

```sql
CREATE TABLE trial_params (
    param_id INTEGER PRIMARY KEY,
    trial_id INTEGER NOT NULL,
    param_name TEXT NOT NULL,
    param_value TEXT NOT NULL,        -- JSON serialized
    distribution_json TEXT,
    FOREIGN KEY (trial_id) REFERENCES trials(trial_id)
);
```

**Parameters Stored**:
- `learning_rate`: PPO learning rate (1e-5 to 1e-3)
- `batch_size`: Minibatch size (2048, 3072, 4096)
- `gamma`: Discount factor (0.95-0.999)
- `net_dimension`: Hidden layer size (1024-1536)
- `worker_num`, `thread_num`: Parallelization
- `lookback`: Historical window (4-6 periods)
- `clip_range`: PPO clipping (0.1-0.3)
- And ~20 more...

### Trial User Attrs Table

```sql
CREATE TABLE trial_user_attrs (
    user_attr_id INTEGER PRIMARY KEY,
    trial_id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,              -- JSON serialized
    FOREIGN KEY (trial_id) REFERENCES trials(trial_id)
);
```

**Custom Attributes**:
- `mean_sharpe_bot`: Mean Sharpe across episodes
- `mean_sharpe_hodl`: Buy-and-hold baseline Sharpe
- `sharpe_list_bot`: List of Sharpe values per episode
- `sharpe_list_hodl`: Baseline Sharpe per episode
- `std_sharpe_bot`: Standard deviation of Sharpe
- `timeframe`: Training timeframe (1h, 4h, 1d, etc.)

---

## Model Lifecycle

### Phase 1: Training

```
1. Start training worker:
   python 1_optimize_unified.py

2. Worker samples hyperparameters from Optuna

3. Worker creates trial directory:
   train_results/cwd_tests/trial_{number}_{timeframe}/

4. Worker trains PPO agent:
   - Initializes actor/critic networks
   - Trains on historical data
   - Evaluates on validation data
   - Calculates Sharpe ratio

5. Worker saves model:
   - actor.pth: Policy network weights
   - critic.pth: Value network weights (optional)

6. Worker reports Sharpe to Optuna

7. Optuna saves trial to database

8. Worker OPTIONALLY saves best_trial pickle
   (Many old trials don't have this!)
```

**Duration**: 30min - 2 hours per trial (depends on hyperparameters)
**Parallelization**: 3-8 workers typical

### Phase 2: Validation

```
1. Run validation script:
   python validate_models.py --study cappuccino_alpaca_v2 --top-n 10

2. Script queries database:
   - Finds top N trials by Sharpe
   - Gets trial_number, trial_id, Sharpe value

3. For each trial:
   a. Check study exists
   b. Check trial unique (no other studies have same number)
   c. Check model directory exists
   d. Check actor.pth exists
   e. Check best_trial exists (create if missing)
   f. Validate Sharpe values match

4. Output: List of ModelInfo objects with validation status
```

**Script**: `validate_models.py`
**Output**: Validation report + fixable issues identified

### Phase 3: Deployment

#### Arena Deployment

```
1. Run setup script:
   python setup_arena_clean.py --top-n 10

2. Script:
   a. Finds best study (or uses specified)
   b. Gets top N models
   c. Validates all models
   d. Creates deployments/ directory
   e. Copies model files
   f. Creates metadata.json for each
   g. Creates arena_config.json
   h. Starts arena_runner.py

3. Arena loads each model independently:
   - Each gets own portfolio ($10k each)
   - Models compete for best performance
   - Best performers get promoted
```

**Deployment Dir**: `deployments/model_{i}/`
**Config**: `arena_state/arena_config.json`

#### Ensemble Deployment

```
1. Manually select top N models

2. Create ensemble directory:
   train_results/ensemble_best/

3. Copy models:
   model_0/, model_1/, ..., model_19/

4. Create ensemble_manifest.json:
   {
     "model_count": 20,
     "trial_numbers": [686, 687, 521, ...],
     "trial_values": [0.1566, 0.1565, ...],
     "study_name": "cappuccino_alpaca_v2"  ← CRITICAL!
   }

5. Start paper trader:
   python paper_trader_alpaca_polling.py --agent-type ensemble

6. Ensemble loads all models:
   - On each decision, all models vote
   - Majority vote determines action
   - Single portfolio executes trades
```

**Deployment Dir**: `train_results/ensemble_best/`
**Manifest**: `ensemble_manifest.json`

### Phase 4: Live Trading

```
1. Paper trader initializes:
   - Loads model(s) from deployment directory
   - Connects to Alpaca paper trading API
   - Initializes portfolio state

2. Main loop (every 60 seconds):
   a. Poll Alpaca for latest prices
   b. Update portfolio values
   c. Construct state vector (normalized prices, holdings, indicators)
   d. Pass state to model(s)
   e. Model(s) output position weights
   f. Calculate required trades
   g. Submit orders to Alpaca
   h. Log all actions
   i. Update positions_state.json

3. Monitoring:
   - Logs written to logs/paper_trading_{name}.log
   - Dashboard shows live performance
   - Compare to buy-and-hold baseline
```

**Script**: `paper_trader_alpaca_polling.py`
**Log**: `logs/paper_trading_BEST.log`
**State**: `paper_trades/positions_state.json`

---

## Trading Systems

### Ensemble System

**Concept**: Multiple models vote on each decision

**Architecture**:
```
State → Model 1 → Vote 1 ┐
     → Model 2 → Vote 2 ├──> Majority Vote → Action → Portfolio
     → Model 3 → Vote 3 ┘
     ...
     → Model N → Vote N
```

**Advantages**:
- Reduces variance (ensemble effect)
- More robust to individual model failures
- Single portfolio (easier to manage)

**Disadvantages**:
- Slower decisions (N model forwards)
- Can't identify best individual model
- Majority vote may dilute strong signals

**Implementation**: `ultra_simple_ensemble.py`

**CRITICAL BUG (FIXED)**: Must store `study_name` in manifest! Otherwise loads wrong models from wrong study.

### Arena System

**Concept**: Models compete independently

**Architecture**:
```
State → Model 1 → Action 1 → Portfolio 1 → Performance 1 ┐
     → Model 2 → Action 2 → Portfolio 2 → Performance 2 ├──> Rankings
     → Model 3 → Action 3 → Portfolio 3 → Performance 3 ┘
     ...
     → Model N → Action N → Portfolio N → Performance N
```

**Advantages**:
- Identifies best individual models
- Can eliminate poor performers
- Parallel experimentation

**Disadvantages**:
- More complex (N portfolios)
- Higher resource usage
- Need good ranking metric

**Implementation**: `arena_runner.py`

**Ranking**: By Sharpe ratio over rolling window (7 days typical)

**Promotion**: Top 3 models get "promoted" (higher visibility, larger allocation in future)

---

## Validation Checkpoints

### Checkpoint 1: Training Completion

**When**: After trial completes
**Checks**:
- Trial state = COMPLETE in database
- Model files exist in train_results/cwd_tests/
- Sharpe value > threshold (e.g., 0.05)

**Script**:
```bash
sqlite3 databases/optuna_cappuccino.db "
  SELECT t.number, t.state, tv.value
  FROM trials t
  JOIN trial_values tv ON t.trial_id = tv.trial_id
  WHERE t.number = 686
"
```

### Checkpoint 2: Model Files Exist

**When**: Before deployment
**Checks**:
- Directory exists: `train_results/cwd_tests/trial_{number}_{timeframe}/`
- actor.pth exists (either in root or stored_agent/)
- File size reasonable (>10KB, <10MB)

**Script**:
```bash
ls -lh train_results/cwd_tests/trial_686_1h/actor.pth
# OR
ls -lh train_results/cwd_tests/trial_686_1h/stored_agent/actor.pth
```

### Checkpoint 3: Trial Uniqueness

**When**: Before deployment
**Checks**:
- Trial number appears in only ONE study (or specify study)
- No ambiguity about which trial to load

**Script**:
```bash
sqlite3 databases/optuna_cappuccino.db "
  SELECT s.study_name, t.number, t.trial_id, tv.value
  FROM trials t
  JOIN studies s ON t.study_id = s.study_id
  JOIN trial_values tv ON t.trial_id = tv.trial_id
  WHERE t.number = 686
"
# Should return ONE row (or specify study)
```

### Checkpoint 4: best_trial Validation

**When**: Before deployment
**Checks**:
- best_trial pickle file exists (or create from DB)
- Unpickles without error
- Contains expected Sharpe value
- Matches database value

**Script**:
```python
import pickle
with open('train_results/cwd_tests/trial_686_1h/best_trial', 'rb') as f:
    trial = pickle.load(f)
assert trial.number == 686
assert abs(trial.value - 0.1566) < 0.0001
```

### Checkpoint 5: Manifest Validation

**When**: Before ensemble/arena start
**Checks**:
- Manifest JSON valid
- All trial_numbers listed
- study_name specified (CRITICAL!)
- trial_values reasonable (>0, <1 typical)

**Script**:
```python
import json
with open('train_results/ensemble_best/ensemble_manifest.json') as f:
    manifest = json.load(f)
assert 'study_name' in manifest  # CRITICAL!
assert len(manifest['trial_numbers']) == manifest['model_count']
assert all(v > 0 for v in manifest['trial_values'])
```

### Checkpoint 6: Deployment Files

**When**: After deployment, before start
**Checks**:
- All model directories exist (deployments/model_{i}/)
- All have actor.pth
- All have best_trial (or metadata.json)
- Config file valid JSON

**Script**:
```bash
for dir in deployments/model_*; do
    if [ -f "$dir/actor.pth" ]; then
        echo "✓ $dir"
    else
        echo "✗ $dir - MISSING actor.pth"
    fi
done
```

### Checkpoint 7: Live Trading Health

**When**: During paper trading
**Checks**:
- Paper trader process running
- Logs being written (recent timestamps)
- Positions updating (not stale)
- No repeated errors in log
- Performance reasonable (not -50% in 1 hour)

**Script**:
```bash
# Check process
pgrep -f paper_trader_alpaca_polling.py

# Check log freshness
stat -c %Y logs/paper_trading_BEST.log

# Check positions
cat paper_trades/positions_state.json | jq '.timestamp'

# Check for errors
tail -100 logs/paper_trading_BEST.log | grep -i error
```

---

## Process Flows

### Flow 1: Fresh Training Run

```
1. Configure training:
   - Edit .env.training (ACTIVE_STUDY_NAME, TRAINING_WORKERS)
   - Choose timeframe (1h, 4h, 1d)
   - Set number of trials

2. Start training:
   ./start_automation.sh
   # OR manually:
   python 1_optimize_unified.py &

3. Monitor training:
   - Dashboard Page 9: Trial progress
   - tail -f logs/training_worker_0.log
   - ./status_automation.sh

4. Wait for trials to complete:
   - Check database for COMPLETE state
   - Verify models saved to train_results/

5. Stop training:
   ./stop_automation.sh
```

### Flow 2: Validate and Deploy Arena

```
1. Validate models:
   python validate_models.py --study cappuccino_alpaca_v2 --top-n 10

   Review output:
   - All models valid?
   - Any missing best_trial files?
   - Sharpe values reasonable?

2. Setup Arena:
   python setup_arena_clean.py --top-n 10

   Script will:
   - Find best study
   - Get top 10 models
   - Validate all
   - Deploy to deployments/
   - Create config
   - Start arena

3. Monitor Arena:
   - Dashboard Page 3: Arena status
   - tail -f logs/arena.log
   - ./status_arena.sh

4. Check performance:
   - Watch for model rankings
   - Compare to baseline
   - Look for consistent winners

5. Stop Arena:
   ./stop_arena.sh
```

### Flow 3: Deploy Ensemble

```
1. Validate models:
   python validate_models.py --study cappuccino_alpaca_v2 --top-n 20

2. Create ensemble directory:
   mkdir -p train_results/ensemble_best

3. Copy top models:
   for i in {0..19}; do
       cp -r train_results/cwd_tests/trial_${TRIAL_NUM}_1h \
             train_results/ensemble_best/model_$i/
   done

4. Create manifest:
   cat > train_results/ensemble_best/ensemble_manifest.json <<EOF
   {
     "model_count": 20,
     "trial_numbers": [686, 687, 521, ...],
     "trial_values": [0.1566, 0.1565, ...],
     "study_name": "cappuccino_alpaca_v2"
   }
   EOF

5. Start paper trader:
   python paper_trader_alpaca_polling.py --agent-type ensemble

6. Monitor:
   - Dashboard Page 2: Ensemble trading
   - tail -f logs/paper_trading_BEST.log
   - cat paper_trades/positions_state.json
```

### Flow 4: Diagnose Performance Issues

```
1. Check if trader running:
   pgrep -f paper_trader_alpaca_polling.py

2. Check recent logs:
   tail -100 logs/paper_trading_BEST.log

3. Check positions:
   cat paper_trades/positions_state.json | jq '.'

4. Calculate alpha:
   python compare_trading_performance.py

5. Analyze trades:
   python analyze_arena_trades.py
   # Look for:
   # - Overtrading (>5 trades/hour)
   # - Churning (repeated buy/sell same asset)
   # - Large losses (>10% in short time)

6. Validate loaded models:
   python validate_models.py --manifest train_results/ensemble_best/ensemble_manifest.json
   # Check:
   # - Correct study?
   # - Correct Sharpe values?
   # - All files exist?

7. Check for known bugs:
   - Ensemble loading wrong study? (Check manifest has study_name)
   - Trial number ambiguity? (Multiple studies with same trial #)
   - Stale data? (Old ensemble from wrong study)
```

### Flow 5: Two-Phase Training

```
Phase 1: Timeframe Optimization
1. Configure:
   TWO_PHASE_ENABLED=true
   TWO_PHASE_MODE="phase1"

2. Run:
   python run_two_phase_training.py

3. Phase 1 optimizes:
   - Timeframe (15m, 30m, 1h, 4h, 1d)
   - Interval (bar data frequency)
   - Feature selection

4. Result: phase1_winner.json with best config

Phase 2: PPO Feature Maximization
1. Configure:
   TWO_PHASE_MODE="phase2"

2. Run:
   python run_two_phase_training.py

3. Phase 2 optimizes:
   - All PPO hyperparameters
   - Using best timeframe from Phase 1
   - Larger search space (200 trials)

4. Result: Best model in train_results/phase2_tests/
```

---

## Configuration Files

### `.env.training`

**Purpose**: Training environment configuration

```bash
# Active study name (CRITICAL for ensemble loading!)
ACTIVE_STUDY_NAME="cappuccino_alpaca_v2"

# Training workers (0 = disabled)
TRAINING_WORKERS=3

# Two-phase training
TWO_PHASE_ENABLED=true
TWO_PHASE_MODE="full"  # or "phase1" or "phase2"
```

**CRITICAL**: `ACTIVE_STUDY_NAME` affects ensemble loading if not specified in manifest!

### `ensemble_manifest.json`

**Purpose**: Ensemble model list

```json
{
  "model_count": 20,
  "trial_numbers": [686, 687, 521, 578, 520],
  "trial_values": [0.1566, 0.1565, 0.1563, 0.1562, 0.1561],
  "study_name": "cappuccino_alpaca_v2",
  "trial_ids": [1506, 1507, 1293, 1350, 1292],
  "created_at": "2025-12-18T10:00:00",
  "created_by": "setup_ensemble.py"
}
```

**CRITICAL FIELDS**:
- `study_name`: Source study (prevents loading wrong models!)
- `trial_ids`: Globally unique identifiers (better than trial_numbers)

### `arena_config.json`

**Purpose**: Arena deployment configuration

```json
{
  "arena_name": "arena_cappuccino_alpaca_v2",
  "study_name": "cappuccino_alpaca_v2",
  "model_count": 10,
  "models": [
    {
      "model_id": 0,
      "trial_number": 686,
      "trial_id": 1506,
      "sharpe_value": 0.1566,
      "model_path": "deployments/model_0",
      "validation_status": {
        "all_valid": true,
        "study_match": true,
        "files_exist": true,
        "best_trial_valid": true
      }
    }
  ]
}
```

### `metadata.json` (per model)

**Purpose**: Model deployment metadata

```json
{
  "trial_number": 686,
  "trial_id": 1506,
  "sharpe_value": 0.1566,
  "source_dir": "train_results/cwd_tests/trial_686_1h",
  "validation_passed": true,
  "deployed_at": "2025-12-18T10:30:00",
  "deployed_by": "setup_arena_clean.py"
}
```

---

## Monitoring and Control

### Dashboard

**Script**: `dashboard.py`

**Pages**:
1. System Overview: Processes, memory, disk
2. Paper Trading - Ensemble: Current trading status
3. Arena Competition: Model rankings (if Arena running)
4. Training Overview: Active training progress
9. Two-Phase Training: Phase 1/2 progress

**Usage**:
```bash
python dashboard.py
# Navigate with number keys
# Refresh every 5 seconds (auto)
```

### Control Scripts

#### Training Control

```bash
# Start training
./start_automation.sh

# Check status
./status_automation.sh

# Stop training
./stop_automation.sh
```

#### Arena Control

```bash
# Start Arena
./start_arena.sh

# Check status
./status_arena.sh

# Stop Arena
./stop_arena.sh
```

### Log Files

**Training**: `logs/training_worker_{i}.log`
**Paper Trading**: `logs/paper_trading_{name}.log`
**Arena**: `logs/arena.log`
**System**: `logs/system.log`

**Example log analysis**:
```bash
# Check for errors
grep -i error logs/paper_trading_BEST.log

# Count trades
grep "TRADE:" logs/paper_trading_BEST.log | wc -l

# Check last 100 lines
tail -100 logs/paper_trading_BEST.log

# Monitor live
tail -f logs/paper_trading_BEST.log
```

### Performance Metrics

**Sharpe Ratio**: Risk-adjusted return
- Formula: (mean_return - risk_free_rate) / std_return
- Target: >0.1 (good), >0.15 (excellent)

**Alpha**: Excess return vs benchmark
- Formula: agent_return - market_return
- Target: >0% (beating market)

**Trade Frequency**: Trades per hour
- Target: 1-3 trades/hour (not overtrading)
- Warning: >5 trades/hour (churning)

**Max Drawdown**: Largest peak-to-trough decline
- Target: <10% (conservative)
- Warning: >20% (risky)

---

## Common Issues and Solutions

### Issue 1: Ensemble Loading Wrong Models

**Symptom**: Ensemble underperforming, Sharpe 0.01 instead of 0.14

**Root Cause**:
- Manifest missing `study_name`
- Code uses `ACTIVE_STUDY_NAME` from environment
- Loads trials from wrong study (same trial # exists in multiple studies)

**Solution**:
1. Add `study_name` to `ensemble_manifest.json`
2. Update `ultra_simple_ensemble.py` to use manifest's `study_name`
3. Rebuild ensemble with correct study

**Prevention**: Always include `study_name` in manifest

### Issue 2: Trial Number Ambiguity

**Symptom**: Validation fails, "trial 686 exists in 5 studies"

**Root Cause**: Trial numbers are NOT unique across studies

**Solution**:
- Always specify study when querying by trial number
- OR use trial_id (globally unique)
- Manifest should store trial_ids, not just numbers

**Prevention**: Use `(study_name, trial_number)` or `trial_id`

### Issue 3: Missing best_trial Files

**Symptom**: Deployment fails, "best_trial not found"

**Root Cause**: Old trials didn't save best_trial pickle

**Solution**:
1. Use `validate_models.py` to identify missing files
2. Script can recreate from database
3. Or manually:
```python
import optuna, pickle
study = optuna.load_study(study_name='...', storage='...')
for trial in study.trials:
    if trial.number == 686:
        with open('best_trial', 'wb') as f:
            pickle.dump(trial, f)
```

**Prevention**: Ensure training script always saves best_trial

### Issue 4: Overtrading

**Symptom**: Many trades per hour, poor performance despite good models

**Root Cause**:
- Models reacting to noise
- Position sizing too aggressive
- Transaction costs eating profits (0.25% per trade)

**Solution**:
1. Reduce position updates (longer intervals)
2. Add minimum trade size threshold
3. Implement trade cost awareness
4. Use ensemble voting (reduces variance)

**Prevention**: Backtest with realistic transaction costs

### Issue 5: Stale Data in Dashboard

**Symptom**: Dashboard shows old data (Dec 12-16)

**Root Cause**:
- Paper trader not running
- Log files not being updated
- Dashboard caching old data

**Solution**:
1. Check if paper trader running: `pgrep -f paper_trader`
2. Check log timestamps: `stat logs/paper_trading_BEST.log`
3. Restart paper trader if needed

**Prevention**: Monitor process health, auto-restart on failure

---

## Appendix: File Formats

### PyTorch Model File (.pth)

**Format**: PyTorch state dict (Python dictionary serialized with `torch.save`)

**Contents**:
```python
{
  'layer1.weight': tensor([[0.1, 0.2, ...], ...]),
  'layer1.bias': tensor([0.1, 0.2, ...]),
  'layer2.weight': tensor([[...], ...]),
  ...
}
```

**Load**:
```python
import torch
state_dict = torch.load('actor.pth')
model.load_state_dict(state_dict)
```

### Optuna Trial Pickle File (best_trial)

**Format**: Python pickle of `optuna.trial._frozen.FrozenTrial` object

**Contents**:
- `number`: Trial number (0, 1, 2, ...)
- `state`: TrialState (COMPLETE, RUNNING, etc.)
- `value`: Objective value (Sharpe ratio)
- `params`: Dictionary of hyperparameters
- `user_attrs`: Custom attributes
- `datetime_start`, `datetime_complete`: Timestamps

**Load**:
```python
import pickle
with open('best_trial', 'rb') as f:
    trial = pickle.load(f)
print(trial.number, trial.value, trial.params)
```

### Positions State JSON

**Format**: JSON dictionary

**Schema**:
```json
{
  "portfolio_value": float,
  "cash": float,
  "positions": [
    {
      "ticker": string,
      "quantity": float,
      "avg_entry": float,
      "current_price": float,
      "position_value": float,
      "pnl_pct": float
    }
  ],
  "timestamp": string (ISO 8601)
}
```

---

## Conclusion

This document provides a complete reference for the Cappuccino trading system architecture. Key takeaways:

1. **Trial numbers are NOT unique** - Always use (study_name, number) or trial_id
2. **Manifests must include study_name** - Prevents loading wrong models
3. **Validation is critical** - Use validate_models.py before deployment
4. **Monitor continuously** - Dashboard, logs, performance metrics
5. **Document everything** - This doc prevents confusion and bugs

For questions or issues, refer to:
- `ROOT_CAUSE_IDENTIFIED.md`: Ensemble bug analysis
- `CRITICAL_ISSUES_FOUND.md`: Known issues and fixes
- `validate_models.py`: Model validation tool
- `setup_arena_clean.py`: Arena deployment tool

**Status**: System ready for deployment with proper validation.
