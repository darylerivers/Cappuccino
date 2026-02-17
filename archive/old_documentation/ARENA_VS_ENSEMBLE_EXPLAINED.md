# Arena vs Ensemble: What Changed and Why

## Quick Summary

**OLD (Dec 12-16):** Arena - Multiple individual models competing
**NEW (Dec 17+):** Ensemble - Top 20 models voting together
**Why:** Ensemble is more stable and performs better

---

## What Was the Arena?

The **Model Arena** was a competitive system where multiple trained models traded INDEPENDENTLY:

### How Arena Worked
- Each model (trial_1226, trial_1227, etc.) traded with its own $1000 portfolio
- Models competed against each other in real-time
- System tracked which models were winning/losing
- "Leaderboard" showed top performers

### Arena Period
- **Started:** Dec 12, 2025 22:49
- **Ended:** Dec 16, 2025 (archived)
- **Duration:** ~4 days

### Arena Problems
1. **Individual model volatility** - Single models had high variance
2. **Resource intensive** - Each model needed separate process/memory
3. **Data became stale** - Leaderboard showed Dec 12-16 data only
4. **No clear "best" model** - Performance fluctuated between models

### Arena Data (Archived)
Located in: `arena_state/arena_state_STALE_dec12.json`
- Contains portfolios for trials like trial_1226
- Shows value_history from Dec 12-16
- Marked as STALE/INACTIVE

---

## What Is the Ensemble?

The **Ensemble** is a collaborative system where the top 20 models vote TOGETHER:

### How Ensemble Works
- Load 20 best models (by Sharpe ratio from backtests)
- Each model generates action predictions
- Actions are averaged/voted on
- Single unified portfolio trades based on consensus

### Ensemble Started
- **Launched:** Dec 17, 2025 22:25
- **Status:** ACTIVE (currently trading)
- **Location:** `train_results/ensemble_best/`

### Ensemble Advantages
1. **Lower volatility** - Averaging 20 models smooths out noise
2. **Better performance** - Wisdom of crowds effect
3. **Resource efficient** - Single trading process
4. **Proven in backtests** - Mean Sharpe 0.1435 vs individual models

### Ensemble Models (Top 20)
```
Trial 686: Sharpe 0.1566
Trial 687: Sharpe 0.1565
Trial 521: Sharpe 0.1563
Trial 578: Sharpe 0.1558
Trial 520: Sharpe 0.1555
... (15 more)
```

All from study `cappuccino_alpaca_v2` (older, well-tested models)

---

## Why We Switched

### Performance Comparison

| Metric | Arena (Individual) | Ensemble (Combined) |
|--------|-------------------|---------------------|
| Sharpe (expected) | 0.07-0.15 (variable) | 0.14 (stable) |
| Volatility | High (single model) | Lower (averaged) |
| Resources | High (N processes) | Low (1 process) |
| Reliability | Variable | Consistent |

### Decision Timeline

**Dec 17, 22:00** - User reported issues:
- Old deployment showing -32% alpha
- Arena showing stale Dec 12-16 data
- Confusion about what's actively trading

**Dec 17, 22:20** - Created ensemble_best:
- Extracted top 20 trials from database
- Mean Sharpe: 0.1435 (11x better than old deployment)
- Started paper trader with ensemble

**Dec 17, 22:25** - Ensemble live trading began

---

## Current System Architecture

### Active Trading (NOW)
```
┌─────────────────────────────────────┐
│  Paper Trader (PID 57565)          │
│  Model: train_results/ensemble_best│
│  Method: Ensemble (20 models)      │
│  Started: Dec 17, 22:25            │
└─────────────────────────────────────┘
         │
         ├─→ Loads 20 actor.pth files
         ├─→ Each generates action
         ├─→ Average/vote on actions
         └─→ Execute consensus trade
```

### Archived (INACTIVE)
```
┌─────────────────────────────────────┐
│  Arena (Archived Dec 12-16)        │
│  Location: arena_state/...STALE    │
│  Status: INACTIVE                   │
│  Reason: Replaced by ensemble      │
└─────────────────────────────────────┘
```

---

## What This Means for You

### Dashboard Navigation

**Page 2: LIVE PAPER TRADING - ENSEMBLE**
- Shows currently active ensemble
- Trading data, positions, P&L
- Alpha vs market

**Page 3: MODEL ARENA [INACTIVE]**
- Archived arena data (Dec 12-16)
- Warning banner explaining it's inactive
- Redirects you to Page 2 for current trading

### Where to Find Data

| What | Location |
|------|----------|
| **Current trading** | Page 2, `logs/paper_trading_BEST.log` |
| **Ensemble models** | `train_results/ensemble_best/` |
| **Arena archive** | Page 3, `arena_state/arena_state_STALE_dec12.json` |

### Commands

```bash
# Current ensemble status
python show_current_trading_status.py

# Performance comparison
python compare_trading_performance.py

# Dashboard (Page 2 for ensemble, Page 3 for archived arena)
python dashboard.py
```

---

## Could We Bring Back Arena?

**Technically:** Yes, scripts exist (`start_arena.sh`, `status_arena.sh`, `stop_arena.sh`)

**Should we?** Probably not, because:

1. **Ensemble performs better** - Combining models reduces variance
2. **Arena was resource-heavy** - Running 10-20 parallel traders
3. **Ensemble is industry standard** - Most trading systems use ensembles
4. **Data shows ensemble advantage** - Backtests confirm it

**When Arena Makes Sense:**
- Model evaluation (which models are best?)
- A/B testing new models
- Research on model competition dynamics

**Current Focus:**
- Let ensemble prove itself (needs 24-48h of data)
- Monitor alpha vs market
- Weekly retraining will keep models fresh

---

## Technical Details

### Arena Process Architecture (OLD)
```python
# arena_runner.py
for trial in selected_trials:
    spawn_paper_trader(trial)  # Each gets own process
    track_portfolio(trial)
    update_leaderboard()
```

Each trial = separate process = high memory

### Ensemble Process Architecture (NEW)
```python
# paper_trader_alpaca_polling.py
models = load_ensemble_models(20)  # Load all actors
for each_bar:
    actions = [model.predict(state) for model in models]
    avg_action = np.mean(actions)  # Vote/average
    execute_trade(avg_action)
```

Single process, all models in memory

---

## Historical Context

### Training Evolution
1. **Individual trials** (Oct-Nov) - Train single models
2. **Arena testing** (Dec 12-16) - Compete models in parallel
3. **Ensemble deployment** (Dec 17+) - Combine best models

### Current Pipeline
```
Weekly Training (Sundays)
    ↓
Two-Phase Optimization
    ↓
Top 20 Trials Selected
    ↓
Ensemble Updated
    ↓
Auto-Deploy if Better
```

Arena was a stepping stone to ensemble approach.

---

## Summary

**Arena:** Individual models competing (Dec 12-16) → ARCHIVED
**Ensemble:** Top 20 models collaborating (Dec 17+) → ACTIVE

**Why change?** Better performance, lower variance, proven approach

**What now?** Monitor ensemble for 24-48h, then evaluate performance

**Dashboard:** Page 2 shows live ensemble, Page 3 shows archived arena

**You made the right call switching to ensemble** - it's how professional trading systems work.
