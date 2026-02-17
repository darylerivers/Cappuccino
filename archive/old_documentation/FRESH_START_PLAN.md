# Fresh Start - Correct Implementation Plan

**Date**: December 18, 2025
**Status**: Ready to execute

---

## Current Situation

### What You Have
- ✅ 3 training workers STOPPED
- ✅ Arena STOPPED
- ✅ Paper trading STOPPED
- ✅ Data from Dec 5, 2025 (13 days stale)
- ✅ Data preparation script exists

### What's Wrong
- ❌ Deployed models trained on Nov 1 data (47 days stale)
- ❌ Current training study (cappuccino_week_20251206) has terrible Sharpe (0.0149)
- ❌ Arena is NOT tournament-style (your vision)
- ❌ No clear pipeline (sandbox → production → live)
- ❌ Training running indefinitely (not limited to 1000 trials)

---

## Immediate Action Plan (Next 24 Hours)

### Step 1: Download Fresh Data (30-60 minutes)

**Get 2 years of data ending TODAY:**

```bash
# Download 24 months of fresh data
python prepare_1year_training_data.py \
    --months 24 \
    --output-dir data/2year_fresh_20251218 \
    --train-pct 0.8
```

**This will:**
- Download data from Dec 18, 2023 → Dec 18, 2025
- Create train/val split (80/20)
- Compute technical indicators
- Save to `data/2year_fresh_20251218/`

**Wait for it to complete** (may take 30-60 minutes due to API rate limits)

---

### Step 2: Verify Data Quality (5 minutes)

```bash
# Check the data
python3 << 'EOF'
import numpy as np
from datetime import datetime

time_array = np.load('data/2year_fresh_20251218/time_array.npy', allow_pickle=True)
first_dt = datetime.fromtimestamp(time_array[0])
last_dt = datetime.fromtimestamp(time_array[-1])

print(f"Data range: {first_dt} → {last_dt}")
print(f"Days covered: {(last_dt - first_dt).days}")
print(f"Data points: {len(time_array)}")
print(f"Days stale: {(datetime.now() - last_dt).days}")

# Should show:
# Data range: ~Dec 2023 → Dec 18, 2025
# Days covered: ~730 (2 years)
# Days stale: 0
EOF
```

---

### Step 3: Start Fresh Training (Limited to 1000 Trials)

**Create new study with fresh data:**

```bash
# Create new study with FRESH data
# Limit to 1000 trials as you specified
python 1_optimize_unified.py \
    --study-name "cappuccino_2year_20251218" \
    --n-trials 1000 \
    --workers 3 \
    --gpu 0 \
    --data-dir data/2year_fresh_20251218
```

**What this does:**
- Trains on FRESH 2-year data
- Stops automatically after 1000 trials (not infinite)
- 3 workers = ~33 hours to complete 1000 trials
- Uses 1h timeframe (can optimize this later)

---

### Step 4: Monitor Training (Next 24-48 hours)

```bash
# Check progress
python dashboard.py  # Page 9 for training

# Or check database
sqlite3 databases/optuna_cappuccino.db "
SELECT COUNT(*) as trials, MAX(value) as best_sharpe
FROM trials t
JOIN trial_values tv ON t.trial_id = tv.trial_id
WHERE t.study_id = (
    SELECT study_id FROM studies WHERE study_name = 'cappuccino_2year_20251218'
)
"
```

**Wait for ~100 trials** before deploying (gives you good candidates)

---

### Step 5: Deploy Best Model to Arena (After 100+ Trials)

```bash
# After you have 100+ trials with Sharpe > 0.1
python setup_arena_clean.py \
    --study cappuccino_2year_20251218 \
    --top-n 10

# Then monitor
python dashboard.py  # Page 3
```

---

## Medium-Term Fixes (Next Week)

These address your vision but require code changes:

### 1. Implement Tournament Arena

**Current arena**: All 10 models trade simultaneously

**Your vision**: Head-to-head elimination tournament

**Need to build**:
```python
# tournament_arena.py
class TournamentArena:
    def __init__(self):
        self.champion = None
        self.consecutive_wins = 0
        self.eliminated = []

    def match(self, challenger):
        """Head-to-head match: champion vs challenger"""
        # 1. Both trade for 24 hours
        # 2. Compare Sharpe ratios
        # 3. Winner stays, loser eliminated

    def promote_champion(self):
        """Champion beat 20 models → promote to sandbox"""
        if self.consecutive_wins >= 20:
            self.move_to_sandbox(self.champion)
```

### 2. Implement Pipeline Manager

**Your vision**: Sandbox (2 days) → Production → Live

**Need to build**:
```python
# pipeline_manager.py
class PipelineManager:
    stages = {
        'arena': [],        # Tournament competitors
        'sandbox': [],      # 2-day trial period
        'production': None, # Current live paper trader
        'live': None        # Real money (future)
    }

    def check_sandbox_ready(self):
        """After 2 days, compare to production"""

    def replace_production(self, sandbox_model):
        """Sandbox model beats production → replace"""
```

### 3. Add Training Limits

**Current**: Infinite training

**Your vision**: Max 1000 trials

**Fix**:
```python
# In 1_optimize_unified.py
def should_continue(study, max_trials=1000):
    return len(study.trials) < max_trials
```

### 4. Implement Two-Phase Training

**Your vision**:
1. Phase 1: Optimize timeframe/interval (small sample, 50-100 trials)
2. Phase 2: Train models with best config (1000 trials max)

**Need to build**:
```bash
# Phase 1: Find best timeframe
python phase1_timeframe_optimizer.py \
    --trials 100 \
    --data-dir data/2year_fresh_20251218

# Output: Best timeframe = 1h, lookback = 5

# Phase 2: Train with best config
python 1_optimize_unified.py \
    --study-name "cappuccino_phase2_20251218" \
    --n-trials 1000 \
    --timeframe 1h \
    --lookback 5 \
    --data-dir data/2year_fresh_20251218
```

---

## Long-Term Vision (Fully Implemented System)

### Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Download 2 years fresh data (weekly update)              │
│  2. Phase 1: Optimize timeframe (100 trials)                 │
│  3. Phase 2: Train models (1000 trials max)                  │
│                                                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                  TOURNAMENT ARENA                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  • Trial 1 vs Trial 2 → Winner stays                         │
│  • Trial 3 challenges → Winner stays                         │
│  • Continue until champion beats 20 models                   │
│  • Champion promoted to SANDBOX                              │
│                                                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                    SANDBOX (2 days)                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  • Multiple models can be in sandbox                         │
│  • Each trades for 2 days                                    │
│  • Head-to-head vs current production model                  │
│  • If beats production → REPLACE IT                          │
│                                                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│              PRODUCTION PAPER TRADING                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  • ONLY ONE model in production at a time                    │
│  • Continuously monitored                                    │
│  • Replaced if sandbox model beats it                        │
│  • OR promoted to LIVE if green-lit                          │
│                                                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓ (Future)
┌─────────────────────────────────────────────────────────────┐
│                    LIVE TRADING                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  • Real money                                                │
│  • Strict monitoring                                         │
│  • Can be demoted if performance degrades                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## What to Do RIGHT NOW

### Immediate (Next 30 minutes):

```bash
# 1. Download fresh data (THIS IS CRITICAL)
python prepare_1year_training_data.py \
    --months 24 \
    --output-dir data/2year_fresh_20251218

# 2. Wait for download to complete
# (Will take 30-60 minutes)
```

### After Download Completes:

```bash
# 3. Verify data is fresh
python3 << 'EOF'
import numpy as np
from datetime import datetime
time_array = np.load('data/2year_fresh_20251218/time_array.npy', allow_pickle=True)
last_dt = datetime.fromtimestamp(time_array[-1])
print(f"Data ends: {last_dt}")
print(f"Days stale: {(datetime.now() - last_dt).days}")
EOF

# 4. Start fresh training (limit 1000 trials)
python 1_optimize_unified.py \
    --study-name "cappuccino_2year_20251218" \
    --n-trials 1000 \
    --workers 3 \
    --gpu 0
```

### Tomorrow (After ~100 Trials):

```bash
# 5. Check best Sharpe
sqlite3 databases/optuna_cappuccino.db "
SELECT MAX(value) FROM trial_values tv
JOIN trials t ON tv.trial_id = t.trial_id
WHERE t.study_id = (SELECT study_id FROM studies WHERE study_name = 'cappuccino_2year_20251218')
"

# If Sharpe > 0.1, deploy to arena
python setup_arena_clean.py \
    --study cappuccino_2year_20251218 \
    --top-n 10
```

---

## Summary: Fix Priority

### Priority 1: Data Staleness (FIX TODAY)
✅ **You can do this NOW**
- Download fresh 2-year data
- Verify data ends today (Dec 18, 2025)

### Priority 2: Training Limits (FIX TODAY)
✅ **You can do this NOW**
- Limit training to 1000 trials
- Use fresh data

### Priority 3: Tournament Arena (REQUIRES CODE - Next Week)
❌ **Need to build this**
- Head-to-head matches
- Beat 20 to promote
- Winner stays, loser eliminated

### Priority 4: Pipeline Manager (REQUIRES CODE - Next Week)
❌ **Need to build this**
- Sandbox → Production → Live
- 2-day sandbox evaluation
- Automatic promotion/demotion

### Priority 5: Two-Phase Training (REQUIRES CODE - Later)
❌ **Nice to have**
- Phase 1: Optimize timeframe
- Phase 2: Train with best config

---

## Your Next Command

**Run this NOW to fix the data staleness issue:**

```bash
python prepare_1year_training_data.py \
    --months 24 \
    --output-dir data/2year_fresh_20251218
```

Then come back and tell me when it's done. We'll verify the data and start fresh training.

**This is the single most important thing to fix first.**
