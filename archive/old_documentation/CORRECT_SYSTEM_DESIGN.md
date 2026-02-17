# Correct Cappuccino System Design

**Date**: December 18, 2025
**Status**: User's intended design vs. what currently exists

---

## User's Intended Design (THE RIGHT WAY)

### Phase 1: Data Preparation
1. Download **2 years of historical data** for selected assets
2. Store in local database
3. Update data regularly (weekly or daily)

### Phase 2: Timeframe Optimization (Two-Phase Training)
1. Test combinations of:
   - Time intervals (15m, 30m, 1h, 4h, 1d)
   - Time frames (lookback windows)
2. Find the BEST combination
3. This should be a SMALL sample (maybe 50-100 trials max)

### Phase 3: Model Training
1. Use ONLY the best interval/timeframe from Phase 2
2. Train **AT MOST 1000 trials**
3. All trials use FRESH data (last 2 years, updated regularly)
4. Stop after 1000 trials or diminishing returns

### Phase 4: Tournament-Style Arena
**This is the key difference from current implementation!**

#### Tournament Structure:
1. **Start**: Trial 1 vs. Trial 2 (head-to-head)
2. **Winner stays**, loser is eliminated
3. **Next challenger**: Trial 3 enters
4. **Winner stays**, loser is eliminated
5. **Continue** until a model has beaten 20 consecutive opponents

#### Promotion Criteria:
- Model must **beat 20 different models** to be promoted
- Beaten models are eliminated from the arena
- New trials continuously enter as challengers

### Phase 5: Sandbox Paper Trading
1. **Promoted model** goes to sandbox paper trading
2. Trades for **up to 2 days** with real market data
3. **Other models** continue competing in arena
4. **Multiple models** can be in sandbox simultaneously

### Phase 6: Production Paper Trading
**Only ONE model in production at a time**

#### Replacement Criteria:
Model is replaced if:
1. **A sandbox model beats it** in head-to-head comparison, OR
2. **It's green-lit for LIVE TRADING** and moves to real money

#### Green-light Criteria for Live Trading:
- Positive alpha over 2 days
- Sharpe ratio > 0.1
- Max drawdown < 15%
- No major errors or crashes

### Phase 7: Live Trading (Real Money)
- Only reached after passing all previous phases
- Continuous monitoring
- Can be demoted back to paper if performance degrades

---

## Current System (WHAT'S ACTUALLY HAPPENING - WRONG!)

### Data Management
❌ **No regular data updates**
❌ Models training on stale data
❌ No clear 2-year data repository

### Training
❌ Multiple studies running without coordination
❌ No clear "1000 trial max" limit
❌ Training workers running even when TRAINING_WORKERS=0
❌ Best models from November, current training produces Sharpe 0.01

### Arena
❌ **NOT tournament-style** - all models trade simultaneously
❌ No head-to-head elimination
❌ No "beat 20 to promote" logic
❌ Just ranks by Sharpe ratio

### Promotion Path
❌ No clear sandbox → production → live pipeline
❌ No automatic promotion/demotion
❌ Manual deployment decisions

---

## Critical Issues to Fix

### Issue 1: Data Staleness (HIGHEST PRIORITY)
**Problem**: Models from Nov 1 trading in Dec 18 markets (47 days stale)

**Fix**:
```bash
# 1. Stop all training
pkill -f 1_optimize_unified.py

# 2. Download fresh 2 years of data
python prepare_training_data.py --start-date 2023-12-18 --end-date 2025-12-18

# 3. Start NEW study with fresh data
# ONLY after data is confirmed fresh
```

### Issue 2: Training Not Controlled
**Problem**: Workers running even though TRAINING_WORKERS=0

**Fix**:
```bash
# Stop all workers
pkill -f 1_optimize_unified.py

# Verify stopped
pgrep -f 1_optimize_unified.py
# Should return nothing
```

### Issue 3: Arena Not Tournament-Style
**Problem**: Current arena doesn't match intended design

**Fix**: Need to rewrite `model_arena.py` to implement:
- Head-to-head matches
- Winner stays, loser eliminated
- Track consecutive wins
- Promote after 20 wins

### Issue 4: No Clear Pipeline
**Problem**: No sandbox → production → live progression

**Fix**: Create pipeline manager:
```python
# pipeline_manager.py
class TradingPipeline:
    stages = {
        'arena': [],          # Competing models
        'sandbox': [],        # Promoted models (2 days trial)
        'production': None,   # Single live paper trader
        'live': None          # Real money (future)
    }

    def promote_from_arena(self, model_id):
        """Model beat 20 opponents, move to sandbox"""

    def promote_to_production(self, model_id):
        """Sandbox model beat production, replace it"""

    def demote_to_sandbox(self, model_id):
        """Production model underperforming, back to sandbox"""
```

---

## Immediate Action Plan

### Step 1: STOP EVERYTHING (Now)
```bash
# Kill all training
pkill -f 1_optimize_unified.py

# Kill arena
pkill -f arena_runner.py

# Kill paper trading
pkill -f paper_trader_alpaca_polling.py
```

### Step 2: Get Fresh Data (1-2 hours)
```bash
# Download 2 years of data
python prepare_training_data.py \
    --start-date 2023-12-18 \
    --end-date 2025-12-18 \
    --tickers BTC/USD,ETH/USD,LTC/USD,BCH/USD,LINK/USD,UNI/USD,AAVE/USD
```

### Step 3: Verify Data Freshness
```bash
# Check what data you have
ls -lh data/
# Should show files with today's date
```

### Step 4: Start Fresh Training (With Limits)
```bash
# NEW study with fresh data
# MAX 1000 trials
python 1_optimize_unified.py \
    --study-name "cappuccino_fresh_20251218" \
    --n-trials 1000 \
    --workers 3 \
    --timeframe 1h \
    --lookback 5
```

### Step 5: Build Tournament Arena (Requires Code)
This needs a rewrite of `model_arena.py` to implement your vision.

### Step 6: Build Pipeline Manager (Requires Code)
Create the sandbox → production → live progression system.

---

## What Needs to Be Built

### 1. Data Manager
```python
class DataManager:
    def download_fresh_data(self, start_date, end_date):
        """Download 2 years of data"""

    def verify_data_freshness(self):
        """Check data is not stale"""

    def get_training_data(self):
        """Return fresh data for training"""
```

### 2. Tournament Arena
```python
class TournamentArena:
    def match(self, model_a, model_b):
        """Head-to-head match"""

    def track_wins(self, model_id):
        """Track consecutive wins"""

    def promote(self, model_id):
        """Model beat 20, promote to sandbox"""
```

### 3. Pipeline Manager
```python
class PipelineManager:
    def check_sandbox_models(self):
        """Monitor sandbox after 2 days"""

    def compare_sandbox_vs_production(self):
        """Head-to-head comparison"""

    def replace_production(self, new_model_id):
        """Replace production model"""
```

### 4. Training Limiter
```python
class TrainingLimiter:
    max_trials = 1000

    def should_continue(self, study):
        """Stop after 1000 trials"""
        return study.trials_count < self.max_trials
```

---

## Summary: What You Were Right About

1. ✅ **Data should be fresh (2 years)** - Currently 47 days stale
2. ✅ **Phase 1 should optimize timeframe** - Currently mixed in with training
3. ✅ **Max 1000 trials** - Currently unlimited, 2728 trials in active study
4. ✅ **Arena should be tournament** - Currently simultaneous trading
5. ✅ **Beat 20 to promote** - Currently no promotion logic
6. ✅ **Sandbox → Production → Live pipeline** - Currently no pipeline

---

## What I Failed To Do

1. ❌ Fix data staleness (you mentioned this multiple times)
2. ❌ Implement YOUR vision (tournament arena)
3. ❌ Create clear pipeline (sandbox → production → live)
4. ❌ Stop training at sensible limits (1000 trials)
5. ❌ Keep training and deployment aligned

**You were right to be frustrated. The system is not what you envisioned.**

---

## Next Steps (Your Call)

### Option A: Stop and Rebuild (Recommended)
1. Stop everything
2. Get fresh data
3. Build tournament arena
4. Build pipeline manager
5. Start from scratch with YOUR design

### Option B: Quick Fix Current System
1. Stop stale model trading
2. Train on fresh data (1000 trial limit)
3. Deploy best new model
4. Iterate later

### Option C: Minimal Fix
1. Just get fresh data
2. Retrain top model
3. Keep current (flawed) arena
4. Live with limitations

**Which direction do you want to go?**
