# Hybrid Trading Approach - Implementation Guide

## Overview
Combines immediate deployment with arena-based verification for optimal model selection.

## How It Works

### 1. **Immediate Deployment** (Speed)
- Best training model → Deployed directly to paper trading
- No waiting for evaluation period
- Positions persist across model changes

### 2. **Arena Verification** (Quality)
- Next 9 best models → Added to arena for evaluation
- Arena runs simulated trading in parallel
- Tracks actual performance over 7 days minimum

### 3. **Promotion System** (Safety)
- If arena model shows better performance:
  - Must beat current paper trader by 2%+ return
  - Must have positive Sharpe ratio
  - Must run at least 168 hours (7 days)
- Then: Promote arena winner to replace current paper trader

## Benefits

### vs Pure Training-Based
- ✅ Immediate deployment (don't wait for evaluation)
- ✅ But with safety net (arena validates before next swap)
- ✅ Positions persist (no reset on model change)

### vs Pure Arena-Based
- ✅ Faster to production (instant vs 7 days)
- ✅ Still get validation (arena runs in background)
- ✅ Best of both worlds

## Current Status

### What's Running
```bash
./status_automation.sh  # Auto deployer + ensemble updater
./status_arena.sh       # Arena with 10 models
```

### Configuration Files
- `deployments/deployment_state.json` - Tracks current paper trader
- `arena_state/arena_state.json` - Tracks arena models
- `arena_state/promotion_candidate.json` - Written when model ready to promote

## Workflow

### Training Completes
1. New trial finishes in Optuna
2. Auto deployer checks if it beats current paper trader's **training score**
3. If yes: Deploy to paper trading immediately
4. Add previous 9 best trials to arena

### Arena Monitors
1. Every hour, arena steps all 10 models
2. Tracks simulated performance
3. After 7 days, checks if any model is promotion-ready
4. Writes `promotion_candidate.json` if found

### Auto Deployer Checks
1. Every cycle, checks for `promotion_candidate.json`
2. If found and candidate beats current paper trader:
   - Stop current paper trader
   - Deploy arena winner
   - Keep positions (load from saved state)

## Implementation Steps

### Phase 1: Enable Hybrid Mode (Current)
```bash
# Stop current automation
./stop_automation.sh

# Start in hybrid mode
python3 auto_model_deployer.py --hybrid-mode &

# Arena already running
# ./start_arena.sh  # Already running
```

### Phase 2: Position Persistence (Already Implemented)
- Paper trader saves positions to `paper_trades/positions_state.json`
- Loads on startup to maintain positions across model swaps

### Phase 3: Monitoring
```bash
# Check paper trader
tail -f logs/paper_trading_auto_*.log

# Check arena
tail -f logs/arena_runner_*.log
cat arena_state/leaderboard.txt

# Check for promotion candidates
cat arena_state/promotion_candidate.json
```

## Key Differences from Old System

### Old (Ensemble-Only)
- Deploy ensemble of top 10 models
- Positions reset when new model found
- No independent evaluation

### New (Hybrid)
- Deploy single best model immediately
- Arena evaluates others in parallel
- Positions persist across swaps
- Promotion requires proof of better performance

## Safety Features

1. **Training threshold**: Must beat current by training score
2. **Arena promotion**: Must show 2%+ return, positive Sharpe, 7+ days
3. **Position persistence**: No reset on model swap
4. **Gradual rollout**: One model at a time
5. **Monitoring**: Full logging and state tracking

## Commands

```bash
# Check current deployment
cat deployments/deployment_state.json | python3 -m json.tool

# Check arena status
./status_arena.sh

# View arena leaderboard
cat arena_state/leaderboard.txt

# Check for promotion candidates
cat arena_state/promotion_candidate.json 2>/dev/null || echo "None ready"

# View paper trader positions
cat paper_trades/positions_state.json | python3 -m json.tool
```

## Next Steps

1. ✅ Arena running and tracking models
2. ⏳ Enable hybrid mode in auto deployer
3. ⏳ Test promotion workflow
4. ⏳ Monitor for 7 days to see first arena promotion
