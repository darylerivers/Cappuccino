# Multi-Timeframe Ensemble Strategy

## Overview

Once you reach 1000 trials on the 1-hour timeframe, you can enhance your ensemble with shorter timeframes (5m, 15m, 30m) for better entry/exit timing.

## Current Status

**Progress to 1000 Trials:**
- Current: 757 completed trials
- Remaining: 243 trials to reach 1000
- Best value: 0.012626
- Top 10% models: 79 trials

**Estimated Time to 1000:**
- At current rate (~6 trials/hour with 3 workers)
- ~40 hours remaining (~1.7 days)

## Strategy Concept

### Hierarchical Multi-Timeframe Ensemble

```
┌─────────────────────────────────────────────────┐
│          MULTI-TIMEFRAME ENSEMBLE               │
└─────────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
    ┌───▼────┐            ┌─────▼─────┐
    │ 1h     │            │ Short TF  │
    │ Models │            │ Models    │
    │ (10)   │            │ (5m/15m)  │
    └───┬────┘            └─────┬─────┘
        │                       │
        │ Strategic             │ Tactical
        │ Direction             │ Entry Timing
        │                       │
        └───────────┬───────────┘
                    │
            ┌───────▼────────┐
            │ FINAL DECISION │
            └────────────────┘
```

### How It Works

**1-Hour Models (Strategic Layer)**
- Decide overall position direction (LONG/SHORT/NEUTRAL)
- Determine position sizing
- Set target assets
- Trained on long-term patterns

**Short Timeframe Models (Tactical Layer)**
- Refine exact entry/exit timing
- Detect short-term momentum
- Identify micro-trends for better fills
- React to immediate price action

**Combined Decision**
- If 1h models say "BUY BTC": Wait for 5m/15m models to confirm good entry
- If short TF shows momentum down: Delay entry or scale in gradually
- If short TF shows momentum up: Execute immediately or increase size

## Implementation Plan

### Phase 1: Prepare Training Data (Before 1000 trials)

```bash
# Prepare 5-minute data
python prepare_multi_timeframe_data.py --timeframe 5m --lookback-days 90

# Prepare 15-minute data
python prepare_multi_timeframe_data.py --timeframe 15m --lookback-days 180

# Prepare 30-minute data (optional)
python prepare_multi_timeframe_data.py --timeframe 30m --lookback-days 270
```

### Phase 2: Train Short Timeframe Models (After 1000 trials)

```bash
# Launch training for 5m timeframe (24 workers recommended)
./train_multi_timeframe.sh --timeframe 5m --workers 24 --trials 500

# Launch training for 15m timeframe (12 workers)
./train_multi_timeframe.sh --timeframe 15m --workers 12 --trials 500
```

### Phase 3: Create Multi-Timeframe Ensemble

```bash
# Create ensemble combining 1h + 5m + 15m models
python create_multi_timeframe_ensemble.py \
    --strategic-timeframe 1h \
    --tactical-timeframes 5m,15m \
    --strategic-models 10 \
    --tactical-models-per-tf 5
```

### Phase 4: Deploy to Paper Trading

```bash
# Test with multi-timeframe ensemble
python paper_trader_alpaca_polling.py \
    --model-dir train_results/ensemble_multi_tf \
    --timeframe 5m \
    --poll-interval 60 \
    --use-multi-timeframe
```

## Training Recommendations

### Timeframe Characteristics

| Timeframe | Data Points/Day | Training Duration | Workers | Trials | Use Case |
|-----------|-----------------|-------------------|---------|--------|----------|
| 1h        | 24              | 3-7 days         | 3       | 1000   | Strategy direction |
| 30m       | 48              | 3-5 days         | 6       | 500    | Medium-term timing |
| 15m       | 96              | 2-4 days         | 12      | 500    | Entry refinement |
| 5m        | 288             | 2-3 days         | 24      | 500    | Precise execution |

### Resource Allocation

**Current Setup (1h training):**
- 3 workers on 1h timeframe
- ~6 trials/hour
- GPU: 99% utilization

**Proposed Multi-Timeframe Setup:**
- Keep 1h training running (1-2 workers for maintenance)
- Add 12-24 workers for short timeframes
- Split GPU time or use multiple GPUs

**GPU Considerations:**
- 5m models train faster (more data points per episode)
- Can train 2-3x more trials per day on short timeframes
- Consider using CPU for some workers if GPU saturated

## Signal Combination Strategies

### Strategy 1: Gating (Conservative)

```python
# 1h model must agree before considering short TF
if strategic_signal > threshold:
    if tactical_signal > threshold:
        execute_trade()  # Both agree
    else:
        wait()  # Strategic yes, tactical no - wait for better entry
else:
    no_trade()  # Strategic says no
```

**Pros:** Safer, fewer false signals
**Cons:** May miss some opportunities

### Strategy 2: Weighted Average (Balanced)

```python
# Combine signals with weights
final_signal = (strategic_signal * 0.7) + (tactical_signal * 0.3)

if final_signal > threshold:
    execute_trade()
```

**Pros:** Smooth combination, uses all info
**Cons:** Can dilute strong signals

### Strategy 3: Confidence-Based (Adaptive)

```python
# Use confidence scores
if strategic_confidence > 0.8:
    weight_strategic = 0.8
else:
    weight_strategic = 0.5

final_signal = (strategic_signal * weight_strategic) +
               (tactical_signal * (1 - weight_strategic))
```

**Pros:** Adapts to market conditions
**Cons:** More complex, needs tuning

### Strategy 4: Timing Refinement (Recommended)

```python
# Strategic decides IF, tactical decides WHEN
strategic_direction = sign(strategic_signal)  # +1, 0, -1

if strategic_direction != 0:
    # Wait for tactical confirmation
    if sign(tactical_signal) == strategic_direction:
        # Both agree on direction
        if abs(tactical_signal) > entry_threshold:
            execute_trade(strategic_direction)
        else:
            wait_for_momentum()
    else:
        # Conflicting signals - wait
        wait_for_alignment()
```

**Pros:** Clear separation of concerns, best entry timing
**Cons:** May delay entries

## Expected Improvements

### Entry Timing

**Before (1h only):**
- Enter at start of 1h bar
- May enter at peak/trough of hour
- Average slippage: ~0.5-1%

**After (with 5m/15m):**
- Enter at optimal point within hour
- Detect momentum before entry
- Expected slippage reduction: ~0.3-0.5%

### Example Scenario

```
1h signal: BUY BTC at 14:00
Price at 14:00: $87,000

Without short TF:
- Execute immediately
- Entry: $87,000
- Price drops to $86,500 by 14:30
- Unrealized loss: -$500

With 5m TF:
- 5m detects downward momentum
- Wait for support
- Entry at 14:25: $86,600
- Price rebounds to $87,200
- Better entry by $400 (0.46% improvement)
```

### Performance Impact

**Estimated improvements:**
- Entry timing: +0.3-0.5% per trade
- Exit timing: +0.3-0.5% per trade
- Overall: +0.6-1.0% improvement on returns
- Reduced drawdowns: ~20-30% smaller pullbacks

## Data Requirements

### Storage

| Timeframe | Data Duration | CSV Size | Processed Size |
|-----------|---------------|----------|----------------|
| 1h        | 1 year        | ~400 MB  | ~1.5 GB        |
| 15m       | 6 months      | ~800 MB  | ~3 GB          |
| 5m        | 3 months      | ~1.2 GB  | ~4.5 GB        |

**Total:** ~10 GB for multi-timeframe setup

### Download Time

- 1h data: Already have ✓
- 15m data: ~2-3 hours
- 5m data: ~4-6 hours

## Monitoring Multi-Timeframe Ensemble

### Dashboard Integration

Add a new page (Page 4) showing:
- Strategic signals (1h ensemble)
- Tactical signals (5m/15m ensembles)
- Alignment score (how much they agree)
- Signal history (last 24 hours)
- Entry timing opportunities (when both align)

### Metrics to Track

1. **Signal Alignment Rate**: How often timeframes agree
2. **Entry Quality**: Price improvement vs naive 1h entry
3. **Exit Quality**: Price improvement vs naive 1h exit
4. **Win Rate**: Percentage of profitable trades
5. **Sharpe Ratio**: Risk-adjusted returns

## Risk Management

### Safeguards

1. **Maximum position size**: 30% per asset (unchanged)
2. **Stop-loss**: 10% from entry (unchanged)
3. **Signal disagreement limit**: Don't trade if timeframes strongly conflict
4. **Rapid reversal protection**: Exit if short TF shows strong reversal
5. **Cooldown after losses**: Wait longer if recent trade was loss

### Conflict Resolution

```python
if strategic_signal == "BUY" and tactical_signal == "SELL":
    # Strong conflict
    if abs(strategic_confidence - tactical_confidence) > 0.3:
        # One is much stronger - trust the stronger signal
        follow_stronger_signal()
    else:
        # Similar confidence - wait for clarity
        no_trade()
```

## Next Steps

### Before 1000 Trials

- [ ] Monitor current 1h training (243 trials remaining)
- [ ] Download and prepare 5m/15m historical data
- [ ] Test data preparation scripts
- [ ] Plan GPU resource allocation

### At 1000 Trials

- [ ] Evaluate 1h ensemble performance
- [ ] Select best 10-20 models from 1000 trials
- [ ] Create updated 1h ensemble
- [ ] Begin 5m timeframe training (highest priority)

### After 5m Training Completes (500 trials)

- [ ] Create initial multi-timeframe ensemble
- [ ] Backtest on historical data
- [ ] Paper trade with multi-TF ensemble
- [ ] Monitor improvements

### Long Term

- [ ] Train 15m models (500 trials)
- [ ] Add 30m if beneficial
- [ ] Optimize signal combination weights
- [ ] A/B test different combination strategies

## Scripts to Create

1. `prepare_multi_timeframe_data.py` - Download and prepare short TF data
2. `train_multi_timeframe.sh` - Launch training for multiple timeframes
3. `create_multi_timeframe_ensemble.py` - Combine models from different TFs
4. `multi_timeframe_ensemble_agent.py` - Agent that uses multiple TFs
5. `backtest_multi_timeframe.py` - Test strategy on historical data

## Questions to Answer Through Experimentation

1. **Optimal tactical timeframe**: Is 5m better than 15m? Or both?
2. **Signal weighting**: What's the optimal weight for strategic vs tactical?
3. **Confirmation delay**: How long to wait for tactical confirmation?
4. **Conflict handling**: Best approach when signals disagree?
5. **Model count per TF**: How many models per timeframe?

---

**Timeline Summary:**
- **Days 0-2**: Reach 1000 trials on 1h (~40 hours remaining)
- **Days 2-3**: Prepare multi-timeframe data
- **Days 3-6**: Train 5m models (500 trials with 24 workers)
- **Days 6-7**: Create and test multi-TF ensemble
- **Days 7-10**: Train 15m models (500 trials with 12 workers)
- **Days 10+**: Optimize and refine

**Ready to start when you hit 1000 trials!**
