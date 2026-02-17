# Migration Plan: 1hr → 5min Timeframe

## Goal
Migrate to 5-minute timeframe trading while maintaining 1hr baseline for comparison.

**Expected Sharpe improvement:** 0.15-0.25 (current) → **0.8-1.5** (5min target)

---

## Current Status

### Running Systems:
✅ **1hr Paper Trader** (Trial #250, 27 hours, -0.00% return)
- Keep running for comparison
- PID: 707256

✅ **1hr Ensemble Training** (19/500 trials, 3.8% complete)
- 5 parallel studies (3 ensemble + 2 FT-Transformer)
- ETA: ~16 hours to completion
- **Decision:** Let it finish, don't halt

⏳ **5min Data Download** (Started)
- Downloading 6 months of 5min bars
- 7 tickers × ~52,000 bars each = ~364,000 total bars
- ETA: ~30-60 minutes
- PID: 861526
- Log: `logs/data_download_5m.log`

---

## Phase 1: Data Preparation ⏳ IN PROGRESS

**Timeline:** 30-60 minutes

### Tasks:
- [x] Start 5min data download (6 months)
- [ ] Wait for download to complete
- [ ] Verify data quality
- [ ] Check preprocessed data size

**Expected output:**
- File: `data/crypto_5m_6mo.pkl`
- Size: ~150-200 MB
- Bars per ticker: ~52,000 (6 months × 30 days × 24 hours × 12 bars/hour)

**Monitor:**
```bash
tail -f logs/data_download_5m.log
```

---

## Phase 2: 5min Training Setup

**Timeline:** When 1hr training completes (~16 hours from now)

### Create 5min Training Script

Modify the existing training setup for 5min:

```bash
# Create start_5min_training.sh
#!/bin/bash
# Launch 5min ensemble + FT training

STUDY_DB="databases/5min_campaign.db"
TIMEFRAME="5m"
N_TRIALS=100

# 1. Ensemble Conservative
nohup python scripts/training/1_optimize_unified.py \
    --study-name ensemble_5m_conservative_$(date +%Y%m%d) \
    --n-trials $N_TRIALS --gpu 0 --timeframe $TIMEFRAME \
    --force-baseline \
    --storage sqlite:///$STUDY_DB \
    > logs/training/ensemble_5m_conservative.log 2>&1 &

# 2. Ensemble Balanced
nohup python scripts/training/1_optimize_unified.py \
    --study-name ensemble_5m_balanced_$(date +%Y%m%d) \
    --n-trials $N_TRIALS --gpu 0 --timeframe $TIMEFRAME \
    --force-baseline \
    --storage sqlite:///$STUDY_DB \
    > logs/training/ensemble_5m_balanced.log 2>&1 &

# 3. Ensemble Aggressive
nohup python scripts/training/1_optimize_unified.py \
    --study-name ensemble_5m_aggressive_$(date +%Y%m%d) \
    --n-trials $N_TRIALS --gpu 0 --timeframe $TIMEFRAME \
    --force-baseline \
    --storage sqlite:///$STUDY_DB \
    > logs/training/ensemble_5m_aggressive.log 2>&1 &

# 4. FT-Transformer Small
nohup python scripts/training/1_optimize_unified.py \
    --study-name ft_5m_small_$(date +%Y%m%d) \
    --n-trials $N_TRIALS --gpu 0 --timeframe $TIMEFRAME \
    --force-ft \
    --storage sqlite:///$STUDY_DB \
    > logs/training/ft_5m_small.log 2>&1 &

# 5. FT-Transformer Large
nohup python scripts/training/1_optimize_unified.py \
    --study-name ft_5m_large_$(date +%Y%m%d) \
    --n-trials $N_TRIALS --gpu 0 --timeframe $TIMEFRAME \
    --force-ft \
    --storage sqlite:///$STUDY_DB \
    > logs/training/ft_5m_large.log 2>&1 &

echo "5min training campaign launched!"
echo "Monitor: python monitor_training.py --db databases/5min_campaign.db"
```

### Key Differences from 1hr Training:

1. **Data size:** 12x larger (5min vs 1hr)
2. **Training time:** ~2-3x longer per trial (more data to process)
3. **Total campaign time:** ~36-48 hours (vs ~24 hours for 1hr)
4. **VRAM usage:** Same (model architecture unchanged)

---

## Phase 3: Parallel Training Execution

**Timeline:** Tomorrow (after 1hr training completes)

### Option A: Sequential (Conservative)
Wait for 1hr training to finish, then start 5min training.

**Pros:**
- No GPU contention
- Full resources for 5min training

**Cons:**
- Delays 5min results by ~16 hours

**Total timeline:** 16 hours (wait) + 48 hours (5min training) = **64 hours to 5min model**

---

### Option B: Parallel (Aggressive) ⭐ RECOMMENDED

Start 5min training alongside 1hr training (they'll time-multiplex GPU).

**Pros:**
- Faster overall timeline
- Get both results sooner

**Cons:**
- Slightly slower per-trial training (GPU sharing)
- Need to monitor GPU memory

**GPU capacity check:**
- RTX 3070: 8GB VRAM
- Current load: 5 workers (1hr training)
- Additional: 5 workers (5min training)
- **Total:** 10 workers, but they're sequential within each study
- **Risk:** Low - GPU will time-multiplex

**Total timeline:** 48 hours (parallel) = **48 hours to both models**

---

## Phase 4: Dual Paper Trading Deployment

**Timeline:** When best 5min model is found (~2-3 days from now)

### Deploy Second Paper Trader (5min)

```bash
# Deploy 5min model alongside 1hr model
python scripts/deployment/deploy_paper_trader.py \
    --model-dir train_results/best_5min_trial \
    --timeframe 5m \
    --session-name trial_5min_live \
    --poll-interval 300  # 5 minutes = 300 seconds
```

### Both Traders Running Simultaneously:

| Trader | Timeframe | Poll Interval | Status | Purpose |
|--------|-----------|---------------|--------|---------|
| Trial #250 | 1hr | 3600s (1hr) | ✅ Running | Baseline comparison |
| Best 5min | 5min | 300s (5min) | 🔜 Pending | New high-frequency model |

### Comparison Metrics:

After 1 week (168 hours), compare:

```python
# Comparison dashboard
import pandas as pd

# Load both traders' data
trader_1h = pd.read_csv('paper_trades/trial250_session.csv')
trader_5m = pd.read_csv('paper_trades/trial_5min_session.csv')

# Calculate metrics
def analyze_trader(df, name):
    returns = df['total_asset'].pct_change()
    sharpe = returns.mean() / returns.std() * np.sqrt(365 * 24)  # Annualized
    total_return = (df['total_asset'].iloc[-1] / df['total_asset'].iloc[0] - 1) * 100

    print(f"\n{name}:")
    print(f"  Total Return: {total_return:.2f}%")
    print(f"  Sharpe Ratio: {sharpe:.3f}")
    print(f"  Max Drawdown: {(df['total_asset'] / df['total_asset'].cummax() - 1).min() * 100:.2f}%")
    print(f"  Trades: {len(df)}")

analyze_trader(trader_1h, "1hr Trader (Trial #250)")
analyze_trader(trader_5m, "5min Trader (New)")
```

---

## Phase 5: Go-Live Decision

**Timeline:** After 1 week of dual paper trading

### Decision Criteria:

| Metric | 1hr Target | 5min Target | Weight |
|--------|-----------|-------------|--------|
| **Sharpe Ratio** | 0.5+ | 1.0+ | 40% |
| **Total Return** | 5%+ | 10%+ | 30% |
| **Max Drawdown** | <10% | <10% | 20% |
| **Consistency** | Stable | Stable | 10% |

**If 5min wins:** Switch live trading to 5min model
**If 1hr wins:** Investigate why 5min underperformed, iterate
**If both good:** Run both in parallel with split capital

---

## Expected Outcomes

### Conservative Estimate:

| Timeframe | Sharpe | 12-Month Return | Comments |
|-----------|--------|-----------------|----------|
| 1hr (current) | 0.5-1.0 | +20-40% | Current baseline |
| 5min (target) | 0.8-1.5 | +40-80% | 12x more opportunities |

### Optimistic Estimate:

| Timeframe | Sharpe | 12-Month Return | Comments |
|-----------|--------|-----------------|----------|
| 1hr (ensemble) | 0.8-1.2 | +30-50% | With ensemble improvement |
| 5min (ensemble) | 1.5-2.0 | +80-120% | With ensemble + higher freq |

### Reality Check:

**Most likely:** 5min model achieves Sharpe 1.0-1.5 (vs current 0.15-0.25)

**Why?**
- More trading opportunities (12x)
- Tighter stop-losses (faster exits)
- Better risk-adjusted returns (smaller positions)
- Exploit intra-hour inefficiencies

---

## Risk Mitigation

### Potential Issues:

1. **5min data too noisy**
   - Mitigation: Use longer lookback periods, more denoising
   - Fallback: Try 15min timeframe instead

2. **Training takes too long**
   - Mitigation: Reduce trial count from 100 to 50
   - Fallback: Use transfer learning from 1hr models

3. **Model doesn't improve on 5min**
   - Mitigation: Analyze why, adjust features
   - Fallback: Stay with 1hr + better features

4. **API rate limits on 5min polling**
   - Mitigation: Alpaca allows frequent polling
   - Fallback: Use websocket streaming instead

---

## Rollback Plan

If 5min doesn't work out:

1. **Stop 5min training** (don't waste compute)
2. **Keep 1hr trader running**
3. **Try 15min timeframe** (compromise between 1hr and 5min)
4. **Focus on improving 1hr model** with better features

---

## Next Steps (Action Items)

### Immediate (Today):
- [x] Start 5min data download ✅ RUNNING
- [ ] Monitor download progress
- [ ] Verify data quality when complete

### Tomorrow (After 1hr training completes):
- [ ] Review 1hr ensemble results
- [ ] Deploy best 1hr ensemble model (if better than Trial #250)
- [ ] Start 5min training campaign (5 parallel studies)

### Day 3-4:
- [ ] Monitor 5min training progress
- [ ] Check for early strong performers
- [ ] Consider early stopping if a clear winner emerges

### Day 5-7:
- [ ] 5min training completes
- [ ] Deploy best 5min model to paper trading
- [ ] Run dual paper traders (1hr + 5min) in parallel

### Week 2-3:
- [ ] Compare 1hr vs 5min live performance
- [ ] Make go-live decision
- [ ] Scale up capital if performance good

---

## Success Metrics

### Data Download (Phase 1):
✅ Successfully downloaded 6 months of 5min data for 7 tickers
✅ ~364,000 bars total (52,000 per ticker)
✅ All technical indicators calculated
✅ Data saved to `data/crypto_5m_6mo.pkl`

### Training (Phase 2-3):
✅ 5min ensemble training completes successfully
✅ Best trial achieves Sharpe > 0.8 in backtest
✅ At least one model shows clear improvement over 1hr

### Paper Trading (Phase 4):
✅ 5min paper trader runs for 168 hours (1 week)
✅ Achieves Sharpe > 1.0 in live trading
✅ Outperforms 1hr trader on key metrics

### Go-Live (Phase 5):
✅ Confident to deploy 5min model to live account
✅ Risk management validated on 5min timeframe
✅ Ready to scale up capital

---

## Timeline Summary

| Phase | Duration | Start | End | Status |
|-------|----------|-------|-----|--------|
| Data Download | 1 hour | Now | +1h | ⏳ In Progress |
| Wait for 1hr Training | 16 hours | +1h | +17h | 🔜 Queued |
| 5min Training | 48 hours | +17h | +65h | 🔜 Queued |
| Paper Trading | 168 hours | +65h | +233h | 🔜 Queued |
| Go-Live Decision | - | +233h | - | 🔜 Queued |

**Total: ~10 days from now to 5min live trading**

---

## Budget

### Compute:
- Data download: ~1 hour (negligible)
- Training: ~48 hours GPU time
- Paper trading: Free (Alpaca paper account)

### Costs:
- Electricity: ~$5-10 (GPU power)
- API calls: $0 (Alpaca free tier)
- **Total: ~$10**

---

## Questions?

**Q: Will 5min training work on my RTX 3070?**
A: Yes! Same VRAM usage, just takes longer (2-3x) per trial.

**Q: Can I run both trainings in parallel?**
A: Yes, GPU will time-multiplex. Slightly slower but faster overall.

**Q: What if 5min is worse than 1hr?**
A: Keep 1hr, try 15min, or improve features. Not a loss—you learned something.

**Q: How do I monitor progress?**
A: Discord notifications + `python monitor_training.py --db databases/5min_campaign.db`

---

**Last updated:** 2026-02-09 (Migration started)
