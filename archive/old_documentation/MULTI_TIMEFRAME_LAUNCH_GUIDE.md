## Multi-Timeframe Trading Launch Guide

Complete guide to implementing multi-timeframe ensemble trading once you reach 1000 trials.

---

## ✅ Prerequisites Checklist

Before starting, ensure you have:

- [ ] Reached 1000 completed trials on 1h timeframe
- [ ] At least 80-100 models in top 10%
- [ ] Stable 1h ensemble performing well in paper trading
- [ ] ~200 GB free disk space for data
- [ ] GPU available for accelerated training

**Current Status:**
```bash
# Check progress
sqlite3 databases/optuna_cappuccino.db \
  "SELECT COUNT(*) FROM trials WHERE state='COMPLETE' AND study_id=(SELECT study_id FROM studies WHERE study_name='cappuccino_1year_20251121')"
```

---

## 📋 Step-by-Step Implementation

### Phase 1: Data Preparation (Day 1)

#### 1.1 Download 5-Minute Data (4-6 hours)

```bash
# Download 3 months of 5m data
python prepare_multi_timeframe_data.py --timeframe 5m --months 3

# Check output
ls -lh data/crypto_5m_3mo.pkl
cat data/crypto_5m_3mo_summary.txt
```

**Expected:**
- File size: ~1-2 GB
- Data points: ~30,000-40,000 per ticker
- Duration: 4-6 hours

#### 1.2 Download 15-Minute Data (Optional, 2-3 hours)

```bash
# Download 6 months of 15m data
python prepare_multi_timeframe_data.py --timeframe 15m --months 6

# Check output
ls -lh data/crypto_15m_6mo.pkl
```

**Expected:**
- File size: ~800 MB - 1.5 GB
- Data points: ~17,000-25,000 per ticker
- Duration: 2-3 hours

---

### Phase 2: Train 5m Models (Days 2-4)

#### 2.1 Launch 5m Training

```bash
# Start with 24 workers (adjust based on GPU capacity)
./train_multi_timeframe.sh --timeframe 5m --workers 24 --trials 500
```

**Configuration:**
- Workers: 24 (adjust to 12-36 based on GPU)
- Trials per worker: 500
- Total trials: 500 (each worker contributes)
- Expected duration: 48-72 hours

#### 2.2 Monitor Training

```bash
# Watch worker count
watch -n 10 'pgrep -f "study-name cappuccino_5m" | wc -l'

# Monitor database
python dashboard_training_detailed.py --study cappuccino_5m_*

# Check logs
tail -f logs/worker_*_5m.log
```

#### 2.3 Wait for Completion

**Progress milestones:**
- 100 trials: ~10 hours (early convergence visible)
- 250 trials: ~24 hours (good coverage of hyperparameter space)
- 500 trials: ~48-72 hours (full training complete)

---

### Phase 3: Create 5m Ensemble (Day 4-5)

#### 3.1 Analyze Results

```bash
# Get study name
sqlite3 databases/optuna_cappuccino.db "SELECT study_name FROM studies WHERE study_name LIKE 'cappuccino_5m%' ORDER BY study_id DESC LIMIT 1"

# Analyze training
python analyze_training_results.py --study <study_name>
```

#### 3.2 Create Ensemble

```bash
# Create ensemble from top 10 models
python create_simple_ensemble.py \
    --study <5m_study_name> \
    --output-dir train_results/ensemble_5m \
    --top-n 10

# Verify
ls -la train_results/ensemble_5m/
cat train_results/ensemble_5m/ensemble_manifest.json
```

---

### Phase 4: Create Multi-Timeframe Ensemble (Day 5)

#### 4.1 Test Individual Ensembles

```bash
# Test 1h ensemble
python ultra_simple_ensemble.py

# Test 5m ensemble
python ultra_simple_ensemble.py --manifest train_results/ensemble_5m/ensemble_manifest.json
```

#### 4.2 Create Multi-TF Ensemble

```bash
# Combine 1h + 5m
python multi_timeframe_ensemble_agent.py
```

**This will:**
- Load 1h strategic ensemble (10 models)
- Load 5m tactical ensemble (10 models)
- Create combined multi-TF agent

---

### Phase 5: Paper Trading with Multi-TF (Day 5-6)

#### 5.1 Stop Current Paper Trading

```bash
# Stop existing paper trader
pkill -f paper_trader_alpaca_polling.py

# Verify stopped
ps aux | grep paper_trader
```

#### 5.2 Update Ensemble Configuration

```bash
# The multi-TF agent will automatically use both ensembles
# No changes needed - just ensure both ensemble directories exist:
ls train_results/ensemble/       # 1h strategic
ls train_results/ensemble_5m/    # 5m tactical
```

#### 5.3 Launch Multi-TF Paper Trading

```bash
# Start paper trader with multi-TF support
# (Note: This requires modifications to paper_trader_alpaca_polling.py
#  to use MultiTimeframeEnsemble instead of UltraSimpleEnsemble)

# For now, test the 5m ensemble alone first:
python paper_trader_alpaca_polling.py \
    --model-dir train_results/ensemble_5m \
    --tickers BTC/USD ETH/USD LTC/USD BCH/USD LINK/USD UNI/USD AAVE/USD \
    --timeframe 5m \
    --history-hours 24 \
    --poll-interval 60 \
    --gpu -1
```

---

### Phase 6: Monitor and Optimize (Ongoing)

#### 6.1 Compare Performance

**Metrics to track:**
- Entry quality (price improvement vs 1h-only)
- Win rate
- Sharpe ratio
- Drawdown reduction

**Dashboard:**
```bash
# View portfolio history
python dashboard.py
# Navigate to Page 3: Portfolio History
```

#### 6.2 A/B Testing

Run parallel paper trading:
- Account 1: 1h-only ensemble
- Account 2: Multi-TF ensemble

Compare after 1-2 weeks.

---

## 🎯 Expected Improvements

### Entry Timing

**Scenario: BTC Buy Signal**

**1h-only:**
```
Signal at 14:00
Price: $87,000
Execute immediately
Entry: $87,000
```

**Multi-TF (1h + 5m):**
```
Strategic (1h) at 14:00: BUY
  Price: $87,000
  Tactical (5m) checks momentum
  14:05: Downward momentum detected → WAIT
  14:15: Support found → ENTER
  Entry: $86,650

Improvement: +$350 (0.40%)
```

### Realistic Improvements

Based on similar strategies:
- **Entry timing**: +0.3-0.5% per trade
- **Exit timing**: +0.3-0.5% per trade
- **Overall returns**: +0.6-1.0% improvement
- **Drawdown reduction**: 20-30% smaller peaks
- **Win rate**: +2-5% more winning trades

---

## 🔧 Advanced: Training 15m Models

Once 5m is working well:

```bash
# Download 15m data
python prepare_multi_timeframe_data.py --timeframe 15m --months 6

# Train 15m models (12 workers)
./train_multi_timeframe.sh --timeframe 15m --workers 12 --trials 500

# Create 15m ensemble
python create_simple_ensemble.py \
    --study <15m_study_name> \
    --output-dir train_results/ensemble_15m \
    --top-n 10

# Use all three timeframes
python multi_timeframe_ensemble_agent.py \
    --strategic-dir train_results/ensemble \
    --tactical-dirs train_results/ensemble_5m,train_results/ensemble_15m
```

---

## 🚨 Troubleshooting

### Issue: Training Too Slow

**Solutions:**
- Reduce workers (24 → 12)
- Use smaller data duration (3mo → 2mo)
- Check GPU utilization: `nvidia-smi`
- Ensure no other processes using GPU

### Issue: Models Not Converging

**Check:**
- Data quality: `head data/crypto_5m_3mo.pkl`
- Hyperparameter ranges in training script
- Trial failures: Check logs for errors

### Issue: Multi-TF Signals Conflicting

**Strategies:**
- Increase `confidence_threshold`
- Change strategy to 'gating' (more conservative)
- Adjust weights (more weight to strategic)

### Issue: Out of Memory

**Solutions:**
- Reduce chunk size in training
- Train fewer workers simultaneously
- Use CPU for some workers: `--gpu-id -1`

---

## 📊 Monitoring Checklist

Daily checks:
- [ ] All workers still running
- [ ] GPU utilization healthy (>80%)
- [ ] Disk space sufficient
- [ ] No errors in logs
- [ ] Trials progressing (check database)

Weekly checks:
- [ ] Compare multi-TF vs 1h-only performance
- [ ] Review signal alignment rates
- [ ] Check entry/exit quality metrics
- [ ] Analyze drawdowns and recoveries

---

## 🎓 Learning Resources

Key concepts to understand:
1. **Multi-timeframe analysis**: Using different timeframes for different purposes
2. **Signal alignment**: When/why timeframes agree or conflict
3. **Momentum detection**: How short TF models detect momentum
4. **Position sizing**: Adjusting size based on confidence
5. **Risk management**: Handling conflicting signals

---

## 📈 Success Criteria

After 2 weeks of multi-TF trading:

**Minimum Success:**
- [ ] No significant degradation vs 1h-only
- [ ] System runs stably
- [ ] No major technical issues

**Good Success:**
- [ ] +0.5-1.0% improvement in returns
- [ ] Lower drawdowns
- [ ] Better entry timing visible in charts

**Excellent Success:**
- [ ] +1.0-2.0% improvement in returns
- [ ] 20-30% drawdown reduction
- [ ] Higher Sharpe ratio
- [ ] Consistent outperformance

---

## 🚀 Quick Start Commands

```bash
# 1. Prepare 5m data
python prepare_multi_timeframe_data.py --timeframe 5m --months 3

# 2. Train 5m models
./train_multi_timeframe.sh --timeframe 5m --workers 24 --trials 500

# 3. Create ensemble (after training completes)
STUDY=$(sqlite3 databases/optuna_cappuccino.db "SELECT study_name FROM studies WHERE study_name LIKE 'cappuccino_5m%' ORDER BY study_id DESC LIMIT 1")
python create_simple_ensemble.py --study $STUDY --output-dir train_results/ensemble_5m

# 4. Test ensemble
python multi_timeframe_ensemble_agent.py

# 5. Monitor
python dashboard.py
```

---

## 📝 Notes

- Start with 5m only, add 15m/30m later
- Keep 1h training running (1-2 workers for continuous improvement)
- Back up models regularly
- Document any configuration changes
- Compare performance objectively with metrics

---

**Timeline Summary:**
- **Day 1**: Download 5m data (4-6 hours)
- **Days 2-4**: Train 5m models (48-72 hours)
- **Day 4-5**: Create ensembles and test (4-8 hours)
- **Day 5**: Deploy to paper trading (1 hour)
- **Days 5-19**: Monitor and compare (2 weeks)
- **Day 19+**: Optimize or expand to 15m

**You're currently at 757/1000 trials. See you at 1000! 🚀**
