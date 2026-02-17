# Complete Process: CGE Augmentation → Production Deployment

## Overview

This guide walks through the entire pipeline from using CGE-augmented data to deploying improved models to production.

---

## 📋 PHASE 1: DATA PREPARATION (✅ COMPLETE)

**Status:** ✅ Done - Ready to use

### What Was Done
1. Generated 200 CGE economic scenarios (normal, bear, crisis markets)
2. Created synthetic crypto price data from macro conditions
3. Normalized prices to realistic ranges
4. Mixed 70% real + 30% synthetic bear market data
5. Shuffled and saved augmented dataset

### Current State
```
/opt/user-data/experiment/cappuccino/data/1h_cge_augmented/
├── price_array    (8,607 timesteps × 7 assets)
├── tech_array     (8,607 timesteps × 77 features)
└── time_array     (8,607 timesteps)
```

### Data Composition
- **Real data:** 6,025 timesteps (70%) from Alpaca historical
- **Synthetic:** 2,582 timesteps (30%) from 60 bear market CGE scenarios
- **Total:** 8,607 timesteps for training

---

## 🎯 PHASE 2: MODEL TRAINING (⏭️ NEXT)

**Estimated Time:** 4-12 hours depending on hyperparameter search

### Step 2.1: Setup Training Data

**Option A: Quick (Recommended)**
```bash
cd /opt/user-data/experiment/cappuccino

# Create symlink to use augmented data
ln -sf data/1h_cge_augmented data/1h_1680

# Verify
ls -lh data/1h_1680
```

**Option B: Explicit Path**
Modify training script to load from `./data/1h_cge_augmented`

### Step 2.2: Run Training

**Basic Training (Single trial for testing):**
```bash
# Quick test to verify everything works
python3 1_optimize_unified.py --n-trials 1 --gpu 0
```

**Full Hyperparameter Optimization:**
```bash
# Recommended: 100-150 trials for good optimization
python3 1_optimize_unified.py --n-trials 100 --gpu 0

# With best known ranges (faster convergence)
python3 1_optimize_unified.py --use-best-ranges --n-trials 50 --gpu 0

# Multi-timeframe (if you want to optimize across timeframes)
python3 1_optimize_unified.py --mode multi-timeframe --n-trials 150
```

**Training Process:**
1. **Optuna** performs hyperparameter search
2. Each trial trains a PPO agent with different hyperparameters
3. Models evaluated using CombPurgedKFoldCV (prevents data leakage)
4. Best trials saved to database: `databases/optuna_cappuccino.db`
5. Trial results include: Sharpe ratio, returns, max drawdown, win rate

**What to Monitor:**
```bash
# Watch training progress
tail -f *.log

# Check GPU usage
nvidia-smi -l 1

# Monitor study progress (if you have dashboard running)
python3 dashboard_optimized.py
```

### Step 2.3: Training Outputs

**Location:** `databases/optuna_cappuccino.db`
- SQLite database with all trial results
- Hyperparameters for each trial
- Performance metrics (Sharpe, return, drawdown, etc.)
- Best trial automatically tracked

**Model Files:**
- Saved in trial directories during training
- Best models identified by Optuna based on Sharpe ratio

---

## 📊 PHASE 3: MODEL EVALUATION (⏭️ AFTER TRAINING)

**Estimated Time:** 30 minutes - 2 hours

### Step 3.1: Analyze Training Results

```bash
# View best trials
python3 analyze_training.py

# Detailed analysis
python3 analyze_training_results.py --study cappuccino_week_20260118
```

**Key Metrics to Compare:**

| Metric | Baseline (Old) | With CGE (New) | Target Improvement |
|--------|----------------|----------------|-------------------|
| Overall Sharpe | 11.5 | ? | 13-14 (+13-22%) |
| Bear Market Sharpe | 4.3 | ? | 5.5-6.5 (+28-51%) |
| Max Drawdown | -22% | ? | -15-18% (improvement) |
| Win Rate | ~60% | ? | 65%+ |

### Step 3.2: Stress Testing

Run the new models through CGE stress scenarios:

```bash
cd /home/mrc/gempack_install

# Update stress test to use new models
python3 cappuccino_stress_test.py
```

**What to Look For:**
- ✅ Better performance in bear market scenarios
- ✅ Similar/improved performance in normal markets
- ✅ Lower worst-case Sharpe ratio
- ✅ More consistent performance across regimes

### Step 3.3: Comparison Analysis

Create comparison report:
```bash
# Compare old vs new model performance
# You may need to create a comparison script or do manual analysis
```

**Decision Criteria:**
- ✅ New model Sharpe > old model Sharpe (or within 5% with better risk metrics)
- ✅ Bear market performance improved by >20%
- ✅ Max drawdown reduced
- ✅ No major degradation in any regime

---

## 🏟️ PHASE 4: ARENA TESTING (⏭️ OPTIONAL BUT RECOMMENDED)

**Estimated Time:** 1-7 days

Arena is your competitive model evaluation system where models trade against each other.

### Step 4.1: Deploy to Arena

```bash
# Deploy new CGE-trained models to arena
python3 deploy_to_arena.py --trial-number <best_trial_number>

# Or deploy multiple top trials
python3 deploy_to_arena.py --top-n 10
```

### Step 4.2: Run Arena Competition

```bash
# Run arena with both old and new models
python3 arena_runner.py --duration 7d
```

**Arena Process:**
- Multiple models trade simultaneously
- Performance tracked in real-time
- Models compete on same market data
- Weak models gradually pruned
- Best performers identified

### Step 4.3: Analyze Arena Results

```bash
# View arena performance
python3 analyze_arena_trades.py

# Compare old vs new models
python3 model_arena.py --compare
```

**Arena Metrics:**
- Sharpe ratio
- Total returns
- Drawdowns
- Trade frequency
- Risk-adjusted returns
- Head-to-head performance

---

## 🚀 PHASE 5: PAPER TRADING (⏭️ BEFORE LIVE)

**Estimated Time:** 1-4 weeks recommended

### Step 5.1: Deploy Best Model(s)

**Automatic Deployment:**
```bash
# Auto-deploy best model from study
python3 auto_model_deployer.py \
    --study cappuccino_week_20260118 \
    --auto-deploy \
    --validation-enabled
```

**Manual Deployment:**
```bash
# Find best trial
python3 -c "
import optuna
storage = 'sqlite:///databases/optuna_cappuccino.db'
study = optuna.load_study(study_name='cappuccino_week_20260118', storage=storage)
print(f'Best trial: {study.best_trial.number}')
print(f'Best Sharpe: {study.best_value}')
"

# Deploy specific trial
python3 auto_model_deployer.py --trial <trial_number> --deploy-now
```

### Step 5.2: Start Paper Trading

```bash
# Start paper trader with Alpaca
python3 paper_trader_alpaca_polling.py
```

**Paper Trading Settings:**
- Uses Alpaca paper trading account (not real money)
- Real market data, simulated execution
- Tracks all metrics as if live
- No financial risk

### Step 5.3: Monitor Paper Trading

```bash
# Real-time dashboard
python3 paper_trading_dashboard.py

# Analyze paper trading results
python3 analyze_paper_trading.py

# Check paper trader status
tail -f logs/paper_trader.log
```

**Monitoring Checklist:**
- [ ] Trade execution working correctly
- [ ] Position sizing appropriate
- [ ] Stop losses triggering properly
- [ ] Returns tracking expectations
- [ ] No execution errors
- [ ] Drawdowns within acceptable limits

### Step 5.4: Paper Trading Evaluation

**Minimum Duration:** 1-2 weeks recommended
**Ideal Duration:** 4 weeks (captures different market conditions)

**Success Criteria:**
- ✅ Sharpe ratio > 1.5 (paper trading)
- ✅ Max drawdown < -25%
- ✅ Win rate > 55%
- ✅ No critical errors
- ✅ Consistent with backtest results (within 20%)
- ✅ Better bear market performance than old model

---

## 💰 PHASE 6: LIVE DEPLOYMENT (⏭️ PRODUCTION)

**⚠️ ONLY after successful paper trading**

### Step 6.1: Pre-Live Checklist

- [ ] Paper trading successful for 2+ weeks
- [ ] Performance meets targets
- [ ] Risk management validated
- [ ] All error handling tested
- [ ] Monitoring systems working
- [ ] Backup/rollback plan ready
- [ ] Position limits configured
- [ ] API keys for live account secured

### Step 6.2: Deploy to Live

```bash
# Switch to live trading (requires live API keys)
# Update config to use live Alpaca account
# Start with SMALL position sizes

python3 coinbase_live_trader.py  # or paper_trader_alpaca_polling.py with live keys
```

**Initial Live Settings:**
- Start with 10-25% of planned capital
- Use conservative position sizing
- Tighter stop losses initially
- Monitor 24/7 for first week

### Step 6.3: Live Monitoring

```bash
# Continuous monitoring
python3 paper_trading_dashboard.py  # Works for live too

# Automated watchdog (monitors for issues)
python3 auto_model_deployer.py --daemon --validation-enabled
```

**Live Monitoring:**
- Real-time P&L tracking
- Position monitoring
- Error detection
- Performance vs paper trading
- Market condition changes

### Step 6.4: Scale Up (Gradual)

**Week 1:** 10-25% capital
**Week 2-3:** If successful, 25-50% capital
**Week 4+:** If successful, 50-100% capital

**Never scale up if:**
- Drawdown exceeds expectations
- Errors occurring
- Performance significantly below paper trading
- Market conditions unusual

---

## 📈 PHASE 7: ONGOING OPTIMIZATION (CONTINUOUS)

### Step 7.1: Regular Retraining

**Monthly:**
```bash
# Generate fresh CGE scenarios with latest macro data
cd /home/mrc/gempack_install
python3 cappuccino_cge_integration.py

# Augment with new data
cd /opt/user-data/experiment/cappuccino
python3 augment_with_cge.py

# Retrain
python3 1_optimize_unified.py --n-trials 50
```

### Step 7.2: A/B Testing

Run multiple model versions simultaneously:
- Old model (baseline)
- New CGE-trained model
- Latest retrained model

Compare performance over time.

### Step 7.3: Performance Tracking

```bash
# Weekly analysis
python3 analyze_paper_trading.py --last-7-days

# Monthly reports
python3 trade_history_analyzer.py --monthly-report
```

---

## 🔧 QUICK REFERENCE COMMANDS

### Training
```bash
# Quick test
python3 1_optimize_unified.py --n-trials 1

# Full optimization
python3 1_optimize_unified.py --n-trials 100 --gpu 0

# With best ranges (faster)
python3 1_optimize_unified.py --use-best-ranges --n-trials 50
```

### Evaluation
```bash
# Stress test
cd /home/mrc/gempack_install && python3 cappuccino_stress_test.py

# Analyze results
cd /opt/user-data/experiment/cappuccino
python3 analyze_training_results.py
```

### Deployment
```bash
# Auto-deploy best model
python3 auto_model_deployer.py --auto-deploy

# Paper trading
python3 paper_trader_alpaca_polling.py

# Monitor
python3 paper_trading_dashboard.py
```

### Data Regeneration
```bash
# Regenerate CGE augmented data
rm -rf data/1h_cge_augmented
python3 augment_with_cge.py

# Different mix
# Edit REAL_RATIO in augment_with_cge.py, then:
python3 augment_with_cge.py
```

---

## ⚠️ IMPORTANT NOTES

### Risk Management
- **Never skip paper trading**
- Start live with small capital
- Set hard stop losses
- Monitor continuously
- Have rollback plan ready

### Expected Timeline
- **Training:** 4-12 hours
- **Evaluation:** 1-2 days
- **Paper Trading:** 2-4 weeks (minimum)
- **Live Deployment:** Gradual over 4+ weeks
- **Total:** 6-10 weeks to full production

### Success Indicators
- ✅ Training completes without errors
- ✅ New model Sharpe > baseline
- ✅ Bear market Sharpe improved 20%+
- ✅ Paper trading profitable
- ✅ Live performance matches paper

### Failure Indicators (Stop and Reassess)
- ❌ Training diverges or fails
- ❌ New model worse than baseline
- ❌ Paper trading losing money
- ❌ Large discrepancy between backtest and paper
- ❌ Unusual errors or behavior

---

## 📞 TROUBLESHOOTING

### Training Issues
**Problem:** Training very slow
- Solution: Check GPU usage, reduce n_trials, use --use-best-ranges

**Problem:** Models not improving
- Solution: Check data quality, try different hyperparameter ranges

**Problem:** Out of memory
- Solution: Reduce batch size, use smaller network, check GPU memory

### Paper Trading Issues
**Problem:** Orders not executing
- Solution: Check API keys, network connection, Alpaca status

**Problem:** Performance much worse than backtest
- Solution: Check slippage assumptions, execution delays, data quality

**Problem:** Unexpected drawdowns
- Solution: Check market conditions, verify risk management, review trades

---

## 🎯 CURRENT STATUS

```
✅ Phase 1: Data Preparation - COMPLETE
⏭️ Phase 2: Model Training - READY TO START
⏭️ Phase 3: Evaluation - Pending
⏭️ Phase 4: Arena Testing - Pending (Optional)
⏭️ Phase 5: Paper Trading - Pending
⏭️ Phase 6: Live Deployment - Pending
⏭️ Phase 7: Ongoing Optimization - Pending
```

---

## 🚀 NEXT IMMEDIATE STEPS

1. **Create symlink for augmented data:**
   ```bash
   cd /opt/user-data/experiment/cappuccino
   ln -sf data/1h_cge_augmented data/1h_1680
   ```

2. **Start training:**
   ```bash
   # Quick test first
   python3 1_optimize_unified.py --n-trials 1 --gpu 0

   # If successful, full training
   python3 1_optimize_unified.py --n-trials 100 --gpu 0
   ```

3. **Monitor progress and wait for completion (4-12 hours)**

4. **Evaluate results and compare to baseline**

5. **Deploy to paper trading if improved**

---

**Ready to start Phase 2 (Training)!**
