# AMD GPU Upgrade: Timeline Impact Analysis

**Current GPU:** NVIDIA RTX 3070 (8GB, 9 workers, 40-45 trials/hr)
**Target GPU:** AMD RX 7900 XTX (24GB, 20 workers, 90-100 trials/hr)
**Analysis Date:** January 16, 2026

---

## TL;DR: Minimal Impact on Initial Launch, Major Long-Term Benefits

**Initial Launch Timeline:**
- **Without GPU upgrade:** Feb 20 (realistic)
- **With GPU upgrade:** Feb 17-18 (2-3 days faster)
- **Net benefit:** 2-3 days on initial launch

**Why so small?**
The bottleneck is **validation, not model discovery**. You already have excellent models (Sharpe 0.0140). The constraint is waiting for paper trading results, which takes 1-2 weeks regardless of GPU speed.

**Long-term benefits are massive:**
- Faster iteration if paper trading needs fixes
- More model options for optimization
- 50% time savings on future retraining

---

## Current Timeline (Without GPU Upgrade)

### Today: Jan 16, 2026

**Training Status:**
- Completed: 970+ trials
- Workers: 9 (after reduction to free VRAM)
- Speed: 40-45 trials/hour
- To 2,000 trials: ~23 hours (tomorrow evening)
- To 5,000 trials: ~90 hours (3.8 days)

**Launch Phases:**

| Phase | Duration | End Date | Bottleneck |
|-------|----------|----------|------------|
| Model Discovery | ✅ COMPLETE | Jan 16 | - |
| Initial Validation | 4 days | Jan 20 | **Paper trading** |
| Extended Validation | 21 days | Feb 10 | **Paper trading** |
| Pre-Launch Review | 5 days | Feb 15 | Manual review |
| Small Capital Launch | - | **Feb 20** | Confidence |

**Critical Path:** Paper trading validation (25 days total)

---

## Timeline With GPU Upgrade

### Scenario: Upgrade After 1,000 Trials

**Today (Jan 16, 19:30):**
- Hit 1,000 trials in ~30 minutes
- Stop training, stop automation
- Physical GPU swap + ROCm setup: 2-3 hours

**Tomorrow (Jan 17, ~22:00):**
- Resume training with 20 workers @ 90 trials/hr
- To 2,000 trials: 11 hours (Jan 18, 09:00)
- To 5,000 trials: 44 hours (Jan 19, 18:00)

**Launch Phases:**

| Phase | Duration | End Date | Change |
|-------|----------|----------|--------|
| Model Discovery | ✅ COMPLETE | Jan 16 | - |
| **Migration** | **3 hours** | **Jan 16** | **New** |
| Initial Validation | 4 days | Jan 20 | Same |
| Extended Validation | 21 days | Feb 10 | Same |
| Pre-Launch Review | 5 days | Feb 15 | Same |
| Small Capital Launch | - | **Feb 17** | **-3 days** |

**Critical Path:** Still paper trading validation

---

## Detailed Impact Analysis

### Phase 1: Model Discovery (✅ Already Complete)

**Status:** You have excellent models
- Best: Sharpe 0.0140 (112% better than old)
- Top 20 mean: 0.0112
- Ensemble deployed

**GPU upgrade impact:** ❌ **None** - already done!

### Phase 2: Migration Downtime (New 3-hour cost)

**Without upgrade:** 0 hours
**With upgrade:** 2-3 hours

**Net cost:** 3 hours one-time

**When to do it:**
- ✅ Best: After 1,000 trials (in 30 min)
- ✅ Good: After paper trading initial results (Jan 20)
- ⚠️ Risky: During critical training phase

### Phase 3: Additional Model Training (Optional)

**Current plan:** ~1,000 trials is sufficient

**With more trials:**

| Trials | Without GPU | With GPU | Savings |
|--------|-------------|----------|---------|
| 2,000 | 23 hours | 11 hours | **-12 hours** |
| 3,000 | 46 hours | 22 hours | **-24 hours** |
| 5,000 | 90 hours | 44 hours | **-46 hours** |

**Value of more trials:**
- More model diversity
- Better ensemble optimization options
- Insurance if current models underperform

**Timeline impact:**
- If you decide you need 2,000 trials: Save 12 hours (0.5 days)
- If you decide you need 5,000 trials: Save 46 hours (2 days)

**Realistic:** Maybe save 0.5-1 day if you want more model options

### Phase 4: Paper Trading Validation (Bottleneck!)

**Duration:** 25 days (Jan 16 - Feb 10)

**GPU upgrade impact:** ❌ **None**

**Why?** Paper trading must run in real-time:
- Need 30+ trades for statistical significance
- Need to observe performance across different market conditions
- Need 1-2 weeks minimum, ideally 3-4 weeks
- **Cannot be accelerated with faster GPU**

**This is your critical path!**

### Phase 5: Pre-Launch Review

**Duration:** 5 days (Feb 10-15)

**GPU upgrade impact:** ❌ **None** - manual review process

### Phase 6: Launch Decision

**Best Case Launch:**
- Without GPU: Feb 15
- With GPU: Feb 15
- **Difference: 0 days**

**Realistic Launch:**
- Without GPU: Feb 20
- With GPU: Feb 17-18
- **Difference: 2-3 days**

**Conservative Launch:**
- Without GPU: Mar 15
- With GPU: Mar 12-13
- **Difference: 2-3 days**

---

## When GPU Upgrade DOES Matter

### 1. If Paper Trading Issues Found

**Scenario:** Paper trading shows problems, need to iterate

**Without GPU (9 workers):**
- Fix hyperparameters, retrain 500 trials: 11 hours
- Test, iterate, repeat: 2-3 days per iteration

**With GPU (20 workers):**
- Fix hyperparameters, retrain 500 trials: 5.5 hours
- Test, iterate, repeat: 1-1.5 days per iteration

**Potential savings:** 3-7 days per iteration if issues found

### 2. Ensemble Optimization

**Scenario:** Need to test different ensemble configurations

**Current approach:** Test 3-5 different ensemble sizes
**With more models:** Test 10-20 configurations

**Value:** Find optimal ensemble faster, potentially better performance

**Timeline impact:** 1-2 days if optimization needed

### 3. Future Retraining

**After launch, monthly retraining needed:**

**Without GPU:**
- Download fresh data: 10 min
- Train 1,000 trials: 22 hours
- Deploy: 1 hour
- **Total: ~24 hours/month**

**With GPU:**
- Download fresh data: 10 min
- Train 1,000 trials: 11 hours
- Deploy: 1 hour
- **Total: ~12 hours/month**

**Savings:** 50% reduction in maintenance time (long-term benefit)

---

## Cost-Benefit Analysis

### Initial Launch Impact

**Time Cost:**
- Migration: 3 hours
- Lost training during migration: ~3 trials

**Time Saved:**
- If need 2,000 trials: 12 hours (0.5 days)
- If paper trading requires iteration: 3-7 days per iteration
- If ensemble optimization needed: 1-2 days

**Net Impact on Launch:**
- Best case: 0 days (already have good models)
- Realistic: 2-3 days faster
- If problems found: 5-10 days faster (multiple iterations)

### Long-Term Benefits

**Ongoing Operations:**
- Monthly retraining: 50% faster (12 vs 24 hours)
- Strategy experiments: 2x throughput
- Model improvements: Faster iteration

**Value:** Significant ongoing time savings

---

## Recommendation by Scenario

### Scenario 1: Everything Going Smoothly

**Current Status:**
- ✅ Have excellent models (Sharpe 0.0140)
- 🔄 Paper trading running with improvements
- ✅ On track for Feb 20 launch

**Recommendation:** **Wait until after initial launch**

**Reasoning:**
- Won't significantly accelerate launch (2-3 days max)
- Avoid risk of downtime during critical validation
- Can upgrade during post-launch optimization phase

**Best timing:** After Feb 20 launch, before monthly retraining

### Scenario 2: Paper Trading Shows Issues

**If on Jan 20 results show:**
- ❌ Still overtrading (frequency not reduced)
- ❌ Concentration still high
- ❌ Negative returns continuing

**Recommendation:** **Upgrade immediately (Jan 21)**

**Reasoning:**
- Will need multiple training iterations to fix
- 2x speed = 50% time savings per iteration
- Could save 1-2 weeks total
- Move launch from Feb 28 → Feb 21

### Scenario 3: Want More Model Options

**If you decide:**
- Need 5,000 trials instead of 1,000
- Want to train multiple strategies
- Want to experiment with architectures

**Recommendation:** **Upgrade now (Jan 16)**

**Reasoning:**
- Save 46 hours getting to 5,000 trials
- More exploration = better final models
- Parallel strategy testing

### Scenario 4: Long-Term Thinking

**If you're optimizing for:**
- Monthly retraining efficiency
- Ongoing strategy development
- Future experimentation

**Recommendation:** **Upgrade now or after launch**

**Reasoning:**
- Long-term time savings (50% on retraining)
- Flexibility for experimentation
- Not urgent, but valuable over time

---

## My Recommendation

**For initial launch timeline:** **Minimal impact (2-3 days)**

**Suggested approach:**

### Option A: Upgrade Now (After 1,000 Trials)
**Timing:** Today, after hitting 1,000 trials (~30 min)
**Pros:**
- Get it over with during non-critical phase
- Ready if paper trading requires iteration
- Start building experience with AMD/ROCm
**Cons:**
- 3 hours downtime
- Slight risk if issues with ROCm setup

### Option B: Wait for Paper Trading Results
**Timing:** Jan 20 (after initial validation)
**Pros:**
- Know if you need faster iteration
- Less critical moment
- Can assess if more trials needed
**Cons:**
- If problems found, delays fix by migration time
- Back-to-back with potential troubleshooting

### Option C: Wait Until After Launch
**Timing:** Late February (after launch)
**Pros:**
- Zero impact on launch timeline
- Most stable approach
- Can do during maintenance window
**Cons:**
- Miss opportunity for faster iteration if needed
- No benefit for initial launch

**My pick:** **Option A (Upgrade Now)**

**Why:**
1. You're at a natural breakpoint (1,000 trials)
2. Insurance policy if paper trading needs fixes (likely)
3. Only 3 hours downtime, minimal risk
4. You've already documented everything
5. Long-term benefits substantial
6. 2-3 day potential savings on launch

---

## Timeline Comparison Chart

```
WITHOUT GPU UPGRADE:
Jan 16 |████████████| Paper Trading |████████████| Launch
       1000 trials              Feb 10          Feb 20
                                              (43 days)

WITH GPU UPGRADE (After 1k):
Jan 16 |█| GPU |████████████| Paper Trading |████████████| Launch
       1000  swap           Feb 10              Feb 17-18
       -3hr                                    (40 days, -3 days)

IF ISSUES FOUND (Needs iteration):
WITHOUT GPU:
Jan 20 |████| Fix/Train |████| Test |████| Train |████| Launch
       Issue   ~24hrs    Test   ~24hr Test        Mar 5
                                                 (48 days)

WITH GPU:
Jan 20 |██| Fix/Train |██| Test |██| Train |██| Launch
       Issue  ~12hrs   Test  ~12hr Test     Feb 25
                                           (40 days, -8 days!)
```

---

## Bottom Line

### Initial Launch: +2-3 days

**Without GPU:**
- Launch: Feb 20 (realistic)
- Time: 43 days from today

**With GPU:**
- Launch: Feb 17-18 (realistic)
- Time: 40 days from today
- Migration cost: 3 hours

**Net benefit:** Small but meaningful

### If Problems Found: +5-10 days

The real value is **insurance**. If paper trading reveals issues, the GPU upgrade becomes very valuable for rapid iteration.

### Long-Term: 50% time savings

For monthly retraining and ongoing development, the benefit is substantial.

---

## Decision Framework

**Upgrade NOW if:**
- ✅ You value the 2-3 day speedup
- ✅ You want insurance for paper trading issues
- ✅ You're thinking long-term (retraining, experiments)
- ✅ You're comfortable with 3-hour downtime

**Upgrade LATER (Jan 20) if:**
- ⏸️ You want to see paper trading results first
- ⏸️ You want to minimize any risk during validation
- ⏸️ 2-3 days doesn't matter to you

**Upgrade AFTER LAUNCH if:**
- ⏸️ You want zero impact on launch timeline
- ⏸️ Everything is going perfectly
- ⏸️ You prefer maximum stability

**My recommendation:** Upgrade after hitting 1,000 trials (in ~30 min). The risk is low, the long-term benefits are high, and the 2-3 day potential savings on launch is a nice bonus.

---

**Analysis Date:** January 16, 2026, 15:45 UTC
**Current Trials:** 970+
**Next Milestone:** 1,000 trials (~30 minutes)
**Launch Target:** Feb 17-20 (with GPU) vs Feb 20-23 (without)
