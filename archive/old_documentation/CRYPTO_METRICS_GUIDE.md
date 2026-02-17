# Crypto Trading Metrics - Interpretation Guide

## Overview

Your arena now tracks **3 risk-adjusted return metrics** optimized for cryptocurrency trading. This guide explains what each metric means, how to interpret values, and which models to promote.

---

## The Three Metrics

### 1. **Sortino Ratio** (Primary Metric - Weighted 3x)

**What it measures:**
```
Sortino = Average Return / Downside Deviation
```
- Only penalizes DOWNSIDE volatility (losses)
- Ignores upside volatility (gains)
- Better for asymmetric returns like crypto

**Interpretation:**

| Sortino Value | Quality | Meaning |
|--------------|---------|----------|
| < 0 | Poor | Negative returns or high downside risk |
| 0 - 0.5 | Weak | Returns don't justify downside risk |
| 0.5 - 1.0 | Average | Acceptable risk-adjusted returns |
| 1.0 - 2.0 | Good | Strong downside protection |
| 2.0 - 3.0 | Excellent | Elite risk management |
| > 3.0 | Outstanding | Institutional-grade performance |

**Why it's the primary metric:**
- Crypto has extreme upside potential (good volatility)
- We want to capture gains without penalty
- Only care about protecting against losses

**Example:**
```
Model A: +15% return, lots of upside spikes, few small dips
  Sharpe: 1.2 (penalized for upside volatility)
  Sortino: 2.8 (rewards controlled downside)
  → Model A is better for crypto!

Model B: +15% return, steady gains, steady losses
  Sharpe: 1.8 (rewards steadiness)
  Sortino: 1.5 (moderate downside control)
  → Model B better for traditional assets
```

**What to look for:**
- Models with Sortino > 2.0 after 48+ hours → Promotion candidates
- Sortino much higher than Sharpe → Model captures asymmetric upside
- Rising Sortino over time → Learning to limit downside

---

### 2. **Calmar Ratio**

**What it measures:**
```
Calmar = Annualized Return / Maximum Drawdown
```
- Return efficiency per unit of worst-case loss
- Popular in hedge funds and crypto funds
- Simple, intuitive risk measure

**Interpretation:**

| Calmar Value | Quality | Meaning |
|--------------|---------|----------|
| < 0 | Poor | Negative returns |
| 0 - 0.5 | Weak | Returns don't justify max loss |
| 0.5 - 1.0 | Average | Moderate efficiency |
| 1.0 - 2.0 | Good | Strong return per risk |
| 2.0 - 3.0 | Excellent | Elite efficiency |
| > 3.0 | Outstanding | Best-in-class |

**Why it matters:**
- Investors care about worst-case scenarios
- "If my account dropped 10%, what return did I get?"
- Simple to explain to non-technical stakeholders

**Example:**
```
Model A: 30% annual return, 10% max drawdown
  Calmar = 30 / 10 = 3.0 (Excellent!)
  → For every 1% of max loss, gained 3%

Model B: 30% annual return, 25% max drawdown
  Calmar = 30 / 25 = 1.2 (Average)
  → For every 1% of max loss, gained 1.2%

Both have same returns, but Model A manages risk 2.5x better!
```

**What to look for:**
- Calmar > 2.0 after 48+ hours → Strong candidate
- Calmar > Sortino → Very tight drawdown control
- Stable Calmar over time → Consistent risk management

---

### 3. **Sharpe Ratio** (Secondary - Weighted 1x)

**What it measures:**
```
Sharpe = Average Return / Total Volatility (up + down)
```
- Traditional finance standard metric
- Penalizes ALL volatility equally
- Good for comparison to traditional portfolios

**Interpretation:**

| Sharpe Value | Quality | Traditional Benchmark |
|--------------|---------|----------------------|
| < 0 | Poor | Losing money |
| 0 - 0.5 | Weak | Below money market rates |
| 0.5 - 1.0 | Average | Similar to bonds |
| 1.0 - 2.0 | Good | Better than S&P 500 |
| 2.0 - 3.0 | Excellent | Top quartile hedge funds |
| > 3.0 | Outstanding | Top 1% performers |

**Why we still track it:**
- Industry standard for reporting
- Compare to traditional benchmarks
- Regulatory/compliance requirements
- Useful for conservative risk assessment

**Crypto caveat:**
- Sharpe will typically be LOWER than Sortino for good crypto models
- That's expected and desired!
- Crypto's upside volatility is a feature, not a bug

**What to look for:**
- Sharpe > 1.0 after 48+ hours → Beating traditional assets
- Sortino >> Sharpe → Confirming asymmetric returns
- Sharpe < 0 → Red flag, negative risk-adjusted returns

---

## Composite Score Formula

**How models are ranked:**
```python
score = return_pct + (sortino * 3) + (sharpe * 1) - (max_dd * 0.5)
```

**Weights:**
- **Return**: 1x (absolute performance matters)
- **Sortino**: 3x (PRIMARY - downside risk management)
- **Sharpe**: 1x (SECONDARY - traditional comparison)
- **Max Drawdown**: -0.5x (penalty for large losses)

**Why this weighting?**
- Sortino is most relevant for crypto's asymmetric returns
- Sharpe provides traditional finance validation
- Direct return still important (no point in great ratios with tiny returns)
- Max drawdown penalty ensures we don't blow up

---

## Interpreting Your Arena Results

### After 1-10 Hours (Current State)
**All metrics show 0.00** - This is normal!
- Need 10+ data points for statistical validity
- Returns too small to calculate meaningful ratios
- Focus on: Are models trading? Any errors?

**What to watch:**
- Models should show some trades (2-10 per hour)
- Returns should be small but varied (-0.5% to +0.5%)
- No crashes or errors in logs

### After 10-24 Hours
**Metrics become meaningful**
- Sortino/Sharpe ratios start showing real values
- Can identify early leaders/laggards
- Performance divergence becomes visible

**What to look for:**
- Any Sortino > 1.0? → Strong early signal
- Models clustered together? → Need more time
- Wide variance? → Clear winners emerging

**Action:**
- Review top 3 models in detail
- Check if any models consistently losing
- Monitor for unusual behavior

### After 24-48 Hours
**Statistical significance achieved**
- Metrics are reliable for evaluation
- Clear performance tiers should emerge
- Promotion candidates identified

**What to look for:**
- **Sortino > 2.0** → Excellent candidate
- **Calmar > 1.5** → Strong risk management
- **Sharpe > 1.0** → Beating traditional benchmarks
- **Return > 2%** → Meets promotion threshold

**Promotion criteria (all must be true):**
1. Hours in arena ≥ 48
2. Return % ≥ 2.0%
3. Sortino ratio > 0
4. No major red flags (crashes, stuck positions, etc.)

### After 48+ Hours
**Full evaluation complete**
- Top model eligible for promotion to paper trading
- Can confidently assess model quality
- Ready for next stage

---

## Benchmark Comparison

**Your arena tracks 3 buy-and-hold benchmarks:**

1. **Equal Weight Portfolio** - $1000 split across all 7 cryptos
2. **BTC Only** - 100% Bitcoin allocation
3. **60/40 BTC/ETH** - 60% BTC, 40% ETH

**What to look for:**

| Comparison | Meaning | Action |
|-----------|---------|--------|
| Model beats all 3 benchmarks | Active trading adds value | Promote to paper trading |
| Model beats 2/3 benchmarks | Conditional success | Review which market conditions favor it |
| Model beats 1/3 benchmarks | Marginal performance | Need longer evaluation or reject |
| Model beats 0/3 benchmarks | Underperforming passive | Reject, cycle in new model |

**Metric-specific comparisons:**
- **Sortino vs benchmarks** → Is downside protection better than buy-and-hold?
- **Calmar vs benchmarks** → Getting more return per drawdown?
- **Sharpe vs benchmarks** → Better risk-adjusted returns overall?

**Market regime matters:**
- **Bull market**: Models should beat BTC-only
- **Bear market**: Models should lose less than benchmarks (negative returns OK if better than -1.23%)
- **Sideways market**: Models should beat equal-weight through active trading

---

## Red Flags to Watch For

### Metric Red Flags

| Pattern | Issue | Meaning |
|---------|-------|---------|
| Sortino < 0 after 24h | Negative downside-adjusted returns | Model losing money with high downside |
| Sharpe < 0 after 24h | Negative risk-adjusted returns | Poor performance overall |
| Calmar < 0 | Negative returns | Losing money |
| Sortino << Sharpe | Excessive downside | More losses than gains |
| Calmar < 0.5 | Poor efficiency | Too much drawdown for returns |

### Behavioral Red Flags

| Pattern | Issue | Action |
|---------|-------|--------|
| 0 trades after 10h | Model not trading | Check actor loading, inference |
| >50 trades in 1h | Overtrading | Review action threshold |
| Return but 0% win rate | All profitable exits from stops | May be luck, not skill |
| Max DD > 25% | Excessive drawdown | Too risky, reject |
| Metrics oscillating wildly | Unstable strategy | Need more evaluation time |

---

## Decision Matrix: Should I Promote This Model?

### Elite Model (Promote Immediately at 48h)
- ✅ Return ≥ 5%
- ✅ Sortino ≥ 2.5
- ✅ Calmar ≥ 2.0
- ✅ Beats all 3 benchmarks
- ✅ Max DD < 10%

### Strong Model (Promote at 48h)
- ✅ Return ≥ 2%
- ✅ Sortino ≥ 1.5
- ✅ Calmar ≥ 1.0
- ✅ Beats 2/3 benchmarks
- ✅ Max DD < 15%

### Marginal Model (Extended Evaluation)
- ⚠️ Return 0-2%
- ⚠️ Sortino 0.5-1.5
- ⚠️ Beats 1/3 benchmarks
- ⚠️ Max DD 15-20%
- **Action**: Run for 72-96 hours, then decide

### Weak Model (Reject)
- ❌ Return < 0%
- ❌ Sortino < 0.5
- ❌ Beats 0/3 benchmarks
- ❌ Max DD > 20%
- **Action**: Remove from arena, cycle in new model

---

## Practical Examples

### Example 1: Clear Winner

```
Model: trial_1226 (after 48h)
Return:    +6.5%
Sortino:   3.2
Sharpe:    1.8
Calmar:    2.9
Max DD:    2.2%
Win Rate:  68%

Benchmarks:
- Equal Weight: +2.1%
- BTC Only: +3.5%
- 60/40: +2.8%

Analysis:
✅ Crushes all benchmarks
✅ Elite Sortino (3.2)
✅ Elite Calmar (2.9)
✅ Tight drawdown control (2.2%)
✅ High win rate (68%)

Decision: PROMOTE TO PAPER TRADING
```

### Example 2: Market Dependent

```
Model: trial_705 (after 48h)
Return:    -1.2%
Sortino:   2.1
Sharpe:    0.8
Calmar:    -1.5
Max DD:    0.8%

Benchmarks:
- Equal Weight: -2.5%
- BTC Only: -3.1%
- 60/40: -2.8%

Analysis:
✅ Beats all benchmarks (loses less in bear market!)
✅ Excellent Sortino (2.1) - great downside protection
✅ Very low drawdown (0.8%)
❌ Negative absolute return
⚠️ Calmar negative (but expected in bear market)

Decision: CONDITIONAL PROMOTE
- Good for bear/sideways markets
- May underperform in bull markets
- Monitor in paper trading, diversify with bull-market models
```

### Example 3: Reject

```
Model: trial_784 (after 48h)
Return:    +1.5%
Sortino:   0.4
Sharpe:    0.3
Calmar:    0.6
Max DD:    2.5%
Win Rate:  25%

Benchmarks:
- Equal Weight: +2.1%
- BTC Only: +3.5%
- 60/40: +2.8%

Analysis:
❌ Underperforms all benchmarks
❌ Poor Sortino (0.4)
❌ Poor Sharpe (0.3)
❌ Low win rate (25%)
❌ Returns don't justify risk

Decision: REJECT
- Remove from arena
- Cycle in next best model from training
```

---

## Quick Reference Card

**Minimum Thresholds for Promotion (48h evaluation):**
```
Return:          ≥ 2.0%
Sortino:         ≥ 1.0 (target 2.0+)
Sharpe:          ≥ 0.5 (target 1.0+)
Calmar:          ≥ 0.5 (target 1.5+)
Max Drawdown:    < 15% (target <10%)
Win Rate:        ≥ 40% (target 55%+)
Benchmarks:      Beat at least 2/3
```

**Ideal Model Profile:**
```
Return:          3-10% (per 48h period)
Sortino:         2.0-4.0
Sharpe:          1.0-2.0
Calmar:          1.5-3.0
Max Drawdown:    2-8%
Win Rate:        55-70%
Benchmarks:      Beats all 3
```

**When to Check:**
- **Hour 10**: Are metrics appearing? Any early signals?
- **Hour 24**: Which models leading? Any clear laggards?
- **Hour 36**: Rankings stabilizing? Clear top 3?
- **Hour 48**: Promotion decision time

---

## Advanced: Interpreting Metric Relationships

### Sortino vs Sharpe

| Pattern | Meaning | Example |
|---------|---------|---------|
| Sortino ≈ Sharpe | Symmetric returns | Steady grinder model |
| Sortino > Sharpe (1.5x) | Good asymmetry | Captures upside, limits downside |
| Sortino >> Sharpe (2x+) | Excellent asymmetry | Crypto-optimized model |
| Sortino < Sharpe | Too much downside | Red flag - review immediately |

### Calmar vs Sortino

| Pattern | Meaning | Best For |
|---------|---------|----------|
| Calmar > Sortino | Tight DD control | Risk-averse investors |
| Calmar ≈ Sortino | Balanced | General use |
| Calmar < Sortino | Higher peaks/valleys | Aggressive growth |

### Return vs Ratios

| Pattern | Meaning | Action |
|---------|---------|--------|
| High return, low ratios | Risky gains | Reject or reduce allocation |
| Low return, high ratios | Safe but slow | Good for large capital |
| High return, high ratios | Sweet spot | Promote immediately |
| Low return, low ratios | Worst case | Reject |

---

## Summary

**Primary Focus:** Sortino Ratio
- Best metric for crypto's asymmetric returns
- Target: 2.0+ for promotion
- Weighted 3x in scoring algorithm

**Secondary Focus:** Calmar Ratio
- Shows return efficiency vs worst-case loss
- Target: 1.5+ for promotion
- Easy to explain to stakeholders

**Context Metric:** Sharpe Ratio
- Traditional finance comparison
- Target: 1.0+ for promotion
- Validates performance against standard benchmarks

**The winning combination:**
- Sortino > 2.0 (elite downside protection)
- Calmar > 1.5 (efficient risk-taking)
- Sharpe > 1.0 (beats traditional assets)
- Beats 2/3 benchmarks (adds value vs passive)
- Return > 2% (absolute performance threshold)

Check your arena at **48-hour mark** (Dec 14, 00:08 UTC) for first promotion candidate!
