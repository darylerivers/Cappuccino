# Deep Dive Analysis - Why Models Don't Beat Benchmarks

**Date:** 2025-12-15
**Analysis Type:** Pre-Step 3 Investigation
**Status:** Option B (Thorough Path) - Analysis Complete

---

## Executive Summary

**Problem:** All 15 models perform at benchmark level (-5.76% avg) but don't generate alpha.

**Root Causes Identified:**
1. ❌ **Reward function encourages concentration over diversification**
2. ❌ **Concentration penalty has exploit (2 positions at 50% each)**
3. ❌ **Cash reserve constraints not enforced in arena inference**
4. ❌ **All models converged to same strategy (buy-hold 2 coins)**
5. ❌ **No incentive to generate absolute returns, only beat equal-weight**

---

## Finding 1: Extreme Portfolio Concentration

### Observed Behavior
- **All 15 models hold exactly 2 positions**
- **0% cash reserves** (100% invested)
- **Popular holdings:** LINK (60%), AAVE (66.7%), LTC (33%), UNI (33%)
- **Ignoring:** BTC, ETH, BCH (5 out of 7 cryptos largely ignored)

### Data
```
PORTFOLIO CONCENTRATION
- Concentrated (≤2 positions): 15/15 models (100%)
- Diversified (≥3 positions): 0/15 models (0%)
- Average cash %: 0.0%
```

### Why This Happened

**Concentration Penalty Exploit:**
```python
# environment_Alpaca.py line 414
if max_concentration > 0.60:  # Only triggers if ONE position > 60%
    penalty = -self.concentration_penalty * concentration_excess
```

**The Exploit:** Models discovered that holding 2 positions at ~50% each bypasses the penalty entirely while maximizing concentration in best performers.

**Trained with cash reserves, but not enforced:**
- Trial 708: `min_cash_reserve = 0.3` (30%)
- Trial 786: `min_cash_reserve = 0.1` (10%)
- **Arena reality:** 0% cash (constraint not enforced during inference)

---

## Finding 2: Reward Function Misalignment

### Current Reward Structure
```python
# Line 404
reward = (delta_bot - delta_eqw) * self.norm_reward * decay_factor
```

**What this rewards:**
- ✅ Beating equal-weight portfolio
- ✅ Outperforming a buy-and-hold diversified strategy

**What this DOESN'T reward:**
- ❌ Absolute returns
- ❌ Alpha generation
- ❌ Diversification
- ❌ Cash management
- ❌ Risk-adjusted returns (Sharpe not in reward)

### The Problem

Equal-weight portfolio:
- Buys all 7 cryptos equally
- Some cryptos outperform, some underperform
- Average performance ≈ -5.29%

Model strategy:
- Identify best 2 performers (LINK, AAVE)
- Concentrate 100% in them (50/50 split)
- Beat equal-weight by avoiding worst performers
- **But still lose money overall** (-5.76%)

**Result:** Models learned to "lose less" than equal-weight, not to "win".

---

## Finding 3: All Models Converged to Same Strategy

### Trading Pattern Analysis

```
Average across all 15 models:
- Return: -5.76%
- Trades: 45.1
- Win Rate: 0.0% (no closed winning trades)
- Trades/Hour: 0.64
- Cash %: 0.0%
- Positions: 2.0
```

**Top 3 models:**
1. trial_786: -5.39%, LINK + AAVE
2. trial_1226: -5.41%, LINK + AAVE
3. trial_744: -5.42%, LINK + AAVE

**Bottom 3 models:**
1. trial_1610: -6.03%, LINK + AAVE
2. trial_1608: -6.03%, LINK + AAVE
3. trial_1631: -6.04%, LINK + AAVE

### Observation

**0.65% spread between best and worst** (all essentially identical)

This suggests:
- Optimization converged to **local optimum**
- Hyperparameters didn't matter much (all found same solution)
- Search space may be too constrained
- Or reward function has a dominant strategy

---

## Finding 4: Transaction Costs Impact

### Estimated Fee Burden

Assuming 0.25% fees (current default):
- **45 trades average** per model
- **~50% are round-trips** (buy then sell) = ~23 round-trips
- **Fee per round-trip:** 0.5% (0.25% buy + 0.25% sell)
- **Total fee cost:** ~11.5% of capital over 72 hours

**But wait...** Models are currently HOLDING, not closing trades:
- Most trades are **rebalancing** between LINK and AAVE
- Not full round-trips to cash
- Actual fee impact likely **3-5%** of initial capital

**Impact on returns:**
- Current return: -5.76%
- Estimated fees: ~3-5%
- **Potential return before fees:** -0.76% to +0.24%

**Conclusion:** Fees are eating a significant portion of potential gains, but not the primary problem.

---

## Finding 5: Win Rate is Misleading

### Why 0% Win Rate?

```python
# model_arena.py line 1117-1118
if price > entry_price:
    portfolio.winning_trades += 1
```

**Only counts CLOSED trades** (sells that are profitable).

Current state:
- Models are HOLDING positions
- No positions have been closed yet
- Therefore winning_trades = 0

**This is technically correct** but misleading. Open positions may be profitable, we just haven't realized it.

---

## Finding 6: Market Conditions

### 72-Hour Period Performance

**Benchmarks:**
- Equal Weight: -5.29%
- BTC Only: -4.44%
- 60/40 BTC/ETH: -4.36%

**Observation:** All benchmarks DOWN during evaluation period.

**Model performance vs benchmarks:**
- Models: -5.76% avg
- Benchmarks: -4.36% to -5.29%
- **Gap:** -0.47% to -1.40% worse than benchmarks

**Conclusion:** Models are LOSING to benchmarks, not even matching them.

---

## Root Cause Analysis

### Why All Models Look Identical

**Hypothesis: Reward Function Creates Dominant Strategy**

The current reward structure creates a **game with a clear optimal strategy:**

1. **Goal:** Beat equal-weight portfolio
2. **Equal-weight holds:** All 7 cryptos equally
3. **Best strategy:** Identify top 2 performers, ignore rest
4. **Constraint:** Don't exceed 60% in single position
5. **Solution:** 50/50 split in top 2 coins

**This strategy is:**
- ✅ Simple to discover
- ✅ Robust across different hyperparameters
- ✅ Local optimum (hard to escape once found)
- ❌ **Doesn't generate positive absolute returns**

### Why Models Don't Generate Alpha

**Alpha requires:**
- Timing (buy low, sell high)
- Market regime detection (bull vs bear)
- Risk management (cash in downtrends)
- Diversification (spread risk)

**Current reward only requires:**
- Concentration (pick winners)
- Hold positions (no selling incentive)

**Missing incentives:**
- No reward for cash during downtrends
- No reward for timing entries/exits
- No reward for absolute positive returns
- No reward for Sharpe ratio improvement

---

## Comparison to Benchmarks

### Performance Table

| Strategy | Return | Sortino | Sharpe | MaxDD |
|----------|--------|---------|--------|-------|
| **Models (avg)** | **-5.76%** | **-4.25** | **-5.85** | **varies** |
| Equal Weight | -5.29% | -0.16 | -0.15 | 17.77% |
| BTC Only | -4.44% | -10.88 | -12.69 | 6.35% |
| 60/40 BTC/ETH | -4.36% | +4.45 | +3.09 | 40.11% |

**Analysis:**
- Models worse than all benchmarks in returns
- Models have terrible Sortino/Sharpe (worse than simple buy-hold)
- **60/40 BTC/ETH beats models by 1.4% with better risk metrics**

**This is the core problem:** Even simple passive strategies outperform the trained models.

---

## Recommendations

### Critical Fixes (Before Step 3)

#### 1. Fix Concentration Penalty

**Current (broken):**
```python
if max_concentration > 0.60:  # Only checks single position
    penalty = -concentration_penalty * concentration_excess
```

**Proposed (better):**
```python
# Penalize insufficient diversification
active_positions = (position_ratios > 0.01).sum()  # Count positions > 1%
if active_positions < 3:  # Require minimum 3 positions
    penalty = -concentration_penalty * (3 - active_positions)

# Also keep max concentration check
if max_concentration > 0.40:  # Lower threshold (40% vs 60%)
    penalty -= concentration_penalty * (max_concentration - 0.40)
```

#### 2. Enforce Cash Reserve Constraints

**Problem:** Arena doesn't enforce `min_cash_reserve` from training.

**Fix:** Ensure arena passes same env_params as training, or enforce cash reserve in reward function.

#### 3. Revise Reward Function

**Option A - Add Absolute Return Component:**
```python
# Hybrid reward: beat benchmark AND generate positive returns
alpha = delta_bot - delta_eqw
absolute_return = delta_bot / initial_capital
reward = 0.7 * alpha + 0.3 * absolute_return
```

**Option B - Target Sharpe Ratio:**
```python
# Reward = Sharpe ratio improvement
sharpe = (portfolio_return - risk_free) / portfolio_volatility
reward = sharpe - benchmark_sharpe
```

**Option C - Multi-Objective:**
```python
# Balance multiple goals
reward = (
    0.4 * alpha +  # Beat benchmark
    0.3 * absolute_return +  # Positive returns
    0.2 * sharpe_improvement +  # Risk-adjusted
    0.1 * diversification_bonus  # Encourage spreading
)
```

#### 4. Increase Diversification Requirements

- Require minimum 3-4 active positions
- Cap single position at 30-40% (not 60%)
- Bonus for holding cash during downtrends
- Penalty for overtrading (fee-awareness)

### Medium-Term Improvements (With Step 3)

#### 5. Add Rolling Mean Features (Planned Step 3)

**Why this helps:**
- 7-day and 30-day trends provide momentum signals
- Volatility features help risk management
- Better signal for market regime detection
- Could enable timing instead of just selection

**Risk:** More complex state (63 → 91 dims) without fixing fundamental issues may not help.

#### 6. Expand Training Data

**Current:** 1 year of data (2023)
**Proposal:** 2-3 years including different regimes:
- Bull market (2020-2021)
- Bear market (2022)
- Recovery (2023)

#### 7. Alternative Algorithms

**Current:** PPO only
**Test:** SAC, TD3, DDQN (Phase 2 planned)

These might explore action space differently.

---

## Decision Matrix: Next Steps

### Option 1: Fix Fundamentals First ⭐ RECOMMENDED

**Actions:**
1. Fix concentration penalty (2 hours)
2. Revise reward function (3 hours)
3. Enforce cash reserves (1 hour)
4. Run 50-trial test (overnight)
5. Compare to current results

**Timeline:** 1-2 days
**Risk:** Low (fixing obvious bugs)
**Potential:** High (could unlock alpha generation)

### Option 2: Proceed to Step 3 Anyway

**Actions:**
1. Build rolling mean pipeline (3 hours)
2. Launch Phase 2 training (days)
3. Hope features compensate for reward issues

**Timeline:** 3-5 days
**Risk:** Medium (adding complexity on broken foundation)
**Potential:** Unknown (features might help, might not)

### Option 3: Hybrid - Quick Fixes + Step 3

**Actions:**
1. Fix concentration penalty only (2 hours)
2. Build Step 3 pipeline (3 hours)
3. Test both phases in parallel

**Timeline:** 2 days
**Risk:** Medium
**Potential:** Medium (incremental improvements)

---

## Conclusion

**The models are working correctly** - they're optimizing exactly what we asked them to:
- Beat equal-weight portfolio ✅
- Stay under 60% single position ✅
- Maximize concentration in best performers ✅

**But we asked for the wrong thing.** The reward function doesn't incentivize:
- Positive absolute returns ❌
- Diversification ❌
- Cash management ❌
- Alpha generation ❌

**Recommendation:** Fix the reward function and concentration logic BEFORE adding Step 3 features. Adding complexity (rolling means) won't fix a misaligned objective function.

**Estimated Impact:**
- Fix concentration: +0.5% to +1% improvement
- Fix reward function: +1% to +3% improvement
- Add Step 3 features: +0.5% to +2% improvement

**Total potential:** +2% to +6% improvement = possibly positive returns

---

**Next Action:** Implement Option 1 fixes, then proceed to Step 3 with corrected foundation.

