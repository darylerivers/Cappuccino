# Arena Analysis - December 15, 2025

## Summary

All 15 remaining models are performing at **benchmark level** (not significantly better or worse).

---

## Current Performance

### Model Performance Range
- **Best models:** -4.40% to -5.47% (top 10)
- **Worst models:** -5.81% to -6.04% (bottom 5)
- **Average:** ~-5.5%

### Benchmark Performance
- **Equal Weight Portfolio:** -5.29%
- **BTC Only:** -4.44%
- **60/40 BTC/ETH:** -4.36%

### Key Observation
**Models are tracking benchmarks** but not beating them. This suggests:
1. Models have learned basic market behavior
2. But haven't learned alpha-generating strategies
3. Essentially performing like "smart buy-and-hold"

---

## Model Rankings (by Sortino Ratio)

### Top Tier (Positive Sortino)
1. trial_708: -5.85%, Sortino 13.78
2. trial_1575: -5.91%, Sortino 13.17
3. trial_783: -5.86%, Sortino 13.05
4. trial_784: -5.87%, Sortino 12.83
5. trial_705: -5.87%, Sortino 12.37

**Analysis:** Positive Sortino means they limit downside volatility well, even though overall returns are negative.

### Middle Tier (Negative Sortino, -8 to -10)
6. trial_786: -5.39%, Sortino -8.08
7. trial_1226: -5.41%, Sortino -8.11
8. trial_744: -5.42%, Sortino -8.12
9. trial_1165: -5.45%, Sortino -8.16
10. trial_785: -5.47%, Sortino -8.18

**Analysis:** Slightly better returns but worse risk-adjusted performance (higher downside volatility).

### Bottom Tier (Negative Sortino, -9 to -13)
11-15: Sortino -9.32 to -12.92

**Analysis:** Worst risk-adjusted returns, candidates for next pruning cycle.

---

## Why Models Aren't Beating Benchmarks

### Hypothesis 1: Insufficient Training Data Variety
- Training on limited market regimes
- Not enough bull/bear/sideways scenarios
- Overfitting to training period conditions

### Hypothesis 2: Reward Function Issues
- Current reward may not incentivize alpha generation
- Focus on Sharpe ratio improvement might lead to conservative strategies
- Transaction costs eating into potential gains

### Hypothesis 3: Feature Limitations
- Current features (MACD, RSI, CCI, DX) may be insufficient
- **This is why Step 3 (rolling means) could help!**
- Need more sophisticated indicators for alpha

### Hypothesis 4: Model Capacity
- Networks might be too small (current: varies by trial)
- Phase 2 proposes larger nets (1024-2560 dims)
- Could need more complex architectures

### Hypothesis 5: Market Regime
- Past 72 hours has been a downtrend (-4 to -6%)
- Models might perform better in different conditions
- Need longer evaluation period

---

## Recommendations

### Short Term (Before Step 3)

1. **Let arena run longer** (7+ days)
   - Current evaluation: 50-73 hours
   - Need to see performance across more market conditions
   - Weekly cycles might reveal patterns

2. **Analyze trade patterns**
   - What trades are models making?
   - Are they over-trading (fees eating profits)?
   - Are they holding too long or selling too quickly?

3. **Review training hyperparameters**
   - Check top trials from Optuna database
   - Look for patterns in best performers
   - Consider if search space is optimal

### Medium Term (With Step 3)

4. **Add rolling mean features**
   - 7-day and 30-day trend indicators
   - Volatility features
   - Could provide better signal for alpha

5. **Expand state dimension carefully**
   - Phase 2: 63 → 91 dimensions (+44%)
   - Need larger networks to handle complexity
   - Risk: more parameters = harder to train

### Long Term

6. **Multi-timeframe analysis**
   - Current: 1h candles only
   - Consider 4h/1d context for better trends
   - Could reduce noise from short-term volatility

7. **Alternative algorithms**
   - Current: PPO only
   - Phase 2 proposes DDQN as well
   - Could try SAC, TD3, or other DRL methods

---

## Data Quality Check

### Time in Arena
- Minimum: 50h (trial_1944)
- Maximum: 73h (trial_708, others)
- All above minimum threshold (48h)

### Trade Activity
- Low traders: 10 trades (trial_1273, trial_1622, trial_1600)
- High traders: 56 trades (trial_1587)
- Most: 35-52 trades

**Observation:** Some models are very conservative (10 trades in 72h = ~1 per 7h). Others more active (56 trades = ~1 per 1.3h).

### Drawdown Analysis
- Best: 6.20-6.35% max drawdown (conservative models)
- Worst: 67.99% max drawdown (aggressive models)
- Benchmarks: 6.35% (BTC) to 40.11% (60/40)

**Interesting:** Top Sortino models have HUGE drawdowns (67%) but still positive Sortino. This seems contradictory and worth investigating.

---

## Next Steps Decision Tree

```
Current Status: Models = Benchmarks (not beating)
│
├─ Option A: Understand Why (Analysis)
│   ├─ Dive into trade logs
│   ├─ Review hyperparameters
│   └─ Check for systematic issues
│   └─ THEN → Step 3
│
├─ Option B: Add Features (Step 3)
│   ├─ Build rolling mean pipeline
│   ├─ Hope new features unlock alpha
│   └─ Risk: More complexity without understanding current issues
│
└─ Option C: Longer Evaluation
    ├─ Let arena run 7-14 days
    ├─ See if patterns emerge
    └─ Then decide on Step 3 vs deeper analysis
```

---

## Pruning Results

**Removed (5 models):**
- trial_1273: -97.62% (actually -4.40%, display bug)
- trial_1600: -98.01% (actually -4.43%)
- trial_1622: -97.90% (actually -4.43%)
- trial_1938: -100.00% (actually -6.03%)
- trial_1944: -100.00% (actually -5.40%)

**Why removed:** Bottom performers by Sortino ratio despite some having competitive returns.

**Remaining:** 15 models with room for 5 more in arena (max: 20).

---

## Conclusion

**Good news:** Models work, aren't catastrophically bad
**Challenge:** Not generating alpha yet
**Path forward:** Either understand current limitations OR add features and hope for improvement

**Recommendation:** Option B (thorough) - Quick analysis of top trials, then proceed to Step 3 with informed parameter choices.

---

**Analysis Date:** 2025-12-15 17:45 UTC
**Analyst:** Claude Sonnet 4.5
**Next Review:** After 7 days arena runtime or Step 3 completion
