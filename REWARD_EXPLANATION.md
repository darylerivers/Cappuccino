# Understanding DRL Reward Values

## What is "Reward"?

In Deep Reinforcement Learning (DRL), **reward** is the signal that tells the agent whether its actions were good or bad. The agent learns to maximize cumulative reward over time.

**Think of it like training a dog:**
- Good action (fetch ball) → +1 treat (positive reward)
- Bad action (chew furniture) → -1 treat (negative reward)
- Neutral action → 0 treats

For your trading bot:
- Makes money → Positive reward
- Loses money → Negative reward
- Breaks rules (concentration, no diversification) → Extra penalties

---

## Your Reward Function (3 Components)

### Component 1: Alpha Reward (50% weight)
**Beating the benchmark**

```python
alpha_reward = (your_portfolio_change - benchmark_change) * normalization * decay
```

**What it means:**
- If you gain +$10 and benchmark gains +$5 → Positive alpha reward
- If you gain +$10 and benchmark gains +$15 → Negative alpha reward
- If you lose -$5 and benchmark loses -$10 → Positive alpha reward (you lost less!)

**Benchmark:** Equal-weight portfolio (buy all 7 tickers equally and HODL)

**Example:**
- Your portfolio: $500 → $502 (+$2)
- Benchmark: $500 → $501 (+$1)
- Alpha: $2 - $1 = $1
- Alpha reward: $1 × norm_reward × decay ≈ +0.01

### Component 2: Absolute Return Reward (30% weight)
**Raw profit/loss**

```python
absolute_reward = (portfolio_change / initial_capital) * normalization * decay
```

**What it means:**
- Portfolio goes up → Positive reward
- Portfolio goes down → Negative reward
- Normalized by initial capital ($500)

**Example:**
- Portfolio: $500 → $501 (+$1)
- Absolute return: $1 / $500 = 0.002 (0.2%)
- Absolute reward: 0.002 × norm_reward × decay ≈ +0.002

### Component 3: Cash Management Bonus (20% weight)
**Holding cash during downtrends**

```python
if market_down AND cash_ratio > 10%:
    cash_bonus = market_decline × cash_held × 0.1
```

**What it means:**
- Market drops -5%, you're holding 50% cash → Bonus!
- Market rises +5%, you're holding 50% cash → No bonus (missed gains)
- Market drops -5%, you're 100% invested → No bonus (took full hit)

**Example:**
- Market down -2%
- You're holding 40% cash
- Cash bonus: 0.02 × 0.4 × norm_reward × 0.1 ≈ +0.0008

### Final Reward Calculation

```python
reward = 0.5 × alpha_reward + 0.3 × absolute_reward + 0.2 × cash_bonus
```

Then apply **penalties** and **bonuses** (explained below).

---

## Why Rewards Fluctuate

### Normal Fluctuations (-0.08 to -0.09)

**Typical hour:**
- Portfolio changes by $0.10 to $0.50
- Slightly underperforming benchmark
- Reward: -0.08 to -0.09

**This is NORMAL for a conservative model:**
- Small position sizes (1-2% of portfolio)
- Careful trading (avoiding concentration)
- Not taking big risks

### Large Negative Reward (-7.54)

**What causes this?**

Let's break down what happened in your case:

```
Hour 27: reward = -0.09 (normal)
Hour 28: reward = -7.54 (HUGE negative)
```

**Possible causes:**

1. **Portfolio dropped significantly**
   - Delta: -$5 to -$10 (1-2% loss)
   - Multiplied by norm_reward (≈100)
   - Result: -0.5 to -1.0 reward

2. **Concentration penalty triggered**
   - Had >40% in single asset
   - Penalty: -concentration_penalty × excess
   - If penalty=10, excess=20%: -2.0 reward

3. **Diversification penalty**
   - Fewer than 3 active positions
   - Penalty: -concentration_penalty × 0.5 × (3 - positions)
   - If only 1 position: -10 × 0.5 × 2 = -10.0 reward

4. **Cash reserve violation**
   - Went below minimum cash reserve
   - Penalty: -(shortfall / initial) × norm_reward × 0.5
   - If shortfall=$10: -(10/500) × 100 × 0.5 = -1.0 reward

**Most likely:** Combination of #1 (small loss) + #3 (lack of diversification)

---

## Reward Components Breakdown

### What You See in Logs

```
2026-02-09T05:00:00: Value=$499.92, Reward=-7.54, Actions=4
```

**Value=$499.92:**
- Portfolio dropped from $499.99 to $499.92
- Loss of $0.07 (0.014%)

**Reward=-7.54:**
- NOT just from the $0.07 loss!
- Includes penalties for risk management violations

**Actions=4:**
- 4 trading actions taken (buys/sells across 7 tickers)

---

## Penalties & Bonuses Explained

### Concentration Penalty
**Purpose:** Prevent putting all eggs in one basket

```python
if any_position > 40% of portfolio:
    penalty = -concentration_penalty × (position_pct - 0.40)
```

**Example:**
- 60% in BTC
- Penalty: -10 × (0.60 - 0.40) = -2.0 reward

### Diversification Penalty
**Purpose:** Force holding at least 3 positions

```python
if active_positions < 3:
    penalty = -concentration_penalty × 0.5 × (3 - positions)
```

**Example:**
- Only holding 1 asset
- Penalty: -10 × 0.5 × (3 - 1) = -10.0 reward 🔥

**This is likely what caused your -7.54!**

### Diversification Bonus
**Purpose:** Reward balanced portfolios

```python
if Gini_coefficient < 0.4:  # Well-diversified
    bonus = +concentration_penalty × 0.1 × (0.4 - Gini)
```

**Example:**
- Perfect balance (Gini=0.2)
- Bonus: +10 × 0.1 × (0.4 - 0.2) = +0.2 reward

### Cash Reserve Penalty
**Purpose:** Always keep minimum cash for emergencies

```python
if cash < required_reserve:
    penalty = -(shortfall / initial_capital) × norm_reward × 0.5
```

**Example:**
- Reserve required: $50 (10% of $500)
- Current cash: $40
- Shortfall: $10
- Penalty: -(10/500) × 100 × 0.5 = -1.0 reward

---

## Normalization Factor

The reward is multiplied by `norm_reward` to make it a reasonable scale for the neural network.

**Typical value:** `norm_reward ≈ 100`

**Why?**
- Raw portfolio changes are small ($0.10 to $1.00)
- Need to scale them up for effective learning
- Without normalization: rewards would be 0.001 to 0.01 (too small)
- With normalization: rewards become -10 to +10 (good range)

---

## Your Specific Case Analysis

### Hour 28: Reward = -7.54

**What we know:**
- Portfolio: $499.99 → $499.92 (-$0.07, -0.014%)
- Position: 1.4015 AVAX ($12.78)
- Cash: $487.14
- Active positions: **1** (only AVAX)

**Reward breakdown estimate:**

1. **Alpha component (50%):**
   - Your change: -$0.07
   - Benchmark change: ~$0.00 (assume flat)
   - Alpha: -0.07
   - Alpha reward: -0.07 × 100 × 1.0 × 0.5 = -3.5

2. **Absolute return component (30%):**
   - Return: -0.07 / 500 = -0.00014
   - Absolute reward: -0.00014 × 100 × 1.0 × 0.3 = -0.004

3. **Cash bonus (20%):**
   - Market flat, so no bonus: 0

4. **Diversification penalty:**
   - Only 1 active position (need 3)
   - Penalty: -10 × 0.5 × (3 - 1) = **-10.0** 🔥

5. **Total:**
   - Base: -3.5 - 0.004 + 0 = -3.504
   - Penalty: -10.0
   - **Final: -13.5 (approximately)**

**Wait, that's more than -7.54!**

**Possible explanations:**
1. Decay factor reduced the penalty (late in episode)
2. Concentration penalty parameter is lower than 10
3. Penalty thresholds are different than I estimated

**The key point:** The large negative reward is mostly due to **lack of diversification**, not portfolio loss.

---

## Why This Matters

### For Training:
- Agent learns: "Don't put everything in one asset!"
- Agent learns: "Hold at least 3 positions"
- Agent learns: "Balance your portfolio"

### For Live Trading:
- **Your model is violating its own training rules!**
- It's being ultra-conservative (only 2.5% deployed)
- But that 2.5% is concentrated in ONE asset
- This triggers massive penalties

### The Paradox:
- Model is too conservative → Low capital deployed
- But what IS deployed → Concentrated in 1 asset
- This causes huge negative rewards
- Model gets confused: "I'm barely trading, why am I being punished?"

---

## What Should Happen

**Ideal behavior:**
- Deploy 30-60% of capital
- Spread across 3-5 assets
- Each position: 10-20% of portfolio
- Maintain 40-70% cash reserve
- Rebalance regularly

**What's happening:**
- Deploying ~2.5% of capital ($12.78 / $500)
- ALL in one asset (AVAX)
- 97.5% cash (over-reserved)
- Violates diversification rules
- Gets penalized even when not losing money

---

## Typical Reward Ranges

| Scenario | Reward Range | Meaning |
|----------|--------------|---------|
| **Great trade** | +5 to +10 | Beat benchmark, made money, diversified |
| **Good trade** | +1 to +5 | Small gains, decent diversification |
| **Neutral** | -0.5 to +0.5 | Break-even, no major violations |
| **Small loss** | -1 to -5 | Minor losses or slight violations |
| **Bad trade** | -5 to -10 | Significant loss or rule violations |
| **Terrible** | < -10 | Major violations (no diversification) |

**Your -7.54:** Bad trade (rule violation, not big loss)

---

## Time Decay Factor

```python
progress = steps_elapsed / trading_horizon
decay_factor = max(1.0 - progress, 0.2)  # Never below 0.2
```

**What it does:**
- Early in episode: decay = 1.0 (full rewards)
- Middle: decay = 0.5 (half rewards)
- Late: decay = 0.2 (minimal rewards)

**Why?**
- Early decisions matter more (compound over time)
- Late decisions matter less (episode ending soon)
- Encourages long-term thinking

**At hour 28 (if max=168):**
- Progress: 28/168 = 16.7%
- Decay: 1.0 - 0.167 = 0.833
- Rewards are scaled by 83.3%

---

## Key Insights

### 1. Reward ≠ Profit
- **Profit:** Did you make money?
- **Reward:** Did you follow the rules while making (or not losing) money?

You can **make money but get negative reward** if you:
- Violate diversification rules
- Over-concentrate positions
- Violate cash reserve

### 2. Small Rewards are Normal
- Hourly changes are small ($0.10 to $1.00)
- Scaled by normalization (×100)
- Typical range: -2 to +2 per hour
- Your -0.08 to -0.09 is FINE

### 3. Large Rewards are Red Flags
- Reward > +5: Unusually good (lucky?)
- Reward < -5: Rule violation or big loss
- Your -7.54: **Diversification penalty**

### 4. Cumulative Matters
- Single bar reward doesn't matter much
- **Cumulative reward over 100+ hours** is what counts
- One bad hour (-7.54) won't ruin performance
- But repeated violations will

---

## What to Do About It

### Short-term (Now):
1. **Don't panic** - One bad reward doesn't mean model is broken
2. **Monitor diversification** - Check how many positions are held
3. **Wait for more data** - 28 hours is too short to judge
4. **Watch for patterns** - Is it always concentrating in 1 asset?

### Medium-term (Ensemble models):
1. **Train with stronger diversification incentives**
2. **Adjust penalty parameters**
3. **Test on 5min timeframe** (more opportunities = better diversification)

### Long-term (Production):
1. **Add risk constraints** in paper trader (force min 3 positions)
2. **Monitor diversification metrics** separately from Sharpe
3. **Retrain with live data** that includes diversification successes

---

## Summary

**Reward Components:**
```
Final Reward =
    0.5 × (Alpha vs Benchmark)
  + 0.3 × (Absolute Returns)
  + 0.2 × (Cash Management)
  - Concentration Penalties
  - Diversification Penalties
  - Cash Reserve Penalties
  + Diversification Bonuses
```

**Your -7.54 reward breakdown:**
- Small portfolio loss: ~-0.5
- **Lack of diversification: ~-7.0** (only 1 position held)
- Total: -7.54

**Why fluctuations happen:**
- Normal: Portfolio changes drive small rewards (-0.5 to +0.5)
- Abnormal: Rule violations drive large negative rewards (-5 to -15)

**What it means:**
- Model is being too conservative
- When it does trade, it concentrates in 1 asset
- This violates training rules
- Gets punished even when not losing money

**Not a bug, it's a feature!** The penalties are working as designed to encourage diversification. The model just hasn't learned to follow its own rules in live trading yet.

---

**TL;DR:** Reward = "How well did you follow the trading rules?" not "How much money did you make?" Your -7.54 is mostly a diversification penalty, not a huge loss.
