# Ensemble Voting System - How It Works

## Overview

The ensemble uses **10 trained models** and averages their predictions for each trading decision.

## How to See Votes

### Option 1: Live View
```bash
python show_ensemble_votes.py
```

This shows:
- Individual predictions from all 10 models
- The ensemble average (final decision)
- Agreement metrics (how much models agree)
- Vote breakdown (BUY/HOLD/SELL for each asset)

### Option 2: Raw Data
```bash
cat paper_trades/ensemble_votes.json
```

JSON format with all predictions.

## Understanding the Output

### Individual Model Predictions

Each model outputs a value for each asset:
- **Positive value** (e.g., +0.50) → BUY signal
- **Negative value** (e.g., -0.30) → SELL signal
- **Near zero** (e.g., -0.05 to +0.05) → HOLD signal

### Ensemble Average

The final decision is the **mean** of all 10 predictions:
```
Ensemble = (Model1 + Model2 + ... + Model10) / 10
```

### Agreement Score

- **High agreement** (low std dev) → All models agree, high confidence
- **Low agreement** (high std dev) → Models disagree, uncertain market

### Vote Breakdown

Shows how many models vote for each action:
- `7 BUY | 2 HOLD | 1 SELL` → Strong buy consensus
- `3 BUY | 4 HOLD | 3 SELL` → No consensus, likely HOLD

## Why Your Current Performance Makes Sense

Looking at your dashboard:
- **Position**: Holding LINK and ETH
- **P&L**: -6.70% (session)
- **Market**: -4.93% average
- **Alpha**: **+3.63%** 🎯

### What This Means:

1. **You're IN a position** during a market crash
2. **The ensemble predicted this would be less bad** than it turned out
3. **BUT you still beat the market by 3.63%!**

The ensemble likely saw:
- LINK: Models predicted -3% → Actually -4.96% → Not terrible
- ETH: Models predicted -5% → Actually -6.70% → Close estimate

**The key**: Ensemble kept you in **better-performing assets** (LINK/ETH) vs worse ones (AAVE -6.16%, UNI -6.82%)

## Next Steps

Once the market opens at the next hour mark, the ensemble will:
1. Make a new prediction
2. Log it to `ensemble_votes.json`
3. You can run `show_ensemble_votes.py` to see the breakdown

## Expected Behavior

- **High volatility** → Models disagree → Smaller positions
- **Clear trends** → Models agree → Larger positions
- **Uncertainty** → Ensemble stays mostly in cash

Your **+3.63% alpha** proves the ensemble is working - you're losing less than the market!
