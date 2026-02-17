# Ensemble Voting Dashboard

A live dashboard that shows what each model in the ensemble voted on, plus their performance metrics.

## What It Shows

### 1. Model Performance Metrics
- Trial number for each model
- Training performance value (reward)
- Rank within the ensemble (1-10)
- Percentile (all models are top 10% from training)
- Ensemble mean and best values

### 2. Individual Model Predictions
- Shows the raw prediction from each of the 10 models
- Color-coded: 🟢 Green = BUY, 🔴 Red = SELL, White = HOLD
- Displays predictions for all 7 cryptocurrencies (BTC, ETH, LTC, BCH, LINK, UNI, AAVE)

### 3. Ensemble Average (Final Decision)
- The averaged prediction across all 10 models
- This is what the paper trader actually executes

### 4. Consensus Analysis
- Mean action value across models
- Standard deviation (lower = more agreement)
- Range of predictions (min to max)
- Agreement score (0-1, higher = better consensus)
- Vote breakdown (how many models voted BUY/HOLD/SELL)

## How to Use

### Run the voting dashboard:
```bash
python dashboard_ensemble_votes.py
```

### Or run the simple text version:
```bash
python show_ensemble_votes.py
```

## Understanding the Data

### Model Values
The "Value" shown for each model is its performance during training on validation data:
- **Higher values = better performance**
- Model 1 (Trial #756): +0.012626 is the best performer
- Model 10 (Trial #633): +0.010293 is still in top 10%

### Action Values
- **Positive values** (e.g., +0.2786 for LINK): BUY signal
- **Negative values** (e.g., -0.3845 for BTC): SELL signal
- **Near zero** (e.g., -0.0430 for UNI): HOLD signal

The magnitude indicates confidence:
- Large absolute value (>0.2): Strong signal
- Medium absolute value (0.1-0.2): Moderate signal
- Small absolute value (<0.1): Weak signal

### Agreement Score
- **0.999**: Nearly perfect consensus (all models agree)
- **0.990-0.995**: High consensus (small variation)
- **<0.990**: Lower consensus (models have different opinions)

## Example Output

```
MODEL PERFORMANCE METRICS
Model 1  │ #756  │ +0.012626 │ 1 │ Top 99%   ← Best model
Model 2  │ #737  │ +0.012550 │ 2 │ Top 98%
...

INDIVIDUAL MODEL PREDICTIONS
Model 1    │  -0.3897 │  -0.2020 │  +0.2766 ← Model 1's vote
Model 2    │  -0.3786 │  -0.1973 │  +0.2806 ← Model 2's vote
...

CONSENSUS ANALYSIS
LINK     │ +0.2786 │ Agreement=0.992 │ 10 BUY | 0 HOLD | 0 SELL
  ↑          ↑           ↑                  ↑
Asset   Average    Consensus     All 10 models agree to BUY
```

## When Is Data Updated?

The voting data is updated:
- **Every time** the ensemble makes a trading decision
- On **1-hour timeframes** (currently configured)
- Next update will be at the top of the next hour (e.g., 17:00, 18:00, etc.)

The dashboard refreshes every 3 seconds and shows how long ago the last decision was made.

## Files

- `dashboard_ensemble_votes.py` - Live updating dashboard (color, formatted)
- `show_ensemble_votes.py` - Simple one-time display
- `paper_trades/ensemble_votes.json` - Raw voting data (JSON)
- `train_results/ensemble/ensemble_manifest.json` - Model metadata

## Tips

1. **High consensus is good**: When all models agree (agreement > 0.99), the signal is more reliable
2. **Watch for divergence**: If agreement drops below 0.99, models disagree - treat signals with caution
3. **Model diversity**: The 10 models were trained with different random seeds, so slight variation is expected
4. **Performance weighting**: Currently all models have equal weight, but Model 1 (best) and Model 10 (10th best) both get one vote
