# Paper Trading - Model Compatibility Guide

## Quick Start

Paper trading is now **model-aware** and will auto-configure to match your trained model.

### Basic Usage

```bash
# Use the model inspector to see what a model needs:
python model_inspector.py train_results/cwd_tests/trial_XXXX_1h --generate-command

# Or just run directly with any tickers - it will auto-correct:
python paper_trader_alpaca_polling.py \
    --model-dir train_results/cwd_tests/trial_3358_1h \
    --tickers BTC/USD ETH/USD \  # Will be corrected to 7 tickers automatically
    --poll-interval 60 \
    --history-hours 24
```

### With Fail-Safe (Recommended)

```bash
# Auto-restart on crashes, with exponential backoff
./paper_trading_failsafe.sh train_results/cwd_tests/trial_3358_1h
```

---

## How Model Compatibility Works

### What the System Auto-Detects

1. **Number of tickers** - Extracted from model action dimension
2. **Network dimension** - Read from checkpoint (may differ from trial params)
3. **Ticker list** - Falls back to training tickers if mismatch

### Example

```bash
# You request 2 tickers, but model was trained on 7
python paper_trader_alpaca_polling.py \
    --model-dir train_results/cwd_tests/trial_3358_1h \
    --tickers BTC/USD ETH/USD

# Output:
# 🔍 Model expects 7 tickers
# ⚠️  Requested 2 tickers, model needs 7
# ✓ Using training tickers: ['AAVE/USD', 'AVAX/USD', 'BTC/USD', 'LINK/USD', 'ETH/USD', 'LTC/USD', 'UNI/USD']
```

---

## Model Inspector Tool

Inspect any model to see its requirements:

```bash
python model_inspector.py train_results/cwd_tests/trial_3358_1h
```

**Output:**
```
================================================================================
MODEL CONFIGURATION
================================================================================

📊 Architecture:
  Network dimension: 1984
  State dimension:   239
  Action dimension:  7
  Number of tickers: 7

💰 Tickers:
  1. AAVE/USD
  2. AVAX/USD
  3. BTC/USD
  4. LINK/USD
  5. ETH/USD
  6. LTC/USD
  7. UNI/USD
```

### Generate Commands

```bash
python model_inspector.py train_results/cwd_tests/trial_3358_1h --generate-command
```

Outputs the exact command needed to run paper trading with this model.

---

## Fail-Safe Wrapper

The fail-safe wrapper provides:
- ✅ Automatic restarts on crashes
- ✅ Exponential backoff (5s → 300s max)
- ✅ Model validation before start
- ✅ Detailed logging
- ✅ Health monitoring

### Usage

```bash
# Start with default model
./paper_trading_failsafe.sh

# Or specify model
./paper_trading_failsafe.sh train_results/cwd_tests/trial_XXXX_1h

# Check status
tail -f logs/paper_trading_failsafe.log
tail -f logs/paper_trading_live.log

# Stop
pkill -f paper_trading_failsafe
```

### Features

**Restart Logic:**
- Attempt 1: Wait 5s
- Attempt 2: Wait 10s
- Attempt 3: Wait 20s
- Attempt 4: Wait 40s
- ... (exponential up to 300s max)

**Reset on Success:**
- If process runs >60s successfully, backoff resets to 5s

**State Tracking:**
- Restart counts saved to `deployments/paper_trading_state.json`
- Can resume after system reboot

---

## Current Limitations

### Ticker Count Must Match

**Models are trained on a specific number of tickers.** While the system will auto-correct your ticker list, it cannot:
- Train a 7-ticker model to work with 3 tickers
- Make a 3-ticker model work with 7 tickers

The model dimensions are baked into the neural network weights.

### Workaround

Use the **model inspector** to check ticker count first:

```bash
python model_inspector.py train_results/cwd_tests/trial_XXXX_1h | grep "Number of tickers"
```

Then ensure your command uses the correct count (order can vary, but count must match).

---

## Files Created

| File | Purpose |
|------|---------|
| `model_inspector.py` | Extract model configuration from checkpoints |
| `paper_trading_failsafe.sh` | Auto-restart wrapper with backoff |
| `create_trial_pickle.py` | Generate trial metadata for deployment |
| `PAPER_TRADING_GUIDE.md` | This guide |

---

## Troubleshooting

### "Size mismatch" errors

```
RuntimeError: size mismatch for net.0.weight: copying a param with shape torch.Size([1984, 239])
```

**Cause:** Wrong number of tickers

**Solution:**
```bash
# Check what model needs
python model_inspector.py <model_dir>

# Use correct ticker count
python paper_trader_alpaca_polling.py --model-dir <model_dir> --tickers $(python model_inspector.py <model_dir> --generate-command | grep -oP '(?<=--tickers ).*(?= --)')
```

### "Missing best_trial" error

**Cause:** Model directory missing trial metadata

**Solution:**
```bash
python create_trial_pickle.py --trial-id 3358 --output-dir train_results/cwd_tests/trial_3358_1h
```

### Paper trading keeps restarting

**Check logs:**
```bash
tail -100 logs/paper_trading_live.log
tail -50 logs/paper_trading_failsafe.log
```

**Common causes:**
1. API credentials not set (`ALPACA_API_KEY`, `ALPACA_API_SECRET`)
2. Network issues
3. Data download failures

---

## Best Practices

1. **Always use model inspector first** to understand model requirements
2. **Use fail-safe wrapper in production** for automatic recovery
3. **Monitor logs regularly** to catch issues early
4. **Test with `--history-hours 24`** first (faster startup)
5. **Use exactly the training tickers** for best results

---

## Example Workflow

```bash
# 1. Find best model from training
sqlite3 databases/optuna_cappuccino.db "
SELECT trial_id, value
FROM trial_values tv
JOIN trials t ON tv.trial_id = t.trial_id
WHERE t.state = 'COMPLETE'
ORDER BY value DESC
LIMIT 5"

# 2. Inspect the model
python model_inspector.py train_results/cwd_tests/trial_3358_1h

# 3. Create trial metadata if needed
python create_trial_pickle.py --trial-id 3358 --output-dir train_results/cwd_tests/trial_3358_1h

# 4. Start paper trading with fail-safe
./paper_trading_failsafe.sh train_results/cwd_tests/trial_3358_1h

# 5. Monitor
tail -f logs/paper_trading_live.log
```

---

**Last Updated:** 2025-11-15 22:30 UTC
