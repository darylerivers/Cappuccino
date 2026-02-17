# Scripts That Need Adjustment

Based on our session, here are scripts that should be updated:

## High Priority - Active Issues

### 1. **1_optimize_unified.py**
**Issues:**
- No error handling for GPU OOM
- Worker count hardcoded instead of using config
- Missing logging for trial failures

**Adjustments needed:**
```python
# Add at top:
from constants import TRAINING

# Use config for workers:
n_workers = TRAINING.get('WORKERS', 1)  # Instead of hardcoded 10

# Add GPU OOM handling:
try:
    # Training code
except torch.cuda.OutOfMemoryError:
    logger.error("GPU OOM - reduce batch size or workers")
    # Cleanup and retry with smaller settings
```

### 2. **paper_trader_alpaca_polling.py**
**Issues:**
- No Discord notifications
- Limited error logging
- No position size validation before trades

**Adjustments needed:**
- Add Discord webhook integration for trade notifications
- Add position size checks against concentration limits
- Better error recovery for API failures

### 3. **environment_Alpaca.py**
**Issues:**
- Fee calculation could be clearer
- Position concentration logic complex
- No validation of initial_capital parameter

**Adjustments needed:**
```python
# Add validation in __init__:
if initial_capital is None:
    initial_capital = TRADING.INITIAL_CAPITAL
if initial_capital <= 0:
    raise ValueError(f"Invalid initial_capital: {initial_capital}")
```

### 4. **constants.py**
**Issues:**
- Missing Discord configuration
- No training worker configuration
- Fee tiers could be more documented

**Adjustments needed:**
```python
# Add Discord config:
DISCORD = SimpleNamespace(
    WEBHOOK_URL="",  # To be filled
    BOT_TOKEN="",    # To be filled
    CHANNEL_ID="",   # To be filled
    ENABLE_NOTIFICATIONS=True
)

# Add training config:
TRAINING = SimpleNamespace(
    WORKERS=1,  # Adjust based on GPU (10 for RX 7900 GRE)
    BATCH_SIZE=32768,
    MAX_VRAM_GB=8  # Will be 16 after GPU upgrade
)
```

### 5. **dashboard.py**
**Issues:**
- Page 6 (ensemble votes) slow to load
- No error handling for missing files
- Could integrate Discord status

**Adjustments needed:**
- Cache ensemble vote data
- Add try/except for file reads
- Add "Send to Discord" button for alerts

---

## Medium Priority - Improvements

### 6. **auto_model_deployer.py**
**Issues:**
- No notification when deploying new model
- Could check if model is better than current before deploying

**Adjustments needed:**
- Add Discord notification for deployments
- Add performance comparison before deploy

### 7. **performance_monitor.py**
**Issues:**
- Should send alerts to Discord
- Alpha decay detection could notify sooner

**Adjustments needed:**
- Integrate Discord webhooks for alerts
- Add configurable alert thresholds

### 8. **system_watchdog.py**
**Issues:**
- Only logs errors, doesn't notify
- Could restart failed processes automatically

**Adjustments needed:**
- Discord notifications for system issues
- Auto-restart with backoff strategy

---

## Low Priority - Nice to Have

### 9. **function_CPCV.py**
**Issues:**
- Variable names could be more descriptive
- Missing docstrings on some functions

**Adjustments needed:**
- Add comprehensive docstrings
- Rename cryptic variables (e.g., `s`, `e` → `start_idx`, `end_idx`)

### 10. **show_ensemble_votes.py**
**Issues:**
- Could send vote summary to Discord
- No command-line arguments for filtering

**Adjustments needed:**
- Add `--discord` flag to post to Discord
- Add `--symbol` flag to filter by ticker

---

## Scripts to Create

### 11. **discord_bot.py** (NEW)
Monitor and control trading system via Discord

### 12. **config/discord_config.py** (NEW)
Discord bot configuration

### 13. **integrations/discord_notifier.py** (NEW)
Discord notification helper class

---

## Recommended Order of Adjustments

1. **First:** Add Discord bot infrastructure (see discord_bot.py below)
2. **Second:** Update constants.py with Discord config
3. **Third:** Add Discord notifications to paper_trader_alpaca_polling.py
4. **Fourth:** Add Discord notifications to performance_monitor.py
5. **Fifth:** Add error handling to 1_optimize_unified.py
6. **Sixth:** Improve dashboard.py with Discord integration

---

## Quick Wins (Do First)

These are small changes with big impact:

### constants.py
```python
# Add Discord section
DISCORD = SimpleNamespace(
    WEBHOOK_URL=os.getenv('DISCORD_WEBHOOK_URL', ''),
    BOT_TOKEN=os.getenv('DISCORD_BOT_TOKEN', ''),
    TRADE_CHANNEL_ID=int(os.getenv('DISCORD_TRADE_CHANNEL', '0')),
    ALERT_CHANNEL_ID=int(os.getenv('DISCORD_ALERT_CHANNEL', '0')),
    ENABLED=os.getenv('DISCORD_ENABLED', 'false').lower() == 'true'
)
```

### paper_trader_alpaca_polling.py
```python
# Add at top
from integrations.discord_notifier import DiscordNotifier

# In __init__:
self.discord = DiscordNotifier() if DISCORD.ENABLED else None

# After each trade:
if self.discord:
    self.discord.send_trade_notification(
        symbol=symbol,
        action=action,
        quantity=quantity,
        price=price,
        portfolio_value=self.portfolio_value
    )
```

### 1_optimize_unified.py
```python
# Add GPU OOM handling
try:
    study.optimize(objective, n_trials=100, n_jobs=n_workers)
except torch.cuda.OutOfMemoryError:
    logger.error("GPU OOM - reducing workers and retrying")
    study.optimize(objective, n_trials=100, n_jobs=max(1, n_workers//2))
```

---

## Summary

**Must adjust (for Discord bot):**
1. constants.py - Add Discord config
2. paper_trader_alpaca_polling.py - Add trade notifications
3. performance_monitor.py - Add alert notifications
4. Create discord_bot.py (see below)

**Should adjust (for stability):**
5. 1_optimize_unified.py - GPU OOM handling
6. environment_Alpaca.py - Input validation
7. dashboard.py - Error handling

**Nice to adjust (for quality):**
8. auto_model_deployer.py - Deployment notifications
9. function_CPCV.py - Better documentation
10. show_ensemble_votes.py - CLI improvements
