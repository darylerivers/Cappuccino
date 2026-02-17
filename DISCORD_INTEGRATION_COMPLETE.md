# Discord Integration - Implementation Complete

**Date**: February 8, 2026
**Status**: ✅ Ready for testing

## What Was Implemented

### 1. Paper Trader Integration

Modified `scripts/deployment/paper_trader_alpaca_polling.py` to add Discord notifications:

**Startup Notifications** (line ~1445)
- Sends embed when paper trader starts
- Shows trial number, tickers, timeframe, risk settings
- 🚀 Green embed with session details

**Trade Notifications** (new method `_send_trade_notifications`)
- Detects significant position changes (>$5 trade value)
- Sends BUY notifications (📈 green) and SELL notifications (📉 red)
- Shows quantity, price, total value, portfolio value
- Only sends for meaningful trades to avoid spam

**Error Notifications** (line ~1480)
- Catches exceptions in main trading loop
- Sends error alerts to Discord with details
- ❌ Red alert with error type and message

### 2. Infrastructure

**Already Available** (no changes needed):
- `integrations/discord_notifier.py` - Full-featured webhook notifier
- `discord_bot.py` - Interactive bot with commands (!status, !portfolio, etc.)
- `constants.py` - Discord configuration loaded from environment
- `.env.discord.template` - Template for environment variables

**New Files Created**:
- `DISCORD_SETUP.md` - Complete setup guide with screenshots instructions
- `test_discord.py` - Test suite to verify notifications are working

### 3. Notification Types Supported

#### Via Webhook (DiscordNotifier)
- ✅ Trade executions (buy/sell)
- ✅ System alerts (errors, warnings)
- ✅ Portfolio summaries
- ✅ Training updates
- ✅ Deployment notifications
- ✅ System status

#### Via Bot Commands (discord_bot.py)
- `!status` - System overview
- `!portfolio` - Portfolio details
- `!trades` - Recent trades
- `!trader start/stop/status` - Control paper trader
- `!training status` - Training progress
- `!help` - Command list

## Quick Start Guide

### Step 1: Create Discord Webhook

1. Open Discord, go to your server
2. Right-click the channel for notifications (e.g., `#trading-alerts`)
3. **Edit Channel** → **Integrations** → **Webhooks**
4. Click **New Webhook**
5. Name it "Cappuccino Bot"
6. **Copy the Webhook URL**

### Step 2: Configure Environment

Create `.env` file in `/opt/user-data/experiment/cappuccino`:

```bash
# Discord Webhook - REQUIRED
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_TOKEN
DISCORD_ENABLED=true

# Optional: For interactive bot commands
DISCORD_BOT_TOKEN=your_bot_token_here
DISCORD_TRADE_CHANNEL_ID=123456789012345678
DISCORD_ALERT_CHANNEL_ID=987654321098765432
```

### Step 3: Test Integration

```bash
cd /opt/user-data/experiment/cappuccino
python test_discord.py
```

Expected output:
```
✅ Discord configuration looks good!
✅ Basic notification sent successfully!
✅ Trade notification sent successfully!
✅ Alert notifications sent!
✅ Portfolio summary sent successfully!

Passed: 4/4
🎉 All tests passed! Discord integration is working correctly.
```

### Step 4: Run Paper Trader

Discord notifications are now automatic:

```bash
python scripts/deployment/paper_trader_alpaca_polling.py \
    --deployment-dir train_results/deployment_trial250_20260207_175728
```

Watch Discord for:
- 🚀 Startup notification
- 📈 BUY trades (green)
- 📉 SELL trades (red)
- ❌ Error alerts

## Configuration Options

### Control Notification Types

Edit `constants.py` (line ~196):

```python
class DiscordConstants:
    NOTIFY_TRADES: bool = True       # Trade executions
    NOTIFY_ALERTS: bool = True       # Errors and warnings
    NOTIFY_TRAINING: bool = False    # Training progress (can be spammy)
    NOTIFY_DEPLOYMENTS: bool = True  # Model deployments
```

### Adjust Trade Notification Threshold

Edit `scripts/deployment/paper_trader_alpaca_polling.py` (line ~1019):

```python
# Only notify on significant changes (> $5 or >1% of position)
trade_value = abs(qty_change * price)
if trade_value < 5.0:  # Change this threshold
    continue
```

## Example Notifications

### Startup (Green Embed)
```
🚀 Paper Trading Session Initialized

Trial: #250
Tickers: AAVE/USD, AVAX/USD, BTC/USD, LINK/USD, ETH/USD, LTC/USD, UNI/USD
Timeframe: 1h
Mode: Paper
Risk Management:
  Max position: 30%
  Stop-loss: 5%
```

### Trade Execution (Green/Red Embed)
```
📈 BUY AVAX/USD

Quantity: 0.047603
Price: $9.23
Total: $0.44
Portfolio Value: $499.997894
```

### Error Alert (Red Embed)
```
❌ ERROR Alert
Paper trader encountered an error

Details: KeyError: 'timestamp'
```

## What to Provide

To complete the setup, you need to:

1. **Discord Webhook URL** (required)
   - From your Discord server webhook settings
   - Format: `https://discord.com/api/webhooks/123456789/ABCDEF...`

2. **Discord Bot Token** (optional - only for interactive commands)
   - From https://discord.com/developers/applications
   - Format: `ABC123.XYZ789.etc...`

3. **Channel IDs** (optional - only for bot)
   - Right-click channels in Discord → Copy ID
   - Format: `123456789012345678`

## Next Steps

1. ✅ Implementation complete
2. ⏸️ **YOU**: Create Discord webhook and provide URL
3. ⏸️ **YOU**: Add webhook to `.env` file
4. ⏸️ Test with `python test_discord.py`
5. ⏸️ Restart paper trader to enable notifications

## Security Notes

⚠️ **Important**:
- Never commit `.env` to git (already in `.gitignore`)
- Webhook URLs are sensitive - treat like passwords
- If exposed, regenerate the webhook in Discord

## Files Modified

```
Modified:
  scripts/deployment/paper_trader_alpaca_polling.py  (+65 lines)
    - Import Discord notifier
    - Initialize in __init__
    - Send startup notification
    - Send trade notifications
    - Send error alerts

Created:
  DISCORD_SETUP.md              (Complete setup guide)
  DISCORD_INTEGRATION_COMPLETE.md (This file)
  test_discord.py                (Test suite)
```

## Technical Details

### Integration Points

1. **Startup** (line 1445)
   ```python
   if self.discord:
       self.discord.send_message(...)
   ```

2. **Trades** (line 983)
   ```python
   if self.discord:
       self._send_trade_notifications(...)
   ```

3. **Errors** (line 1480)
   ```python
   if self.discord and DISCORD.NOTIFY_ALERTS:
       self.discord.send_alert(...)
   ```

### Notification Method (line 1007)
```python
def _send_trade_notifications(self, old_holdings, new_holdings, prices, portfolio_value):
    """Send Discord notifications for significant trades."""
    for i, ticker in enumerate(self.tickers):
        qty_change = new_holdings[i] - old_holdings[i]
        trade_value = abs(qty_change * prices[i])

        if trade_value < 5.0:  # Skip small trades
            continue

        action = "BUY" if qty_change > 0 else "SELL"
        self.discord.send_trade_notification(...)
```

## Troubleshooting

### No notifications appearing

```bash
# Check configuration
python test_discord.py

# Check environment
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('URL:', 'SET' if os.getenv('DISCORD_WEBHOOK_URL') else 'NOT SET'); print('Enabled:', os.getenv('DISCORD_ENABLED'))"

# Test webhook manually
curl -X POST "YOUR_WEBHOOK_URL" -H "Content-Type: application/json" -d '{"content": "Test"}'
```

### Too many notifications

1. Increase trade threshold (edit line 1019 in paper_trader_alpaca_polling.py)
2. Disable specific notification types in constants.py

### Bot commands not working

1. Enable "Message Content Intent" in Discord Developer Portal
2. Verify bot has permissions in channel
3. Use webhook-only mode instead (simpler, no bot needed)

---

**Status**: Ready for testing! Just need Discord webhook URL from you.

**Questions?** See `DISCORD_SETUP.md` for detailed setup instructions.
