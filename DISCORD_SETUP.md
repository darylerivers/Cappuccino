# Discord Integration Setup Guide

The Cappuccino trading system supports Discord notifications for monitoring your paper trader and training progress.

## Quick Start (Webhook Only - Recommended)

The simplest setup uses Discord webhooks for one-way notifications:

### 1. Create a Discord Webhook

1. Open Discord and go to your server
2. Right-click on the channel where you want notifications (e.g., `#trading-alerts`)
3. Select **Edit Channel** → **Integrations** → **Webhooks**
4. Click **New Webhook**
5. Give it a name (e.g., "Cappuccino Bot")
6. Copy the **Webhook URL**

### 2. Configure Environment Variables

Create or edit `.env` file in the cappuccino directory:

```bash
# Discord Webhook Configuration
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_WEBHOOK_TOKEN
DISCORD_ENABLED=true
```

### 3. Test the Integration

```bash
# Test notifications
python -c "
from integrations.discord_notifier import DiscordNotifier
notifier = DiscordNotifier()
notifier.send_message('🤖 Cappuccino Discord integration is working!')
"
```

### 4. Start Paper Trading

Discord notifications are now enabled automatically:

```bash
python scripts/deployment/paper_trader_alpaca_polling.py \
    --deployment-dir train_results/deployment_trial250_20260207_175728
```

## What Gets Notified (Webhook Mode)

✅ **Startup**: When paper trader starts
✅ **Trades**: Buy/sell executions (for trades >$5)
✅ **Errors**: System errors and exceptions
✅ **Risk Events**: Stop-loss triggers, profit-taking, portfolio protection

## Advanced Setup (Interactive Bot)

For two-way interaction with commands like `!status`, `!portfolio`, `!trades`:

### 1. Create a Discord Bot

1. Go to https://discord.com/developers/applications
2. Click **New Application**
3. Give it a name (e.g., "Cappuccino Trading Bot")
4. Go to **Bot** tab → **Add Bot**
5. Enable **Message Content Intent** (under Privileged Gateway Intents)
6. Copy the **Bot Token**

### 2. Invite Bot to Server

1. Go to **OAuth2** → **URL Generator**
2. Select scopes: `bot`
3. Select permissions: `Send Messages`, `Read Messages`, `Embed Links`
4. Copy the generated URL and open it in your browser
5. Select your server and authorize

### 3. Get Channel IDs

1. In Discord, enable **Developer Mode** (Settings → Advanced → Developer Mode)
2. Right-click your channels and select **Copy ID**
3. Save the IDs for your trade and alert channels

### 4. Configure Environment

Add to your `.env`:

```bash
# Discord Bot Configuration
DISCORD_BOT_TOKEN=your_bot_token_here
DISCORD_WEBHOOK_URL=your_webhook_url_here
DISCORD_TRADE_CHANNEL_ID=123456789012345678
DISCORD_ALERT_CHANNEL_ID=987654321098765432
DISCORD_ENABLED=true
```

### 5. Run the Discord Bot

In a separate terminal:

```bash
python discord_bot.py
```

### 6. Available Commands

Once the bot is running, try these commands in Discord:

```
!status              - Show system status
!portfolio           - Show portfolio details
!trades              - Show recent trades
!trader status       - Show paper trader status
!trader start        - Start paper trading
!trader stop         - Stop paper trading
!training status     - Show training progress
!help                - Show all commands
```

## Notification Types

### Trade Notifications (Green/Red Embeds)
- **BUY**: Green embed with 📈 emoji
- **SELL**: Red embed with 📉 emoji
- Shows: quantity, price, total value, portfolio value

### Alert Notifications (Color-coded)
- **Info** (Blue ℹ️): General information
- **Warning** (Orange ⚠️): Non-critical warnings
- **Error** (Red ❌): Errors and failures
- **Critical** (Dark Red 🚨): Critical system failures

### System Status
- Paper trader running/stopped
- Training workers active
- GPU/memory usage
- Portfolio value and positions

## Troubleshooting

### No notifications appearing

1. **Check webhook URL is valid:**
   ```bash
   python -c "
   import os
   from dotenv import load_dotenv
   load_dotenv()
   print('Webhook:', os.getenv('DISCORD_WEBHOOK_URL', 'NOT SET'))
   print('Enabled:', os.getenv('DISCORD_ENABLED', 'NOT SET'))
   "
   ```

2. **Test webhook directly:**
   ```bash
   curl -X POST "YOUR_WEBHOOK_URL" \
     -H "Content-Type: application/json" \
     -d '{"content": "Test message"}'
   ```

3. **Check logs for errors:**
   ```bash
   tail -f logs/paper_trader_trial250.log
   ```

### Bot not responding to commands

1. **Verify Message Content Intent is enabled** in Discord Developer Portal
2. **Check bot has permissions** in the channel (Read/Send Messages)
3. **Restart bot** after changing settings

### Notifications too frequent

Edit `constants.py` to control notification frequency:

```python
class DiscordConstants:
    NOTIFY_TRADES: bool = True      # Trade executions
    NOTIFY_ALERTS: bool = True      # Error alerts
    NOTIFY_TRAINING: bool = False   # Training updates (can be spammy)
```

## Configuration Reference

### Environment Variables (.env)

```bash
# Required for webhook notifications
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
DISCORD_ENABLED=true

# Required for bot commands (optional)
DISCORD_BOT_TOKEN=your_bot_token_here
DISCORD_TRADE_CHANNEL_ID=123456789012345678
DISCORD_ALERT_CHANNEL_ID=987654321098765432
```

### constants.py Settings

Located in `constants.py` ~line 186:

```python
@dataclass(frozen=True)
class DiscordConstants:
    """Discord notification settings."""

    # Loaded from environment
    BOT_TOKEN: str = os.getenv('DISCORD_BOT_TOKEN', '')
    WEBHOOK_URL: str = os.getenv('DISCORD_WEBHOOK_URL', '')
    TRADE_CHANNEL_ID: int = int(os.getenv('DISCORD_TRADE_CHANNEL_ID', '0'))
    ALERT_CHANNEL_ID: int = int(os.getenv('DISCORD_ALERT_CHANNEL_ID', '0'))

    # Feature flags
    ENABLED: bool = os.getenv('DISCORD_ENABLED', 'false').lower() == 'true'
    NOTIFY_TRADES: bool = True       # Trade execution notifications
    NOTIFY_ALERTS: bool = True       # Error and warning alerts
    NOTIFY_TRAINING: bool = True     # Training progress updates
    NOTIFY_DEPLOYMENTS: bool = True  # Model deployment notifications
```

## Example Notifications

### Trade Execution
```
📈 BUY BTC/USD
Quantity: 0.0015
Price: $69,244.22
Total: $103.87
Portfolio Value: $499.98
```

### Error Alert
```
❌ ERROR Alert
Paper trader encountered an error

Details: KeyError: 'timestamp'
```

### System Status
```
📊 System Status
Paper Trader: ✅ Running (PID: 12345)
Training: ❌ Not Running
Portfolio: Total: $499.98, Cash: $497.79, Positions: 1
```

## Security Notes

⚠️ **Important Security Practices:**

1. **Never commit `.env` file** to git (already in `.gitignore`)
2. **Webhook URLs are sensitive** - treat like passwords
3. **Bot tokens are secrets** - never share or commit them
4. **Regenerate tokens** if accidentally exposed
5. **Use separate channels** for different notification types
6. **Limit bot permissions** to only what's needed

## Next Steps

1. ✅ Set up webhook URL (recommended)
2. ✅ Test notifications with paper trader
3. ⏸️ (Optional) Set up interactive bot for commands
4. ⏸️ (Optional) Configure separate channels for trades vs alerts
5. ⏸️ (Optional) Add daily summary notifications

---

**Need help?** Check `discord_bot.py` and `integrations/discord_notifier.py` for implementation details.
