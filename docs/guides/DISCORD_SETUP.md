# Discord Bot Setup Guide

Complete guide to set up Discord notifications and bot control for Cappuccino.

## Part 1: Create Discord Bot

### Step 1: Create Discord Application

1. Go to https://discord.com/developers/applications
2. Click "New Application"
3. Name it "Cappuccino Trading Bot"
4. Click "Create"

### Step 2: Create Bot User

1. In your application, go to "Bot" tab (left sidebar)
2. Click "Add Bot" → "Yes, do it!"
3. Under "Token", click "Copy" to copy your bot token
4. **Save this token securely** - you'll need it later

### Step 3: Configure Bot Permissions

1. Still in "Bot" tab, scroll down to "Privileged Gateway Intents"
2. Enable:
   - ✅ **Message Content Intent** (required for commands)
   - ✅ **Server Members Intent** (optional)

### Step 4: Invite Bot to Your Server

1. Go to "OAuth2" → "URL Generator" (left sidebar)
2. Select scopes:
   - ✅ `bot`
   - ✅ `applications.commands`
3. Select bot permissions:
   - ✅ Send Messages
   - ✅ Embed Links
   - ✅ Read Message History
   - ✅ Add Reactions
4. Copy the generated URL at the bottom
5. Open URL in browser and select your Discord server
6. Click "Authorize"

---

## Part 2: Get Channel IDs

### Enable Developer Mode

1. Open Discord
2. User Settings → Advanced
3. Enable "Developer Mode"

### Get Channel IDs

1. Right-click on your **#trading** channel → "Copy Channel ID"
2. Save as `DISCORD_TRADE_CHANNEL_ID`

3. Right-click on your **#alerts** channel → "Copy Channel ID"
4. Save as `DISCORD_ALERT_CHANNEL_ID`

(Create these channels first if they don't exist)

---

## Part 3: Create Webhook (For Notifications)

1. Right-click on #trading channel → "Edit Channel"
2. Go to "Integrations" tab
3. Click "Create Webhook"
4. Name it "Cappuccino Trades"
5. Click "Copy Webhook URL"
6. Save as `DISCORD_WEBHOOK_URL`

---

## Part 4: Configure Cappuccino

### Option 1: Environment Variables (Recommended)

Create `.env` file in cappuccino directory:

```bash
# Discord Configuration
DISCORD_BOT_TOKEN=your_bot_token_here
DISCORD_WEBHOOK_URL=your_webhook_url_here
DISCORD_TRADE_CHANNEL_ID=123456789012345678
DISCORD_ALERT_CHANNEL_ID=123456789012345678
DISCORD_ENABLED=true
```

Then load in shell:
```bash
export $(cat .env | xargs)
```

### Option 2: Update constants.py

Edit `constants.py`:

```python
# Discord Configuration
DISCORD = SimpleNamespace(
    BOT_TOKEN="your_bot_token_here",
    WEBHOOK_URL="your_webhook_url_here",
    TRADE_CHANNEL_ID=123456789012345678,  # Replace with your ID
    ALERT_CHANNEL_ID=123456789012345678,  # Replace with your ID
    ENABLED=True
)
```

---

## Part 5: Install Dependencies

```bash
pip install discord.py requests psutil
```

---

## Part 6: Test Discord Bot

### Test Bot Connection

```bash
# Make sure .env is loaded or constants.py is updated
python discord_bot.py
```

You should see:
```
INFO - discord_bot - Starting Discord bot...
INFO - discord_bot - Cappuccino Trading Bot#1234 has connected to Discord!
```

In your Discord server, you should see:
> 🤖 **Cappuccino Trading Bot Online**
> Type `!help` for commands

### Test Bot Commands

In Discord, try:
```
!status
!portfolio
!trader status
!help
```

### Test Notifications

```python
# Test webhook notifications
python << EOF
from integrations.discord_notifier import DiscordNotifier

notifier = DiscordNotifier()
notifier.send_alert("Test alert from Cappuccino!")
notifier.send_trade_notification(
    symbol='BTC/USD',
    action='BUY',
    quantity=0.01,
    price=45000.0,
    portfolio_value=500.0
)
EOF
```

You should see messages in your #trading channel!

---

## Part 7: Integrate with Paper Trader

### Update paper_trader_alpaca_polling.py

Add at the top:
```python
from integrations.discord_notifier import DiscordNotifier
```

In `__init__`:
```python
# Discord notifications
self.discord = DiscordNotifier()
```

After each trade (in `execute_trade` method):
```python
# Send Discord notification
if self.discord:
    self.discord.send_trade_notification(
        symbol=symbol,
        action=action,
        quantity=quantity,
        price=filled_price,
        portfolio_value=self.get_portfolio_value()
    )
```

---

## Part 8: Add to Other Scripts

### 1_optimize_unified.py (Training Updates)

```python
from integrations.discord_notifier import DiscordNotifier

discord = DiscordNotifier()

# After each trial
discord.send_training_update(
    trial_number=trial.number,
    sharpe_ratio=trial.value,
    total_return=metrics['total_return'],
    status='completed'
)
```

### auto_model_deployer.py (Deployment Alerts)

```python
from integrations.discord_notifier import DiscordNotifier

discord = DiscordNotifier()

# After deployment
discord.send_deployment_notification(
    trial_number=best_trial,
    sharpe_ratio=sharpe,
    deployment_dir=deployment_dir
)
```

### performance_monitor.py (System Alerts)

```python
from integrations.discord_notifier import DiscordNotifier

discord = DiscordNotifier()

# When alpha decay detected
discord.send_alert(
    "Alpha decay detected! Sharpe ratio below threshold.",
    level="warning",
    details=f"Current: {current_sharpe:.4f}, Threshold: {threshold:.4f}"
)
```

---

## Part 9: Run Discord Bot

### Start Bot

```bash
# In a tmux/screen session
python discord_bot.py
```

Or as a service:
```bash
# Create systemd service
sudo nano /etc/systemd/system/cappuccino-discord.service
```

```ini
[Unit]
Description=Cappuccino Discord Bot
After=network.target

[Service]
Type=simple
User=mrc
WorkingDirectory=/opt/user-data/experiment/cappuccino
Environment="DISCORD_BOT_TOKEN=your_token_here"
Environment="DISCORD_WEBHOOK_URL=your_webhook_here"
ExecStart=/home/mrc/.pyenv/versions/finrl-crypto/bin/python discord_bot.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable cappuccino-discord
sudo systemctl start cappuccino-discord
sudo systemctl status cappuccino-discord
```

---

## Discord Commands Reference

### Status Commands
- `!status` - Overall system status
- `!portfolio` - Portfolio details
- `!trades` - Recent trades (default: 5)
- `!trades 10` - Last 10 trades

### Trader Controls
- `!trader status` - Paper trader status
- `!trader start` - Start paper trading
- `!trader stop` - Stop paper trading

### Training
- `!training status` - Training progress

### Utilities
- `!dashboard` - Get dashboard link
- `!help` - Show all commands

---

## Example Notifications

### Trade Notification
> 📈 **BUY BTC/USD**
> Quantity: 0.010000
> Price: $45,000.00
> Total: $450.00
> Portfolio Value: $500.00

### Alert Notification
> ⚠️ **WARNING Alert**
> Alpha decay detected! Sharpe ratio below threshold.
> **Details:** Current: 0.0012, Threshold: 0.0020

### Training Update
> 🧠 **Training Update - Trial #250**
> Status: Completed
> Sharpe Ratio: 0.0043
> Total Return: 12.45%

### Deployment
> 🚀 **New Model Deployed**
> Trial: #250
> Sharpe Ratio: 0.0043
> Directory: deployments/trial_250_live

---

## Troubleshooting

### Bot Not Responding
```bash
# Check if bot is running
ps aux | grep discord_bot.py

# Check logs
tail -f logs/discord_bot.log
```

### Notifications Not Sending
```python
# Test webhook
python << EOF
from integrations.discord_notifier import DiscordNotifier
n = DiscordNotifier()
print(f"Webhook configured: {n.enabled}")
n.send_alert("Test")
EOF
```

### Invalid Token
```bash
# Regenerate token in Discord Developer Portal
# Update .env or constants.py
# Restart bot
```

---

## Security Notes

1. **Never commit tokens to git**:
   ```bash
   echo ".env" >> .gitignore
   ```

2. **Use environment variables** in production

3. **Restrict bot permissions** to only what's needed

4. **Rotate tokens** periodically

---

## Summary

After setup, you'll have:
- ✅ Discord bot monitoring your system
- ✅ Trade notifications in Discord
- ✅ Alert notifications for errors
- ✅ Commands to control trader from Discord
- ✅ Training progress updates
- ✅ Portfolio summaries on demand

**Test everything:**
```bash
# Start bot
python discord_bot.py

# In Discord:
!status
!trader status
!portfolio
```

Your trading system is now Discord-enabled! 🚀
