# Discord Integration Summary

## What You Got

### 1. Discord Bot (`discord_bot.py`)
**Full-featured bot for controlling and monitoring your trading system**

Commands:
- `!status` - System status (trader, training, portfolio)
- `!portfolio` - Detailed portfolio view
- `!trades [limit]` - Recent trades
- `!trader start/stop/status` - Control paper trader
- `!training status` - Training progress
- `!dashboard` - Dashboard link
- `!help` - All commands

### 2. Discord Notifier (`integrations/discord_notifier.py`)
**Helper class for sending notifications from any script**

Features:
- Trade notifications (with embeds)
- Alert notifications (info, warning, error, critical)
- Training updates
- Deployment notifications
- Portfolio summaries
- System status

### 3. Documentation
- **DISCORD_SETUP.md** - Complete setup guide
- **SCRIPTS_TO_ADJUST.md** - List of scripts to modify
- **DISCORD_INTEGRATION_SUMMARY.md** - This file

---

## Quick Start

### 1. Create Discord Bot (5 minutes)
```bash
# Follow DISCORD_SETUP.md Part 1-3:
# 1. Create bot at https://discord.com/developers/applications
# 2. Get bot token
# 3. Get channel IDs
# 4. Create webhook
```

### 2. Configure (2 minutes)
```bash
# Create .env file
cat > .env << 'EOF'
DISCORD_BOT_TOKEN=your_token_here
DISCORD_WEBHOOK_URL=your_webhook_here
DISCORD_TRADE_CHANNEL_ID=123456789012345678
DISCORD_ALERT_CHANNEL_ID=123456789012345678
DISCORD_ENABLED=true
EOF

# Load environment
export $(cat .env | xargs)
```

### 3. Install Dependencies (1 minute)
```bash
pip install discord.py requests psutil
```

### 4. Test Bot (1 minute)
```bash
python discord_bot.py

# In Discord, type:
# !status
# !help
```

### 5. Integrate with Scripts (10-30 minutes)
See **SCRIPTS_TO_ADJUST.md** for detailed changes.

**Priority order:**
1. constants.py - Add Discord config
2. paper_trader_alpaca_polling.py - Trade notifications
3. performance_monitor.py - Alert notifications
4. auto_model_deployer.py - Deployment notifications
5. 1_optimize_unified.py - Training updates

---

## Example Integrations

### Paper Trader (Trade Notifications)

```python
# paper_trader_alpaca_polling.py
from integrations.discord_notifier import DiscordNotifier

class PaperTrader:
    def __init__(self):
        self.discord = DiscordNotifier()

    def execute_trade(self, symbol, action, quantity, price):
        # ... existing trade logic ...

        # Send Discord notification
        if self.discord:
            self.discord.send_trade_notification(
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=price,
                portfolio_value=self.total_value
            )
```

### Training Updates

```python
# 1_optimize_unified.py
from integrations.discord_notifier import DiscordNotifier

discord = DiscordNotifier()

def objective(trial):
    # ... training logic ...

    # Send update for successful trials
    if sharpe_ratio > 0:
        discord.send_training_update(
            trial_number=trial.number,
            sharpe_ratio=sharpe_ratio,
            total_return=total_return,
            status='completed'
        )
```

### System Alerts

```python
# performance_monitor.py
from integrations.discord_notifier import DiscordNotifier

discord = DiscordNotifier()

# When alpha decay detected
if sharpe_ratio < threshold:
    discord.send_alert(
        "Alpha decay detected!",
        level="warning",
        details=f"Sharpe: {sharpe_ratio:.4f} < {threshold:.4f}"
    )
```

---

## What Notifications Look Like

### Trade Executed
```
📈 BUY BTC/USD
━━━━━━━━━━━━━━━━
Quantity: 0.010000
Price: $45,000.00
Total: $450.00
━━━━━━━━━━━━━━━━
Portfolio Value: $500.00
```

### System Alert
```
⚠️ WARNING Alert
━━━━━━━━━━━━━━━━
Alpha decay detected! Sharpe ratio below threshold.

Details:
Current: 0.0012, Threshold: 0.0020
```

### Bot Command Response
```
User: !status

Bot:
📊 Cappuccino Trading System Status
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 Paper Trader
✅ Running (PID: 12345)
Memory: 125.3 MB
CPU: 2.5%

🧠 Training
✅ 1 worker(s)
Memory: 458.7 MB

💰 Portfolio
Total: $525.43
Cash: $50.00
Positions: 2
```

---

## Scripts That Need Updates

See **SCRIPTS_TO_ADJUST.md** for complete list.

### High Priority (for Discord)
1. ✅ **discord_bot.py** - Already created
2. ✅ **integrations/discord_notifier.py** - Already created
3. ⏳ **constants.py** - Add Discord config
4. ⏳ **paper_trader_alpaca_polling.py** - Add trade notifications
5. ⏳ **performance_monitor.py** - Add alert notifications

### Medium Priority
6. ⏳ **auto_model_deployer.py** - Deployment notifications
7. ⏳ **1_optimize_unified.py** - Training updates
8. ⏳ **system_watchdog.py** - System alerts

### Low Priority (Nice to Have)
9. ⏳ **dashboard.py** - "Send to Discord" button
10. ⏳ **show_ensemble_votes.py** - Post votes to Discord

---

## Using Aider to Make Changes

Since you have Aider set up, you can use it to make these changes:

```bash
# Update constants.py
aider constants.py --message "Add Discord configuration section with BOT_TOKEN, WEBHOOK_URL, TRADE_CHANNEL_ID, ALERT_CHANNEL_ID, and ENABLED flag" --no-auto-commits

# Update paper trader
aider paper_trader_alpaca_polling.py --message "Import DiscordNotifier and add trade notifications after each executed trade" --no-auto-commits

# Update performance monitor
aider performance_monitor.py --message "Import DiscordNotifier and send alerts when alpha decay is detected or system issues occur" --no-auto-commits

# Review changes
git diff

# Commit if good
git commit -am "Add Discord integration"
```

---

## Running the Bot

### Development (foreground)
```bash
python discord_bot.py
```

### Production (background with tmux)
```bash
tmux new -s discord-bot
python discord_bot.py
# Ctrl+B, then D to detach
```

### Production (systemd service)
```bash
# Create service file (see DISCORD_SETUP.md Part 9)
sudo systemctl enable cappuccino-discord
sudo systemctl start cappuccino-discord
sudo systemctl status cappuccino-discord
```

---

## Testing Checklist

After setup, test:

- [ ] Bot comes online in Discord
- [ ] `!status` shows system status
- [ ] `!portfolio` shows portfolio (if trading)
- [ ] `!trader status` shows trader status
- [ ] `!help` lists all commands
- [ ] Trade notifications appear in #trading channel
- [ ] Alert notifications appear in #alerts channel
- [ ] Can start/stop trader with `!trader start/stop`

---

## Security Checklist

- [ ] Bot token not committed to git
- [ ] `.env` file in `.gitignore`
- [ ] Webhook URL kept private
- [ ] Bot has minimal Discord permissions
- [ ] Only trusted users can use trader controls

---

## Next Steps

1. **Setup Discord Bot** - Follow DISCORD_SETUP.md
2. **Test Bot** - Make sure `!status` works
3. **Update Scripts** - Use Aider or manual edits
4. **Test Notifications** - Verify trades/alerts appear
5. **Run in Production** - Use systemd service

---

## Files Created

```
cappuccino/
├── discord_bot.py                      # Main bot
├── integrations/
│   └── discord_notifier.py            # Notification helper
├── DISCORD_SETUP.md                    # Setup guide
├── SCRIPTS_TO_ADJUST.md               # Scripts to update
└── DISCORD_INTEGRATION_SUMMARY.md     # This file
```

---

## Benefits

After full integration, you'll have:

1. **Real-time Trade Alerts** - Know every trade instantly
2. **System Monitoring** - Check status from Discord
3. **Remote Control** - Start/stop trader from phone
4. **Error Alerts** - Get notified of issues immediately
5. **Training Progress** - See best trials as they complete
6. **Portfolio Tracking** - Check holdings anytime with `!portfolio`

All from Discord, anywhere! 📱🚀
