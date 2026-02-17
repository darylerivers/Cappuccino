#!/usr/bin/env python3
"""
Discord Bot for Cappuccino Crypto Trading System

Controls and monitors the trading system via Discord.

Commands:
  !status           - Show system status
  !trader status    - Show paper trader status
  !trader start     - Start paper trading
  !trader stop      - Stop paper trading
  !training status  - Show training progress
  !portfolio        - Show current portfolio
  !trades           - Show recent trades
  !ensemble         - Show ensemble votes
  !dashboard        - Get dashboard link
  !help             - Show all commands

Setup:
  1. Create Discord bot at https://discord.com/developers/applications
  2. Enable "Message Content Intent" in bot settings
  3. Get bot token and channel IDs
  4. Set environment variables:
     - DISCORD_BOT_TOKEN
     - DISCORD_TRADE_CHANNEL_ID
     - DISCORD_ALERT_CHANNEL_ID
  5. Run: python discord_bot.py
"""

import discord
from discord.ext import commands, tasks
import asyncio
import os
import sys
import json
import subprocess
import psutil
from datetime import datetime
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from constants import DISCORD, TRADING

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('discord_bot')

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Channels
TRADE_CHANNEL_ID = DISCORD.TRADE_CHANNEL_ID
ALERT_CHANNEL_ID = DISCORD.ALERT_CHANNEL_ID


class TradingSystemMonitor:
    """Monitor trading system processes and files"""

    def __init__(self):
        self.base_path = Path(__file__).parent

    def get_paper_trader_status(self):
        """Check if paper trader is running"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and 'paper_trader_alpaca_polling.py' in ' '.join(cmdline):
                    return {
                        'running': True,
                        'pid': proc.info['pid'],
                        'memory_mb': proc.memory_info().rss / 1024 / 1024,
                        'cpu_percent': proc.cpu_percent()
                    }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return {'running': False}

    def get_training_status(self):
        """Check training processes"""
        training_procs = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and '1_optimize_unified.py' in ' '.join(cmdline):
                    training_procs.append({
                        'pid': proc.info['pid'],
                        'memory_mb': proc.memory_info().rss / 1024 / 1024,
                        'cpu_percent': proc.cpu_percent()
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return training_procs

    def get_portfolio_status(self):
        """Read portfolio state from positions file"""
        positions_file = self.base_path / 'paper_trades' / 'positions_state.json'
        if positions_file.exists():
            try:
                with open(positions_file, 'r') as f:
                    data = json.load(f)
                return data
            except Exception as e:
                logger.error(f"Error reading portfolio: {e}")
                return None
        return None

    def get_recent_trades(self, limit=5):
        """Get recent trades from log files"""
        trades = []
        trade_logs = sorted(
            (self.base_path / 'paper_trades').glob('trades_*.json'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        for log_file in trade_logs[:limit]:
            try:
                with open(log_file, 'r') as f:
                    trade_data = json.load(f)
                    trades.append(trade_data)
            except Exception as e:
                logger.error(f"Error reading trade log: {e}")
                continue

        return trades

    def start_paper_trader(self):
        """Start paper trading process"""
        cmd = [
            sys.executable,
            str(self.base_path / 'paper_trader_alpaca_polling.py'),
            '--deployment-dir', 'deployments/trial_250_live'
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.base_path
            )
            return True, f"Started paper trader (PID: {proc.pid})"
        except Exception as e:
            return False, f"Failed to start: {str(e)}"

    def stop_paper_trader(self):
        """Stop paper trading process"""
        status = self.get_paper_trader_status()
        if status['running']:
            try:
                proc = psutil.Process(status['pid'])
                proc.terminate()
                proc.wait(timeout=10)
                return True, f"Stopped paper trader (PID: {status['pid']})"
            except Exception as e:
                return False, f"Failed to stop: {str(e)}"
        return False, "Paper trader is not running"


# Initialize monitor
monitor = TradingSystemMonitor()


@bot.event
async def on_ready():
    logger.info(f'{bot.user} has connected to Discord!')
    logger.info(f'Trade channel: {TRADE_CHANNEL_ID}')
    logger.info(f'Alert channel: {ALERT_CHANNEL_ID}')

    # Start monitoring tasks
    monitor_system.start()

    # Send startup message
    if ALERT_CHANNEL_ID:
        channel = bot.get_channel(ALERT_CHANNEL_ID)
        if channel:
            await channel.send("🤖 **Cappuccino Trading Bot Online**\nType `!help` for commands")


@bot.command(name='status', help='Show system status')
async def status(ctx):
    """Show overall system status"""
    trader_status = monitor.get_paper_trader_status()
    training_procs = monitor.get_training_status()
    portfolio = monitor.get_portfolio_status()

    embed = discord.Embed(
        title="📊 Cappuccino Trading System Status",
        color=discord.Color.green() if trader_status['running'] else discord.Color.red(),
        timestamp=datetime.now()
    )

    # Paper Trader Status
    if trader_status['running']:
        embed.add_field(
            name="📈 Paper Trader",
            value=f"✅ Running (PID: {trader_status['pid']})\n"
                  f"Memory: {trader_status['memory_mb']:.1f} MB\n"
                  f"CPU: {trader_status['cpu_percent']:.1f}%",
            inline=True
        )
    else:
        embed.add_field(
            name="📈 Paper Trader",
            value="❌ Not Running",
            inline=True
        )

    # Training Status
    if training_procs:
        total_mem = sum(p['memory_mb'] for p in training_procs)
        embed.add_field(
            name="🧠 Training",
            value=f"✅ {len(training_procs)} worker(s)\n"
                  f"Memory: {total_mem:.1f} MB",
            inline=True
        )
    else:
        embed.add_field(
            name="🧠 Training",
            value="❌ Not Running",
            inline=True
        )

    # Portfolio Value
    if portfolio:
        value = portfolio.get('total_assets', 0)
        cash = portfolio.get('cash', 0)
        positions = portfolio.get('positions', {})
        embed.add_field(
            name="💰 Portfolio",
            value=f"Total: ${value:.2f}\n"
                  f"Cash: ${cash:.2f}\n"
                  f"Positions: {len(positions)}",
            inline=True
        )

    await ctx.send(embed=embed)


@bot.command(name='portfolio', help='Show portfolio details')
async def portfolio_cmd(ctx):
    """Show detailed portfolio information"""
    portfolio = monitor.get_portfolio_status()

    if not portfolio:
        await ctx.send("❌ No portfolio data found")
        return

    embed = discord.Embed(
        title="💼 Portfolio Details",
        color=discord.Color.blue(),
        timestamp=datetime.now()
    )

    total_value = portfolio.get('total_assets', 0)
    cash = portfolio.get('cash', 0)
    positions = portfolio.get('positions', {})

    embed.add_field(name="Total Value", value=f"${total_value:.2f}", inline=True)
    embed.add_field(name="Cash", value=f"${cash:.2f}", inline=True)
    embed.add_field(name="Invested", value=f"${total_value - cash:.2f}", inline=True)

    if positions:
        position_text = "\n".join([
            f"**{symbol}**: {data['quantity']:.4f} @ ${data['avg_price']:.2f}"
            for symbol, data in positions.items()
        ])
        embed.add_field(name="Positions", value=position_text, inline=False)

    await ctx.send(embed=embed)


@bot.command(name='trades', help='Show recent trades')
async def trades(ctx, limit: int = 5):
    """Show recent trades"""
    recent_trades = monitor.get_recent_trades(limit)

    if not recent_trades:
        await ctx.send("❌ No recent trades found")
        return

    embed = discord.Embed(
        title=f"📜 Recent Trades (Last {len(recent_trades)})",
        color=discord.Color.gold(),
        timestamp=datetime.now()
    )

    for trade in recent_trades:
        action = trade.get('action', 'UNKNOWN')
        symbol = trade.get('symbol', 'UNKNOWN')
        quantity = trade.get('quantity', 0)
        price = trade.get('price', 0)
        timestamp = trade.get('timestamp', 'Unknown')

        embed.add_field(
            name=f"{action} {symbol}",
            value=f"Qty: {quantity:.4f}\n"
                  f"Price: ${price:.2f}\n"
                  f"Time: {timestamp}",
            inline=True
        )

    await ctx.send(embed=embed)


@bot.group(name='trader', help='Paper trader controls')
async def trader(ctx):
    """Paper trader command group"""
    if ctx.invoked_subcommand is None:
        await ctx.send("Use: `!trader status`, `!trader start`, or `!trader stop`")


@trader.command(name='status', help='Show trader status')
async def trader_status(ctx):
    """Show paper trader status"""
    status = monitor.get_paper_trader_status()

    if status['running']:
        embed = discord.Embed(
            title="📈 Paper Trader Status",
            color=discord.Color.green()
        )
        embed.add_field(name="Status", value="✅ Running", inline=True)
        embed.add_field(name="PID", value=status['pid'], inline=True)
        embed.add_field(name="Memory", value=f"{status['memory_mb']:.1f} MB", inline=True)
        embed.add_field(name="CPU", value=f"{status['cpu_percent']:.1f}%", inline=True)
    else:
        embed = discord.Embed(
            title="📈 Paper Trader Status",
            color=discord.Color.red()
        )
        embed.add_field(name="Status", value="❌ Not Running", inline=False)

    await ctx.send(embed=embed)


@trader.command(name='start', help='Start paper trading')
async def trader_start(ctx):
    """Start paper trader"""
    success, message = monitor.start_paper_trader()

    if success:
        await ctx.send(f"✅ {message}")
    else:
        await ctx.send(f"❌ {message}")


@trader.command(name='stop', help='Stop paper trading')
async def trader_stop(ctx):
    """Stop paper trader"""
    success, message = monitor.stop_paper_trader()

    if success:
        await ctx.send(f"✅ {message}")
    else:
        await ctx.send(f"❌ {message}")


@bot.group(name='training', help='Training controls')
async def training(ctx):
    """Training command group"""
    if ctx.invoked_subcommand is None:
        await ctx.send("Use: `!training status`")


@training.command(name='status', help='Show training status')
async def training_status(ctx):
    """Show training progress"""
    procs = monitor.get_training_status()

    if not procs:
        await ctx.send("❌ No training processes running")
        return

    embed = discord.Embed(
        title="🧠 Training Status",
        color=discord.Color.purple()
    )

    total_mem = sum(p['memory_mb'] for p in procs)
    embed.add_field(name="Workers", value=len(procs), inline=True)
    embed.add_field(name="Total Memory", value=f"{total_mem:.1f} MB", inline=True)

    await ctx.send(embed=embed)


@bot.command(name='dashboard', help='Get dashboard link')
async def dashboard_cmd(ctx):
    """Send dashboard link"""
    await ctx.send("📊 Dashboard: http://localhost:8050\n"
                  "Run locally: `python dashboard.py`")


@tasks.loop(minutes=5)
async def monitor_system():
    """Periodic system monitoring"""
    # Check for alerts
    trader_status = monitor.get_paper_trader_status()

    # Alert if paper trader stopped unexpectedly
    if not trader_status['running'] and ALERT_CHANNEL_ID:
        channel = bot.get_channel(ALERT_CHANNEL_ID)
        if channel:
            # Could add logic here to track if it was running before
            pass


@bot.event
async def on_command_error(ctx, error):
    """Handle command errors"""
    if isinstance(error, commands.CommandNotFound):
        await ctx.send("❌ Unknown command. Type `!help` for available commands")
    else:
        logger.error(f"Command error: {error}")
        await ctx.send(f"❌ Error: {str(error)}")


def main():
    """Run the Discord bot"""
    # Check for bot token
    bot_token = os.getenv('DISCORD_BOT_TOKEN')
    if not bot_token:
        logger.error("DISCORD_BOT_TOKEN environment variable not set!")
        logger.error("Set it with: export DISCORD_BOT_TOKEN='your-token-here'")
        sys.exit(1)

    # Check for channel IDs
    if not TRADE_CHANNEL_ID and not ALERT_CHANNEL_ID:
        logger.warning("No channel IDs configured in constants.py")
        logger.warning("Bot will run but notifications won't be sent")

    # Run bot
    logger.info("Starting Discord bot...")
    try:
        bot.run(bot_token)
    except discord.errors.LoginFailure:
        logger.error("Invalid bot token!")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
