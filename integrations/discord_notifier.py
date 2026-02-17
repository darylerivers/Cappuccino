"""
Discord Notifier - Send notifications to Discord channels

Usage:
    from integrations.discord_notifier import DiscordNotifier

    notifier = DiscordNotifier()
    notifier.send_trade_notification(symbol='BTC', action='BUY', quantity=0.01, price=45000)
    notifier.send_alert('GPU OOM error in training!')
"""

import requests
import json
import os
from datetime import datetime
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class DiscordNotifier:
    """Send notifications to Discord via webhooks"""

    def __init__(self, webhook_url: Optional[str] = None):
        """
        Initialize Discord notifier

        Args:
            webhook_url: Discord webhook URL (or use DISCORD_WEBHOOK_URL env var)
        """
        self.webhook_url = webhook_url or os.getenv('DISCORD_WEBHOOK_URL')
        self.enabled = bool(self.webhook_url)

        if not self.enabled:
            logger.warning("Discord webhook not configured - notifications disabled")

    def send_message(self, content: str, embed: Optional[Dict] = None) -> bool:
        """
        Send a message to Discord

        Args:
            content: Message text
            embed: Optional embed dict

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        payload = {"content": content}
        if embed:
            payload["embeds"] = [embed]

        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
            return False

    def send_trade_notification(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        portfolio_value: Optional[float] = None
    ) -> bool:
        """
        Send trade notification

        Args:
            symbol: Ticker symbol (e.g., 'BTC/USD')
            action: Trade action ('BUY' or 'SELL')
            quantity: Trade quantity
            price: Execution price
            portfolio_value: Current portfolio value
        """
        color = 0x00ff00 if action == 'BUY' else 0xff0000  # Green for buy, red for sell
        emoji = "📈" if action == 'BUY' else "📉"

        embed = {
            "title": f"{emoji} {action} {symbol}",
            "color": color,
            "fields": [
                {"name": "Quantity", "value": f"{quantity:.6f}", "inline": True},
                {"name": "Price", "value": f"${price:.2f}", "inline": True},
                {"name": "Total", "value": f"${quantity * price:.2f}", "inline": True}
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Cappuccino Trading Bot"}
        }

        if portfolio_value is not None:
            embed["fields"].append({
                "name": "Portfolio Value",
                "value": f"${portfolio_value:.2f}",
                "inline": False
            })

        return self.send_message("", embed=embed)

    def send_alert(
        self,
        message: str,
        level: str = "warning",
        details: Optional[str] = None
    ) -> bool:
        """
        Send alert notification

        Args:
            message: Alert message
            level: Alert level ('info', 'warning', 'error', 'critical')
            details: Optional additional details
        """
        color_map = {
            "info": 0x0099ff,     # Blue
            "warning": 0xffaa00,  # Orange
            "error": 0xff0000,    # Red
            "critical": 0x990000  # Dark red
        }
        emoji_map = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "critical": "🚨"
        }

        color = color_map.get(level, 0x808080)
        emoji = emoji_map.get(level, "⚠️")

        embed = {
            "title": f"{emoji} {level.upper()} Alert",
            "description": message,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Cappuccino Trading Bot"}
        }

        if details:
            embed["fields"] = [{"name": "Details", "value": details, "inline": False}]

        return self.send_message("", embed=embed)

    def send_training_update(
        self,
        trial_number: int,
        sharpe_ratio: float,
        total_return: float,
        status: str = "completed"
    ) -> bool:
        """
        Send training progress notification

        Args:
            trial_number: Trial number
            sharpe_ratio: Trial Sharpe ratio
            total_return: Trial total return
            status: Trial status
        """
        color = 0x00ff00 if sharpe_ratio > 0 else 0xff0000

        embed = {
            "title": f"🧠 Training Update - Trial #{trial_number}",
            "color": color,
            "fields": [
                {"name": "Status", "value": status.title(), "inline": True},
                {"name": "Sharpe Ratio", "value": f"{sharpe_ratio:.4f}", "inline": True},
                {"name": "Total Return", "value": f"{total_return:.2%}", "inline": True}
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Cappuccino Training System"}
        }

        return self.send_message("", embed=embed)

    def send_deployment_notification(
        self,
        trial_number: int,
        sharpe_ratio: float,
        deployment_dir: str
    ) -> bool:
        """
        Send model deployment notification

        Args:
            trial_number: Deployed trial number
            sharpe_ratio: Trial Sharpe ratio
            deployment_dir: Deployment directory
        """
        embed = {
            "title": "🚀 New Model Deployed",
            "color": 0x00ff00,
            "fields": [
                {"name": "Trial", "value": f"#{trial_number}", "inline": True},
                {"name": "Sharpe Ratio", "value": f"{sharpe_ratio:.4f}", "inline": True},
                {"name": "Directory", "value": deployment_dir, "inline": False}
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Cappuccino Auto-Deployer"}
        }

        return self.send_message("", embed=embed)

    def send_portfolio_summary(
        self,
        total_value: float,
        cash: float,
        positions: Dict[str, Dict],
        pnl_24h: Optional[float] = None
    ) -> bool:
        """
        Send portfolio summary

        Args:
            total_value: Total portfolio value
            cash: Cash balance
            positions: Position dictionary {symbol: {quantity, avg_price}}
            pnl_24h: 24-hour P&L (optional)
        """
        invested = total_value - cash
        color = 0x00ff00 if pnl_24h is None or pnl_24h >= 0 else 0xff0000

        fields = [
            {"name": "Total Value", "value": f"${total_value:.2f}", "inline": True},
            {"name": "Cash", "value": f"${cash:.2f}", "inline": True},
            {"name": "Invested", "value": f"${invested:.2f}", "inline": True}
        ]

        if pnl_24h is not None:
            fields.append({
                "name": "24h P&L",
                "value": f"${pnl_24h:+.2f} ({pnl_24h/total_value*100:+.2f}%)",
                "inline": False
            })

        if positions:
            position_text = "\n".join([
                f"**{symbol}**: {data['quantity']:.4f} @ ${data['avg_price']:.2f}"
                for symbol, data in positions.items()
            ])
            fields.append({"name": "Positions", "value": position_text, "inline": False})

        embed = {
            "title": "💼 Portfolio Summary",
            "color": color,
            "fields": fields,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Cappuccino Trading Bot"}
        }

        return self.send_message("", embed=embed)

    def send_system_status(
        self,
        trader_running: bool,
        training_running: bool,
        gpu_usage: Optional[float] = None,
        memory_usage: Optional[float] = None
    ) -> bool:
        """
        Send system status notification

        Args:
            trader_running: Whether paper trader is running
            training_running: Whether training is running
            gpu_usage: GPU usage percentage (optional)
            memory_usage: Memory usage percentage (optional)
        """
        fields = [
            {
                "name": "Paper Trader",
                "value": "✅ Running" if trader_running else "❌ Stopped",
                "inline": True
            },
            {
                "name": "Training",
                "value": "✅ Running" if training_running else "❌ Stopped",
                "inline": True
            }
        ]

        if gpu_usage is not None:
            fields.append({
                "name": "GPU Usage",
                "value": f"{gpu_usage:.1f}%",
                "inline": True
            })

        if memory_usage is not None:
            fields.append({
                "name": "Memory Usage",
                "value": f"{memory_usage:.1f}%",
                "inline": True
            })

        color = 0x00ff00 if trader_running and training_running else 0xffaa00

        embed = {
            "title": "📊 System Status",
            "color": color,
            "fields": fields,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Cappuccino System Monitor"}
        }

        return self.send_message("", embed=embed)
