#!/usr/bin/env python3
"""
Test Discord Integration

Quick test script to verify Discord notifications are working.

Usage:
    python test_discord.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")

from integrations.discord_notifier import DiscordNotifier
from constants import DISCORD


def test_discord_config():
    """Test Discord configuration."""
    print("\n" + "="*80)
    print("DISCORD CONFIGURATION TEST")
    print("="*80 + "\n")

    print("Environment Variables:")
    print(f"  DISCORD_ENABLED:          {os.getenv('DISCORD_ENABLED', 'NOT SET')}")
    print(f"  DISCORD_WEBHOOK_URL:      {'SET' if os.getenv('DISCORD_WEBHOOK_URL') else 'NOT SET'}")
    print(f"  DISCORD_BOT_TOKEN:        {'SET' if os.getenv('DISCORD_BOT_TOKEN') else 'NOT SET'}")
    print(f"  DISCORD_TRADE_CHANNEL_ID: {os.getenv('DISCORD_TRADE_CHANNEL_ID', 'NOT SET')}")
    print(f"  DISCORD_ALERT_CHANNEL_ID: {os.getenv('DISCORD_ALERT_CHANNEL_ID', 'NOT SET')}")
    print()

    print("Constants Configuration:")
    print(f"  DISCORD.ENABLED:             {DISCORD.ENABLED}")
    print(f"  DISCORD.WEBHOOK_URL:         {'SET' if DISCORD.WEBHOOK_URL else 'NOT SET'}")
    print(f"  DISCORD.NOTIFY_TRADES:       {DISCORD.NOTIFY_TRADES}")
    print(f"  DISCORD.NOTIFY_ALERTS:       {DISCORD.NOTIFY_ALERTS}")
    print(f"  DISCORD.NOTIFY_TRAINING:     {DISCORD.NOTIFY_TRAINING}")
    print(f"  DISCORD.NOTIFY_DEPLOYMENTS:  {DISCORD.NOTIFY_DEPLOYMENTS}")
    print()

    if not DISCORD.ENABLED:
        print("❌ DISCORD_ENABLED is false - notifications will not be sent")
        print("\n💡 To enable Discord notifications:")
        print("   1. Create a Discord webhook")
        print("   2. Add to .env file:")
        print("      DISCORD_WEBHOOK_URL=your_webhook_url_here")
        print("      DISCORD_ENABLED=true")
        print("\n   See DISCORD_SETUP.md for detailed instructions")
        return False

    if not DISCORD.WEBHOOK_URL:
        print("❌ DISCORD_WEBHOOK_URL is not set")
        print("\n💡 To set webhook URL:")
        print("   1. Create a webhook in Discord (Channel Settings → Integrations → Webhooks)")
        print("   2. Add to .env file:")
        print("      DISCORD_WEBHOOK_URL=your_webhook_url_here")
        return False

    print("✅ Discord configuration looks good!")
    return True


def test_basic_notification():
    """Test basic Discord notification."""
    print("\n" + "="*80)
    print("BASIC NOTIFICATION TEST")
    print("="*80 + "\n")

    notifier = DiscordNotifier()

    if not notifier.enabled:
        print("❌ Discord notifier is not enabled")
        return False

    print("Sending test message...")
    success = notifier.send_message("🧪 Test message from Cappuccino trading system!")

    if success:
        print("✅ Basic notification sent successfully!")
        print("   Check your Discord channel for the message")
        return True
    else:
        print("❌ Failed to send notification")
        print("   Check webhook URL and network connectivity")
        return False


def test_trade_notification():
    """Test trade notification."""
    print("\n" + "="*80)
    print("TRADE NOTIFICATION TEST")
    print("="*80 + "\n")

    notifier = DiscordNotifier()

    if not notifier.enabled:
        print("❌ Discord notifier is not enabled")
        return False

    print("Sending test trade notification (BUY)...")
    success = notifier.send_trade_notification(
        symbol="BTC/USD",
        action="BUY",
        quantity=0.0015,
        price=69244.22,
        portfolio_value=499.98
    )

    if success:
        print("✅ Trade notification sent successfully!")
        print("   Check Discord for green BUY notification with 📈 emoji")
        return True
    else:
        print("❌ Failed to send trade notification")
        return False


def test_alert_notification():
    """Test alert notifications."""
    print("\n" + "="*80)
    print("ALERT NOTIFICATION TEST")
    print("="*80 + "\n")

    notifier = DiscordNotifier()

    if not notifier.enabled:
        print("❌ Discord notifier is not enabled")
        return False

    # Test different alert levels
    alert_levels = [
        ("info", "This is an info alert"),
        ("warning", "This is a warning alert"),
        ("error", "This is an error alert"),
    ]

    for level, message in alert_levels:
        print(f"Sending {level.upper()} alert...")
        success = notifier.send_alert(
            message=message,
            level=level,
            details="Test alert from test_discord.py"
        )

        if success:
            print(f"  ✅ {level.upper()} alert sent")
        else:
            print(f"  ❌ {level.upper()} alert failed")

    print("\n✅ Alert notifications sent!")
    print("   Check Discord for color-coded alerts (blue, orange, red)")
    return True


def test_portfolio_summary():
    """Test portfolio summary notification."""
    print("\n" + "="*80)
    print("PORTFOLIO SUMMARY TEST")
    print("="*80 + "\n")

    notifier = DiscordNotifier()

    if not notifier.enabled:
        print("❌ Discord notifier is not enabled")
        return False

    print("Sending test portfolio summary...")
    success = notifier.send_portfolio_summary(
        total_value=499.98,
        cash=497.79,
        positions={
            "AVAX/USD": {"quantity": 0.2384, "avg_price": 9.15},
            "BTC/USD": {"quantity": 0.0010, "avg_price": 69000.00}
        },
        pnl_24h=-0.02
    )

    if success:
        print("✅ Portfolio summary sent successfully!")
        print("   Check Discord for portfolio summary with positions")
        return True
    else:
        print("❌ Failed to send portfolio summary")
        return False


def main():
    """Run all Discord tests."""
    print("\n🧪 Discord Integration Test Suite")
    print("This will send several test notifications to your Discord channel\n")

    # Test configuration
    if not test_discord_config():
        print("\n❌ Configuration test failed")
        print("\n📖 See DISCORD_SETUP.md for setup instructions")
        return 1

    # Run tests
    tests = [
        ("Basic Notification", test_basic_notification),
        ("Trade Notification", test_trade_notification),
        ("Alert Notifications", test_alert_notification),
        ("Portfolio Summary", test_portfolio_summary),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80 + "\n")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}  {name}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All tests passed! Discord integration is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check configuration and webhook URL.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
