#!/usr/bin/env python3
"""
Comprehensive Market Analysis Tool

Provides detailed historical performance analysis across multiple asset classes.
Uses Tiburtina's data sources for multi-asset comparison.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from tiburtina_integration import get_tiburtina_client


def print_section(title: str):
    """Print section header."""
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def analyze_crypto_performance(client):
    """Analyze crypto market performance."""
    print_section("CRYPTO MARKET ANALYSIS")

    cryptos = client.get_crypto_overview(use_cache=True)

    if not cryptos or "error" in cryptos[0]:
        print("Error fetching crypto data")
        return

    print()
    print(f"{'Rank':<6} {'Symbol':<10} {'Price':<20} {'24h Change':<15} {'7d Change':<15} {'Market Cap'}")
    print("-" * 80)

    for i, crypto in enumerate(cryptos[:10], 1):
        symbol = crypto.get("symbol", "N/A")
        price = crypto.get("price", 0)
        change_24h = crypto.get("change_24h", 0)
        change_7d = crypto.get("change_7d", 0)
        mcap = crypto.get("market_cap", 0)

        price_str = f"${price:,.2f}" if price and price > 1 else f"${price:.6f}" if price else "N/A"
        change_24h_str = f"{change_24h:+.2f}%" if change_24h else "N/A"
        change_7d_str = f"{change_7d:+.2f}%" if change_7d else "N/A"
        mcap_str = f"${mcap/1e9:.1f}B" if mcap else "N/A"

        print(f"{i:<6} {symbol:<10} {price_str:<20} {change_24h_str:<15} {change_7d_str:<15} {mcap_str}")

    # Calculate average performance
    changes_24h = [c.get("change_24h", 0) for c in cryptos[:10] if c.get("change_24h")]
    changes_7d = [c.get("change_7d", 0) for c in cryptos[:10] if c.get("change_7d")]

    if changes_24h:
        avg_24h = sum(changes_24h) / len(changes_24h)
        print()
        print(f"Top 10 Average Performance:")
        print(f"  24h: {avg_24h:+.2f}%")
        if changes_7d:
            avg_7d = sum(changes_7d) / len(changes_7d)
            print(f"  7d:  {avg_7d:+.2f}%")


def analyze_macro_context(client):
    """Analyze macro economic context."""
    print_section("MACRO ECONOMIC CONTEXT")

    macro = client.get_macro_snapshot(use_cache=True)

    if "error" in macro:
        print(f"Error: {macro['error']}")
        print()
        print("Run './prefetch_tiburtina.sh' to populate macro data cache")
        return

    if not macro:
        print("No macro data available")
        return

    print()
    print(f"{'Indicator':<30} {'Current Value':<15} {'Date'}")
    print("-" * 80)

    indicators = {
        "fed_funds": "Federal Funds Rate",
        "treasury_10y": "10-Year Treasury Yield",
        "unemployment": "Unemployment Rate",
        "cpi": "CPI (Inflation)",
        "vix": "VIX (Volatility Index)"
    }

    for key, name in indicators.items():
        if key in macro and isinstance(macro[key], dict):
            value = macro[key].get("value", "N/A")
            date = macro[key].get("date", "")[:10]
            value_str = f"{value:.2f}%" if isinstance(value, (int, float)) else str(value)
            print(f"{name:<30} {value_str:<15} {date}")


def analyze_asset_performance(client):
    """Analyze multi-asset performance."""
    print_section("ASSET CLASS PERFORMANCE")

    performance = client.get_asset_performance()

    if "error" in performance:
        print(f"Error: {performance['error']}")
        return

    print()
    print(f"{'Asset Class':<30} {'Performance':<15} {'Period'}")
    print("-" * 80)

    if "crypto_24h" in performance:
        change = performance["crypto_24h"]
        print(f"{'Crypto (Top 5 avg)':<30} {change:+.2f}%{' ':<10} 24 hours")

    if "stocks_daily" in performance:
        change = performance["stocks_daily"]
        print(f"{'Stocks (S&P 500)':<30} {change:+.2f}%{' ':<10} Today")

    if not ("crypto_24h" in performance or "stocks_daily" in performance):
        print("Limited performance data available")

    # Macro context
    if "macro" in performance and isinstance(performance["macro"], dict):
        print()
        print("Macro Context:")
        macro = performance["macro"]
        if "fed_funds" in macro:
            ff = macro["fed_funds"].get("value", 0)
            print(f"  Fed Funds Rate: {ff:.2f}%")
        if "treasury_10y" in macro:
            ty = macro["treasury_10y"].get("value", 0)
            print(f"  10Y Treasury:   {ty:.2f}%")


def get_ai_analysis(client):
    """Get AI-powered market analysis."""
    print_section("AI-POWERED MARKET ANALYSIS")

    print()
    print("Generating comprehensive AI analysis...")
    print("(This may take 30-60 seconds as it uses local LLM)")
    print()

    try:
        analysis = client.get_market_analysis_detailed()
        print(analysis)
    except Exception as e:
        print(f"Error generating AI analysis: {e}")
        print()
        print("For AI analysis, use Tiburtina terminal directly:")
        print("  $ cd /home/mrc/experiment/tiburtina")
        print("  $ python terminal/cli.py")
        print("  $ /brief")


def show_news_summary(client):
    """Show latest news headlines."""
    print_section("LATEST FINANCIAL NEWS")

    news = client.get_news_summary(use_cache=True)

    if not news or "error" in news[0]:
        print("No news available")
        return

    print()
    for i, article in enumerate(news[:10], 1):
        if isinstance(article, dict):
            title = article.get("title", "No title")
            source = article.get("source", "")
            published = article.get("published_at", "")[:10]

            print(f"{i}. {title}")
            print(f"   Source: {source} | {published}")
            print()


def main():
    """Run comprehensive market analysis."""
    client = get_tiburtina_client()

    if not client.is_available():
        print("Tiburtina not available:", client.get_error())
        print()
        print("To enable:")
        print("  1. Install Tiburtina dependencies")
        print("  2. Configure API keys in /home/mrc/experiment/tiburtina/.env")
        print("  3. Re-run this script")
        return

    print()
    print("=" * 80)
    print("  CAPPUCCINO MARKET ANALYSIS")
    print("  Powered by Tiburtina AI")
    print("=" * 80)
    print()
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run analyses
    analyze_asset_performance(client)
    analyze_crypto_performance(client)
    analyze_macro_context(client)
    show_news_summary(client)

    # Ask if user wants AI analysis (slow)
    print()
    print("=" * 80)
    response = input("Generate AI market analysis? (takes 30-60s) [y/N]: ").strip().lower()
    if response == 'y':
        get_ai_analysis(client)

    print()
    print("=" * 80)
    print("  For more detailed analysis, use Tiburtina terminal:")
    print("  $ cd /home/mrc/experiment/tiburtina")
    print("  $ python terminal/cli.py")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
