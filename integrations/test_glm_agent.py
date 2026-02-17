#!/usr/bin/env python3
"""
Test GLM-4 Agent with different personas
Shows how to make GLM-4 behave like specialized assistants
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations.glm_agent import GLMAgent
from integrations.glm_prompts import get_prompt


def test_market_analyst():
    """Test GLM-4 as a market analyst"""
    print("=" * 70)
    print("TEST 1: MARKET ANALYST PERSONA")
    print("=" * 70)

    # Create temporary prompt file
    prompt_file = Path("/tmp/analyst_prompt.txt")
    prompt_file.write_text(get_prompt('analyst'))

    agent = GLMAgent(system_prompt_file=str(prompt_file), temperature=0.2)

    # Test sentiment analysis
    result = agent.analyze_sentiment(
        "Bitcoin ETF approval sees $2B inflows in first week, institutional adoption accelerating"
    )

    print("\nMarket Analyst Response:")
    print(json.dumps(result, indent=2))

    # Follow-up question (multi-turn)
    response = agent.chat("Given this, what's your 7-day price prediction for BTC?")
    print("\nFollow-up Response:")
    print(response)

    return agent


def test_risk_manager():
    """Test GLM-4 as a risk manager"""
    print("\n" + "=" * 70)
    print("TEST 2: RISK MANAGER PERSONA")
    print("=" * 70)

    prompt_file = Path("/tmp/risk_prompt.txt")
    prompt_file.write_text(get_prompt('risk'))

    agent = GLMAgent(system_prompt_file=str(prompt_file), temperature=0.1)

    # Test risk assessment
    portfolio = {
        "total_value": 10000,
        "positions": {
            "BTC": {"value": 6000, "pct": 60},
            "ETH": {"value": 3000, "pct": 30},
            "CASH": {"value": 1000, "pct": 10}
        },
        "current_pnl": "+12%",
        "max_drawdown": "-8%"
    }

    market = {
        "volatility": "high",
        "trend": "upward but weakening",
        "btc_price": 45000,
        "sentiment": "euphoric",
        "fear_greed_index": 78
    }

    result = agent.risk_assessment(portfolio, market)

    print("\nRisk Manager Assessment:")
    print(json.dumps(result, indent=2))

    return agent


def test_signal_validator():
    """Test GLM-4 as a trading signal validator"""
    print("\n" + "=" * 70)
    print("TEST 3: SIGNAL VALIDATOR PERSONA")
    print("=" * 70)

    prompt_file = Path("/tmp/validator_prompt.txt")
    prompt_file.write_text(get_prompt('validator'))

    agent = GLMAgent(system_prompt_file=str(prompt_file), temperature=0.1)

    # Test signal validation
    signal = {
        "action": "buy",
        "ticker": "BTC",
        "position_size": 0.45,  # 45% of portfolio
        "entry_price": 45000,
        "stop_loss": 43500,
        "take_profit": 48000,
        "drl_confidence": 0.87,
        "reason": "Strong momentum, positive sentiment"
    }

    market = {
        "price": 45000,
        "trend": "upward",
        "volatility": "medium",
        "volume": "high",
        "recent_trades": [
            {"type": "buy", "result": "loss", "ticker": "BTC"},
            {"type": "sell", "result": "profit", "ticker": "ETH"}
        ]
    }

    result = agent.evaluate_trade_signal(signal, market)

    print("\nSignal Validator Response:")
    print(json.dumps(result, indent=2))

    return agent


def test_news_analyzer():
    """Test GLM-4 as a news impact analyzer"""
    print("\n" + "=" * 70)
    print("TEST 4: NEWS ANALYZER PERSONA")
    print("=" * 70)

    prompt_file = Path("/tmp/news_prompt.txt")
    prompt_file.write_text(get_prompt('news'))

    agent = GLMAgent(system_prompt_file=str(prompt_file), temperature=0.2)

    # Test news analysis
    headlines = [
        "SEC approves spot Bitcoin ETF applications from major institutions",
        "Ethereum network upgrade reduces gas fees by 40%",
        "Major exchange reports $500M hack, halts withdrawals"
    ]

    result = agent.analyze_news_impact(headlines, ["BTC", "ETH"])

    print("\nNews Analyzer Response:")
    print(json.dumps(result, indent=2))

    return agent


def test_claude_style():
    """Test GLM-4 with Claude Code-style prompts"""
    print("\n" + "=" * 70)
    print("TEST 5: CLAUDE CODE STYLE PERSONA")
    print("=" * 70)

    prompt_file = Path("/tmp/claude_prompt.txt")
    prompt_file.write_text(get_prompt('claude'))

    agent = GLMAgent(system_prompt_file=str(prompt_file), temperature=0.3)

    # Ask for code analysis
    question = """I have a DRL trading agent that's showing 84% win rate in backtests
but only 60% in paper trading. The Sharpe ratio dropped from 11.5 to 4.2.
What could be causing this performance degradation?"""

    response = agent.chat(question)

    print("\nClaude-style Response:")
    print(response)

    # Follow-up
    followup = "What specific metrics should I track to identify the root cause?"
    response2 = agent.chat(followup)

    print("\nFollow-up Response:")
    print(response2)

    return agent


def show_usage_comparison():
    """Show all agent stats"""
    print("\n" + "=" * 70)
    print("USAGE STATISTICS")
    print("=" * 70)

    # You can track and compare usage across different personas
    # This helps optimize which persona to use when
    print("""
Agent Comparison:
- Market Analyst: Best for sentiment analysis, price predictions
- Risk Manager: Best for portfolio review, position sizing
- Signal Validator: Best for pre-trade checks, risk validation
- News Analyzer: Best for real-time news impact assessment
- Claude Style: Best for debugging, code review, explanations

Token Efficiency:
- JSON outputs: More tokens but structured
- Free-form: Fewer tokens but needs parsing
- Temperature: Lower = more consistent, Higher = more creative
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GLM-4 AGENT TESTING SUITE")
    print("Making GLM-4 behave like specialized AI assistants")
    print("=" * 70)

    try:
        # Run all tests
        analyst = test_market_analyst()
        print(f"\n✓ Market Analyst Stats: {analyst.get_stats()}")

        risk_mgr = test_risk_manager()
        print(f"\n✓ Risk Manager Stats: {risk_mgr.get_stats()}")

        validator = test_signal_validator()
        print(f"\n✓ Signal Validator Stats: {validator.get_stats()}")

        news = test_news_analyzer()
        print(f"\n✓ News Analyzer Stats: {news.get_stats()}")

        claude = test_claude_style()
        print(f"\n✓ Claude Style Stats: {claude.get_stats()}")

        show_usage_comparison()

        print("\n" + "=" * 70)
        print("ALL TESTS COMPLETE")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
