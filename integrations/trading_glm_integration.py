#!/usr/bin/env python3
"""
Practical GLM-4 integration for your trading system
Shows how to add AI validation to trading decisions
"""

import requests
from typing import Dict, Any, Optional


class TradingGLM:
    """
    Fast GLM-4 integration for trading decisions
    Optimized for real-time use (3-5 second responses)
    """

    def __init__(self, url: str = "http://localhost:11434"):
        self.url = url
        self.model = "glm4"

    def _ask(self, prompt: str, max_tokens: int = 100, temperature: float = 0.2) -> str:
        """Internal method for quick queries"""
        try:
            response = requests.post(
                f"{self.url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                },
                timeout=15
            )
            return response.json().get('response', '').strip()
        except Exception as e:
            return f"Error: {e}"

    def validate_signal(
        self,
        action: str,
        ticker: str,
        price: float,
        position_size: float,
        reason: str
    ) -> Dict[str, Any]:
        """
        Quick validation of DRL trading signal

        Args:
            action: 'buy' or 'sell'
            ticker: e.g., 'BTC', 'ETH'
            price: Current price
            position_size: 0.0-1.0 (fraction of portfolio)
            reason: Why the DRL agent wants to trade

        Returns:
            {"approved": bool, "advice": str, "confidence": str}
        """
        prompt = f"""You are a conservative crypto risk manager.

Signal: {action.upper()} {ticker} at ${price:,.0f}
Position Size: {position_size*100:.0f}% of portfolio
Reason: {reason}

Red flags: >40% position, extreme prices, weak reasoning
Answer format: "YES/NO - reason (1 sentence)"
"""

        response = self._ask(prompt, max_tokens=50)

        # Parse response
        approved = response.lower().startswith('yes')
        confidence = "high" if "definitely" in response.lower() or "clearly" in response.lower() else "medium"

        return {
            "approved": approved,
            "advice": response,
            "confidence": confidence,
            "raw_response": response
        }

    def sentiment_check(self, ticker: str, context: str = "") -> Dict[str, Any]:
        """
        Quick market sentiment check

        Args:
            ticker: e.g., 'BTC'
            context: Optional market context

        Returns:
            {"sentiment": str, "confidence": str, "reasoning": str}
        """
        prompt = f"""Quick crypto sentiment for {ticker}.
Context: {context if context else 'general market'}

Answer: BULLISH/BEARISH/NEUTRAL - why? (1 sentence)
"""

        response = self._ask(prompt, max_tokens=40)

        # Parse
        if "bullish" in response.lower():
            sentiment = "bullish"
        elif "bearish" in response.lower():
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        return {
            "sentiment": sentiment,
            "reasoning": response,
            "confidence": "medium"
        }

    def risk_check(self, portfolio: Dict[str, float], market_conditions: str) -> Dict[str, Any]:
        """
        Quick portfolio risk assessment

        Args:
            portfolio: {"BTC": 0.6, "ETH": 0.3, "CASH": 0.1}
            market_conditions: Brief description

        Returns:
            {"risk_level": str, "suggestion": str}
        """
        positions = ", ".join([f"{k}: {v*100:.0f}%" for k, v in portfolio.items()])

        prompt = f"""Risk assessment for crypto portfolio.
Portfolio: {positions}
Market: {market_conditions}

Risk level (LOW/MEDIUM/HIGH) and 1 suggestion:
"""

        response = self._ask(prompt, max_tokens=60)

        # Parse risk level
        if "high" in response.lower():
            risk_level = "high"
        elif "low" in response.lower():
            risk_level = "low"
        else:
            risk_level = "medium"

        return {
            "risk_level": risk_level,
            "suggestion": response,
            "raw_response": response
        }

    def news_impact(self, headline: str, tickers: list) -> Dict[str, str]:
        """
        Quick news impact assessment

        Args:
            headline: News headline
            tickers: List of tickers to assess

        Returns:
            {"impact": str, "direction": str, "affected": [tickers]}
        """
        prompt = f"""News: "{headline}"
Impact on {', '.join(tickers)}?

Answer: HIGH/MEDIUM/LOW impact, BULLISH/BEARISH, affects [tickers]
"""

        response = self._ask(prompt, max_tokens=50)

        # Parse impact
        if "high" in response.lower():
            impact = "high"
        elif "low" in response.lower():
            impact = "low"
        else:
            impact = "medium"

        direction = "bullish" if "bullish" in response.lower() else "bearish" if "bearish" in response.lower() else "neutral"

        return {
            "impact": impact,
            "direction": direction,
            "reasoning": response,
            "headline": headline
        }


# ============================================================================
# INTEGRATION EXAMPLES
# ============================================================================

def example_1_signal_validation():
    """Example: Validate DRL agent signal before executing"""
    print("=== Example 1: Signal Validation ===\n")

    glm = TradingGLM()

    # DRL agent wants to buy BTC
    signal = {
        "action": "buy",
        "ticker": "BTC",
        "price": 45000,
        "position_size": 0.35,  # 35% of portfolio
        "reason": "Strong momentum and positive sentiment"
    }

    # Ask GLM-4 to validate
    validation = glm.validate_signal(**signal)

    print(f"DRL Signal: {signal['action'].upper()} {signal['ticker']} at ${signal['price']:,}")
    print(f"Position Size: {signal['position_size']*100:.0f}%")
    print(f"DRL Reason: {signal['reason']}\n")
    print(f"GLM-4 Validation: {validation['advice']}")
    print(f"Approved: {validation['approved']}")
    print(f"Confidence: {validation['confidence']}\n")

    if validation['approved']:
        print("✓ Executing trade")
    else:
        print("✗ Trade rejected by GLM-4")


def example_2_morning_check():
    """Example: Morning market sentiment check before trading"""
    print("\n=== Example 2: Morning Market Check ===\n")

    glm = TradingGLM()

    # Check sentiment for each ticker
    tickers = ['BTC', 'ETH']

    print("Morning sentiment check:\n")
    for ticker in tickers:
        sentiment = glm.sentiment_check(
            ticker,
            context="Market opened higher, moderate volume"
        )
        print(f"{ticker}: {sentiment['sentiment'].upper()} - {sentiment['reasoning']}")


def example_3_risk_review():
    """Example: End-of-day risk review"""
    print("\n=== Example 3: Daily Risk Review ===\n")

    glm = TradingGLM()

    # Current portfolio
    portfolio = {
        "BTC": 0.65,  # 65% in BTC
        "ETH": 0.25,  # 25% in ETH
        "CASH": 0.10  # 10% cash
    }

    # Check risk
    risk = glm.risk_check(
        portfolio,
        market_conditions="Volatile, uptrend weakening"
    )

    print(f"Portfolio Allocation:")
    for asset, pct in portfolio.items():
        print(f"  {asset}: {pct*100:.0f}%")

    print(f"\nRisk Assessment: {risk['risk_level'].upper()}")
    print(f"Suggestion: {risk['suggestion']}")


def example_4_news_filter():
    """Example: Filter trades based on breaking news"""
    print("\n=== Example 4: News Impact Filter ===\n")

    glm = TradingGLM()

    # Breaking news
    headline = "SEC announces investigation into major crypto exchange"

    impact = glm.news_impact(headline, ['BTC', 'ETH'])

    print(f"Breaking News: {headline}\n")
    print(f"Impact Level: {impact['impact'].upper()}")
    print(f"Direction: {impact['direction'].upper()}")
    print(f"Analysis: {impact['reasoning']}\n")

    if impact['impact'] == 'high' and impact['direction'] == 'bearish':
        print("⚠️ Pausing trading due to major negative news")
    else:
        print("✓ News impact acceptable, continuing trading")


if __name__ == "__main__":
    print("=" * 70)
    print("GLM-4 TRADING INTEGRATION EXAMPLES")
    print("Fast AI validation for trading decisions (3-5s per call)")
    print("=" * 70)
    print()

    try:
        example_1_signal_validation()
        example_2_morning_check()
        example_3_risk_review()
        example_4_news_filter()

        print("\n" + "=" * 70)
        print("All examples complete!")
        print("=" * 70)
        print("\nIntegration tips:")
        print("1. Use validate_signal() before executing DRL trades")
        print("2. Run sentiment_check() each morning before trading")
        print("3. Run risk_check() daily or when volatility increases")
        print("4. Use news_impact() when breaking news occurs")
        print("\nEach call takes 3-5 seconds - fast enough for real-time use!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure Ollama is running: ollama serve")
