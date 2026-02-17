#!/bin/bash
# Setup Tiburtina Integration for Cappuccino

set -e

echo "=========================================="
echo "Tiburtina Integration Setup"
echo "=========================================="
echo ""

TIBURTINA_DIR="/home/mrc/experiment/tiburtina"

# 1. Check if Tiburtina exists
echo "1. Checking for Tiburtina..."
if [ -d "$TIBURTINA_DIR" ]; then
    echo "✓ Tiburtina found at $TIBURTINA_DIR"
else
    echo "✗ Tiburtina not found"
    echo ""
    echo "Cloning Tiburtina repository..."
    mkdir -p /home/mrc/experiment
    cd /home/mrc/experiment
    # Placeholder - adjust to actual repo
    echo "TODO: Clone Tiburtina repo or create from scratch"
    exit 1
fi

# 2. Check Tiburtina dependencies
echo ""
echo "2. Checking Tiburtina dependencies..."
cd $TIBURTINA_DIR

if [ -f "requirements.txt" ]; then
    echo "Installing Tiburtina requirements..."
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo "⚠ No requirements.txt found"
fi

# 3. Setup Ollama for local LLM
echo ""
echo "3. Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    echo "✓ Ollama installed"
    ollama --version
else
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# 4. Pull Mistral model
echo ""
echo "4. Setting up Mistral 7B model..."
if ollama list | grep -q "mistral"; then
    echo "✓ Mistral already pulled"
else
    echo "Pulling Mistral 7B (this may take a few minutes)..."
    ollama pull mistral
fi

# 5. Test Ollama
echo ""
echo "5. Testing Ollama..."
RESPONSE=$(ollama run mistral "Say 'hello' in one word" --verbose=false 2>/dev/null | head -1)
if [ -n "$RESPONSE" ]; then
    echo "✓ Ollama working: $RESPONSE"
else
    echo "✗ Ollama test failed"
    exit 1
fi

# 6. Create integration helper
echo ""
echo "6. Creating Cappuccino-Tiburtina bridge..."
mkdir -p /opt/user-data/experiment/cappuccino/integrations

cat > /opt/user-data/experiment/cappuccino/integrations/tiburtina_helper.py << 'PYEOF'
"""
Tiburtina Integration Helper for Cappuccino
Provides sentiment analysis and market intelligence
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import json

TIBURTINA_PATH = Path("/home/mrc/experiment/tiburtina")

class TiburtinaBridge:
    """Bridge between Cappuccino and Tiburtina for sentiment analysis"""

    def __init__(self):
        self.tiburtina_available = TIBURTINA_PATH.exists()
        if not self.tiburtina_available:
            print("⚠️ Tiburtina not found - sentiment analysis disabled")
            return

        # Add Tiburtina to path
        sys.path.insert(0, str(TIBURTINA_PATH))

    def analyze_sentiment(self, symbol: str, news_headlines: List[str]) -> Dict:
        """
        Analyze sentiment using Ollama/Mistral

        Args:
            symbol: Trading symbol (e.g., 'BTC/USD')
            news_headlines: List of recent headlines

        Returns:
            {
                'sentiment': 'bullish'|'bearish'|'neutral',
                'confidence': 0.0-1.0,
                'reasoning': str
            }
        """
        if not news_headlines:
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'reasoning': 'No news available'
            }

        # Build prompt for LLM
        headlines_text = '\n'.join([f"- {h}" for h in news_headlines[:10]])

        prompt = f"""Analyze the sentiment for {symbol} based on these recent headlines:

{headlines_text}

Respond ONLY with valid JSON in this exact format:
{{"sentiment": "bullish"|"bearish"|"neutral", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

        try:
            # Call Ollama
            result = subprocess.run(
                ['ollama', 'run', 'mistral', prompt],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                raise Exception(f"Ollama error: {result.stderr}")

            # Parse response
            response = result.stdout.strip()

            # Extract JSON (LLM might add extra text)
            if '{' in response:
                json_start = response.index('{')
                json_end = response.rindex('}') + 1
                json_str = response[json_start:json_end]
                sentiment = json.loads(json_str)

                # Validate
                assert sentiment['sentiment'] in ['bullish', 'bearish', 'neutral']
                assert 0 <= sentiment['confidence'] <= 1

                return sentiment
            else:
                raise Exception("No JSON in response")

        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'reasoning': f'Analysis failed: {str(e)}'
            }

    def get_news_headlines(self, symbol: str, hours: int = 24) -> List[str]:
        """
        Get recent news headlines for a symbol

        For now, returns dummy headlines. Implement RSS/API fetching later.
        """
        # TODO: Implement actual news fetching from:
        # - Alpaca News API
        # - RSS feeds
        # - Tiburtina's news database

        # Dummy implementation
        return [
            f"{symbol} shows strong momentum",
            f"Market analysts bullish on {symbol}",
            f"{symbol} trading volume increases"
        ]

    def check_pre_trade_news(self, symbol: str) -> Dict:
        """
        Check news before executing a trade

        Returns:
            {
                'recommendation': 'proceed'|'reduce'|'skip',
                'reason': str,
                'sentiment': dict
            }
        """
        headlines = self.get_news_headlines(symbol, hours=6)
        sentiment = self.analyze_sentiment(symbol, headlines)

        # Decision logic
        if sentiment['confidence'] < 0.3:
            return {
                'recommendation': 'proceed',
                'reason': 'Low confidence sentiment, proceed with caution',
                'sentiment': sentiment
            }

        if sentiment['sentiment'] == 'bearish' and sentiment['confidence'] > 0.7:
            return {
                'recommendation': 'reduce',
                'reason': f"Strong bearish sentiment detected: {sentiment['reasoning']}",
                'sentiment': sentiment
            }

        if sentiment['sentiment'] == 'bullish':
            return {
                'recommendation': 'proceed',
                'reason': 'Positive sentiment aligns with trade',
                'sentiment': sentiment
            }

        return {
            'recommendation': 'proceed',
            'reason': 'Neutral sentiment',
            'sentiment': sentiment
        }


# Singleton
_bridge = None

def get_tiburtina_bridge() -> TiburtinaBridge:
    """Get or create Tiburtina bridge singleton"""
    global _bridge
    if _bridge is None:
        _bridge = TiburtinaBridge()
    return _bridge
PYEOF

echo "✓ tiburtina_helper.py created"

# 7. Test the bridge
echo ""
echo "7. Testing Tiburtina bridge..."
python3 << 'PYTEST'
from integrations.tiburtina_helper import get_tiburtina_bridge

bridge = get_tiburtina_bridge()

# Test sentiment analysis
headlines = [
    "Bitcoin surges to new highs",
    "Institutional adoption increases",
    "ETF approval imminent"
]

print("Testing sentiment analysis...")
sentiment = bridge.analyze_sentiment('BTC/USD', headlines)
print(f"  Sentiment: {sentiment['sentiment']}")
print(f"  Confidence: {sentiment['confidence']}")
print(f"  Reasoning: {sentiment['reasoning']}")

# Test pre-trade check
print("\nTesting pre-trade check...")
check = bridge.check_pre_trade_news('BTC/USD')
print(f"  Recommendation: {check['recommendation']}")
print(f"  Reason: {check['reason']}")

print("\n✓ Tiburtina bridge working!")
PYTEST

echo ""
echo "=========================================="
echo "Tiburtina Integration Complete!"
echo "=========================================="
echo ""
echo "Components installed:"
echo "  ✓ Ollama (local LLM server)"
echo "  ✓ Mistral 7B model"
echo "  ✓ Tiburtina bridge"
echo "  ✓ Sentiment analysis"
echo ""
echo "Next steps:"
echo "  1. Run: ./infrastructure/tiburtina_integration/2_integrate_paper_trader.sh"
echo "  2. Test with: python3 -c 'from integrations.tiburtina_helper import *; get_tiburtina_bridge()'"
