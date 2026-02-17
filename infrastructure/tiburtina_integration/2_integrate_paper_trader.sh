#!/bin/bash
# Integrate Tiburtina sentiment into paper trader

set -e

echo "=========================================="
echo "Integrating Tiburtina with Paper Trader"
echo "=========================================="
echo ""

# 1. Backup paper trader
echo "1. Backing up paper_trader_alpaca_polling.py..."
cp paper_trader_alpaca_polling.py paper_trader_alpaca_polling.py.backup_$(date +%Y%m%d_%H%M%S)
echo "✓ Backup created"

# 2. Create sentiment-enhanced paper trader patch
echo ""
echo "2. Creating sentiment integration patch..."
cat > infrastructure/tiburtina_integration/sentiment_integration.patch << 'EOF'
Add Tiburtina sentiment analysis to paper trader:

1. Import at top of file:
   from integrations.tiburtina_helper import get_tiburtina_bridge

2. In __init__ method:
   self.use_sentiment = True  # Enable sentiment
   self.sentiment_bridge = get_tiburtina_bridge() if self.use_sentiment else None

3. Before trade execution (in process_signal or similar method):
   if self.use_sentiment and self.sentiment_bridge:
       news_check = self.sentiment_bridge.check_pre_trade_news(symbol)

       if news_check['recommendation'] == 'skip':
           print(f"  📰 NEWS: Skipping trade - {news_check['reason']}")
           return  # Skip trade

       elif news_check['recommendation'] == 'reduce':
           print(f"  ⚠️  NEWS: Reducing position - {news_check['reason']}")
           position_size *= 0.5  # Cut position in half

       else:
           print(f"  📈 NEWS: {news_check['sentiment']['sentiment']} sentiment - proceeding")

4. Add --enable-sentiment flag to argparse:
   parser.add_argument('--enable-sentiment', action='store_true',
                      help='Enable Tiburtina sentiment analysis')
EOF

echo "✓ Patch guide created"

# 3. Create test script
echo ""
echo "3. Creating sentiment test script..."
cat > test_sentiment_paper_trader.py << 'PYEOF'
#!/usr/bin/env python3
"""
Test sentiment-enhanced paper trading
"""
from integrations.tiburtina_helper import get_tiburtina_bridge

def test_sentiment_workflow():
    """Test full sentiment workflow"""

    print("=" * 60)
    print("Testing Sentiment-Enhanced Trading Workflow")
    print("=" * 60)
    print()

    bridge = get_tiburtina_bridge()

    # Test symbols
    symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD']

    for symbol in symbols:
        print(f"\n{symbol}:")
        print("-" * 40)

        # Get news check
        check = bridge.check_pre_trade_news(symbol)

        print(f"  Recommendation: {check['recommendation']}")
        print(f"  Reason: {check['reason']}")
        print(f"  Sentiment: {check['sentiment']['sentiment']}")
        print(f"  Confidence: {check['sentiment']['confidence']:.2f}")

        # Simulate trade decision
        if check['recommendation'] == 'skip':
            print(f"  ❌ Trade SKIPPED")
        elif check['recommendation'] == 'reduce':
            print(f"  ⚠️  Position REDUCED by 50%")
        else:
            print(f"  ✅ Trade PROCEEDING at full size")

    print("\n" + "=" * 60)
    print("✓ Sentiment workflow test complete")
    print("=" * 60)

if __name__ == '__main__':
    test_sentiment_workflow()
PYEOF

chmod +x test_sentiment_paper_trader.py

echo "✓ Test script created"

# 4. Run test
echo ""
echo "4. Running sentiment test..."
python3 test_sentiment_paper_trader.py

echo ""
echo "=========================================="
echo "Integration Ready!"
echo "=========================================="
echo ""
echo "To use sentiment in paper trading:"
echo ""
echo "Option 1: Manual integration (recommended for control)"
echo "  Edit paper_trader_alpaca_polling.py following:"
echo "  infrastructure/tiburtina_integration/sentiment_integration.patch"
echo ""
echo "Option 2: Test mode"
echo "  python3 test_sentiment_paper_trader.py"
echo ""
echo "After integration, start paper trader with:"
echo "  python paper_trader_alpaca_polling.py --enable-sentiment [other args...]"
echo ""
echo "Sentiment analysis adds:"
echo "  • Pre-trade news checks (bearish headlines = skip/reduce)"
echo "  • LLM-powered sentiment (bullish/bearish/neutral)"
echo "  • Confidence scoring (0-1)"
echo "  • Automatic position sizing adjustment"
