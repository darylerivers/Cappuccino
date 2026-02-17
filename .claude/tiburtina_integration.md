# Tiburtina Integration Guide for Cappuccino

**For Claude Code working on Cappuccino**

## Overview

**Cappuccino** = ML-powered algorithmic trading system
**Tiburtina** = AI-powered financial research terminal (located at `/home/mrc/experiment/tiburtina`)

These systems are integrated to create a complete research → analysis → execution pipeline.

---

## 🚨 IMPORTANT: No Retraining Required

**Q: Does Tiburtina integration require retraining ML models?**
**A: NO - The integration is workflow-level, not model-level.**

- Cappuccino's ML models continue to work unchanged
- Integration adds intelligence AROUND the trading system
- Data sharing just changes where data is read from (same format)
- New features are additive, not replacing existing functionality

**What changes:**
- Data source location (reads from Tiburtina's parquet files)
- Additional context for decisions (macro, news, AI analysis)
- Enhanced monitoring and alerts

**What stays the same:**
- All existing ML models
- Training pipeline
- Strategy ensemble
- Core trading logic

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   TIBURTINA                         │
│  - Market data ingestion (Yahoo, Alpaca, FRED)     │
│  - News monitoring (60+ RSS sources)                │
│  - AI analysis (Local Mistral LLM)                  │
│  - Macro indicators (Fed, inflation, employment)    │
│  - SEC filings monitoring                           │
└─────────────────────────────────────────────────────┘
                         │
                         │ Shared Data & Signals
                         ↓
┌─────────────────────────────────────────────────────┐
│                   CAPPUCCINO                        │
│  - ML strategy training                             │
│  - Ensemble trading                                 │
│  - Position management                              │
│  - Risk controls                                    │
│  - Trade execution (Alpaca)                         │
└─────────────────────────────────────────────────────┘
```

---

## Integration Points

### 1. Shared Data Layer (ACTIVE)

**Location:** `/home/mrc/experiment/tiburtina/data/`

**Structure:**
```
tiburtina/data/
├── market/          # Parquet files by symbol/year
│   ├── AAPL/
│   │   ├── 2024.parquet
│   │   └── 2025.parquet
│   └── SPY/
├── macro/           # FRED indicators
├── filings/         # SEC documents
└── news/            # News article cache
```

**How to Use in Cappuccino:**
```python
from pathlib import Path
import pandas as pd

# Option 1: Import Tiburtina's storage module
import sys
sys.path.insert(0, "/home/mrc/experiment/tiburtina")
from src.storage import MarketDataStore

store = MarketDataStore(base_dir=Path("/home/mrc/experiment/tiburtina/data/market"))
df = store.load("AAPL", start="2024-01-01", end="2024-12-31")

# Option 2: Direct parquet reading
df = pd.read_parquet("/home/mrc/experiment/tiburtina/data/market/AAPL/2024.parquet")
```

**Benefits:**
- No duplicate API calls
- Single source of truth
- Consistent data across systems
- Automatic updates from Tiburtina's ingestion

---

### 2. Shared Database (AVAILABLE)

**Cappuccino DB:** `/home/mrc/experiment/cappuccino/optuna_cappuccino.db`
**Tiburtina DB:** `/home/mrc/experiment/tiburtina/db/tiburtina.db`

**Cross-System Access:**
```python
import sqlite3

# Read Tiburtina's news from Cappuccino
tiburtina_db = sqlite3.connect("/home/mrc/experiment/tiburtina/db/tiburtina.db")
news = tiburtina_db.execute("""
    SELECT title, source, published, symbols
    FROM news
    WHERE published > datetime('now', '-1 day')
    ORDER BY published DESC
""").fetchall()

# Read Cappuccino's positions from Tiburtina
cappuccino_db = sqlite3.connect("/home/mrc/experiment/cappuccino/optuna_cappuccino.db")
# (Query your positions table)
```

---

### 3. Environment Variables

**Tiburtina's .env:**
```bash
# Shared Alpaca credentials (already configured)
ALPACA_API_KEY=PKNEUS3YJZSKSGAO5AHJWO2DMI
ALPACA_SECRET_KEY=F2d9LqYFX6nqrGR2u9Bcme2FaetDycpmJZabcjNvCMDB

# Optional: Link to Cappuccino
CAPPUCCINO_DB_PATH=/home/mrc/experiment/cappuccino/optuna_cappuccino.db
```

**Cappuccino's .env (Add these):**
```bash
# Link to Tiburtina
TIBURTINA_DATA_DIR=/home/mrc/experiment/tiburtina/data
TIBURTINA_DB_PATH=/home/mrc/experiment/tiburtina/db/tiburtina.db
```

---

## Available Tiburtina Features for Cappuccino

### 1. Market Data Access

**What it provides:**
- Real-time stock quotes (Yahoo Finance)
- Historical OHLCV data
- Fundamentals (PE, margins, growth rates)
- Crypto prices (CoinGecko)

**How to use:**
```python
# Import Tiburtina's data hub
sys.path.insert(0, "/home/mrc/experiment/tiburtina")
from src.ingestion import get_hub

hub = get_hub()

# Get quote
quote = hub.get_quote("AAPL")
# Returns: {symbol, price, change, volume, market_cap, pe_ratio, ...}

# Get fundamentals
fundamentals = hub.yahoo.get_fundamentals("AAPL")
# Returns: {pe_ratio, peg_ratio, debt_to_equity, roe, ...}
```

---

### 2. Macro Indicators (FRED)

**What it provides:**
- Federal Funds Rate
- Treasury yields (10Y, 2Y, 3M)
- Inflation (CPI, Core CPI)
- Unemployment
- VIX (volatility)
- GDP, M2, housing data

**How to use:**
```python
from src.ingestion import get_hub

hub = get_hub()
macro = hub.get_macro()

# Returns dict with all indicators:
# {
#   'fed_funds': {'value': 3.88, 'date': '2025-11-01'},
#   'treasury_10y': {'value': 4.14, 'date': '2025-12-05'},
#   'unemployment': {'value': 4.40, 'date': '2025-09-01'},
#   ...
# }

# Use for regime detection
if macro['vix']['value'] > 20:
    # High volatility regime - adjust strategy weights
    pass
```

---

### 3. News Monitoring

**What it provides:**
- Real-time financial news (60+ RSS sources)
- Symbol-specific news
- Search capabilities
- Sentiment analysis (via sentiment models)

**How to use:**
```python
from src.ingestion import get_hub

hub = get_hub()

# Get latest news
news = hub.get_news()  # Latest from all sources

# Search for specific symbol/topic
tesla_news = hub.news.search("Tesla")

# Check news for your active positions
for position in get_active_positions():
    symbol_news = hub.news.search(position['symbol'])
    # Analyze sentiment, detect events
```

---

### 4. AI Analysis (Local LLM - FREE)

**What it provides:**
- Natural language analysis
- Stock comparisons
- Market regime detection
- News summarization
- Strategy suggestions

**How to use:**
```python
from src.ai.analyst import Analyst
from src.ai.summarizer import Summarizer

analyst = Analyst()
summarizer = Summarizer()

# Analyze market conditions
analysis = analyst.analyze("What is the current macro outlook for tech stocks?")

# Compare potential trades
comparison = analyst.compare(["AAPL", "MSFT"])

# Summarize news for a symbol
news_summary = summarizer.summarize_news(tesla_news, focus="TSLA")

# Generate market brief
brief = analyst.market_brief()
```

---

### 5. SEC Filings

**What it provides:**
- 10-K, 10-Q, 8-K filings
- Insider trading (Form 4)
- Institutional holdings (13F)

**How to use:**
```python
from src.ingestion import get_hub

hub = get_hub()

# Get recent filings for a symbol
filings = hub.edgar.get_company_filings("AAPL", limit=10)

# Get specific filing type
earnings = hub.edgar.get_company_filings("AAPL", form_type="10-Q")
```

---

## Integration Patterns

### Pattern 1: Pre-Trade Analysis

**Before executing a strategy:**
```python
def enhanced_entry_signal(symbol, base_signal):
    """Enhance ML signal with Tiburtina context"""

    # Get Tiburtina context
    quote = hub.get_quote(symbol)
    news = hub.news.search(symbol)
    macro = hub.get_macro()

    # AI analysis
    context = {
        'quotes': {symbol: quote},
        'macro': macro
    }
    analysis = analyst.analyze(f"Should I trade {symbol}?", context=context)

    # Combine with ML signal
    if "bearish" in analysis.lower() and base_signal > 0:
        # Reduce position size or skip
        return base_signal * 0.5

    return base_signal
```

---

### Pattern 2: Macro-Aware Position Sizing

**Adjust position sizes based on macro:**
```python
def get_macro_adjusted_size(base_size):
    """Adjust position sizing based on macro conditions"""

    macro = hub.get_macro()

    # High volatility → Reduce size
    vix = macro['vix']['value']
    if vix > 25:
        return base_size * 0.5
    elif vix > 20:
        return base_size * 0.75

    # Rising rates → Reduce leverage
    fed_funds = macro['fed_funds']['value']
    if fed_funds > 4.0:
        return base_size * 0.8

    return base_size
```

---

### Pattern 3: News-Based Exits

**Monitor news for exit signals:**
```python
def check_news_exit_signals():
    """Check if news suggests exiting positions"""

    for position in get_active_positions():
        # Get recent news
        news = hub.news.search(position['symbol'])

        # AI analysis
        summary = summarizer.summarize_news(news, focus=position['symbol'])

        # Check for bearish signals
        if "downgrade" in summary.lower() or "warning" in summary.lower():
            # Consider exiting or reducing
            logger.warning(f"Bearish news for {position['symbol']}: {summary}")
            # Your exit logic here
```

---

### Pattern 4: Strategy Discovery

**Let AI suggest new strategies:**
```python
def discover_strategies():
    """Use Tiburtina to find trading opportunities"""

    # Get market conditions
    macro = hub.get_macro()
    news = hub.get_news()[:20]

    # AI generates hypothesis
    prompt = f"""
    Given current macro: {macro}
    And recent headlines: {[n['title'] for n in news]}

    What trading strategies might work in current conditions?
    Suggest specific approaches with reasoning.
    """

    suggestion = analyst.analyze(prompt)

    # Parse suggestion and backtest
    # Add to strategy ensemble if profitable
    return suggestion
```

---

## Best Practices

### 1. Data Consistency

✅ **DO:**
- Use Tiburtina as the single source of market data
- Read from shared parquet files
- Keep data schemas aligned

❌ **DON'T:**
- Download the same data twice
- Use different time ranges in Tiburtina vs Cappuccino
- Mix data sources for the same symbol

---

### 2. AI Integration

✅ **DO:**
- Use AI for hypothesis generation and validation
- Combine AI insights with ML signals
- Log all AI suggestions for later analysis
- Set confidence thresholds for auto-execution

❌ **DON'T:**
- Blindly follow AI suggestions without validation
- Let AI override risk controls
- Execute large trades based solely on AI
- Skip backtesting AI-suggested strategies

---

### 3. Error Handling

✅ **DO:**
```python
try:
    analysis = analyst.analyze(query)
except Exception as e:
    logger.error(f"Tiburtina AI error: {e}")
    # Fall back to pure ML signal
    analysis = None
```

❌ **DON'T:**
- Assume Tiburtina is always available
- Crash Cappuccino if Tiburtina fails
- Skip trades if AI analysis fails

---

### 4. Performance

✅ **DO:**
- Cache Tiburtina results (especially AI calls)
- Use async for I/O operations
- Batch requests when possible

❌ **DON'T:**
- Call AI analysis in tight trading loops
- Make redundant API calls
- Block trading on slow AI responses

---

## Example Integration Script

**File: `cappuccino/integrations/tiburtina_helper.py`**

```python
"""
Helper functions for Tiburtina integration
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional
import sqlite3

# Add Tiburtina to path
TIBURTINA_PATH = Path("/home/mrc/experiment/tiburtina")
sys.path.insert(0, str(TIBURTINA_PATH))

from src.ingestion import get_hub
from src.ai.analyst import Analyst
from src.storage import MarketDataStore

class TiburtinaBridge:
    """Bridge between Cappuccino and Tiburtina"""

    def __init__(self):
        self.hub = get_hub()
        self.analyst = Analyst()
        self.store = MarketDataStore()
        self.db = sqlite3.connect(TIBURTINA_PATH / "db/tiburtina.db")

    def get_market_context(self, symbols: List[str]) -> Dict:
        """Get comprehensive market context for symbols"""
        return {
            'quotes': {s: self.hub.get_quote(s) for s in symbols},
            'macro': self.hub.get_macro(),
            'news': self.hub.get_news()[:20]
        }

    def analyze_trade(self, symbol: str, action: str, reasoning: str) -> str:
        """Get AI analysis of proposed trade"""
        context = self.get_market_context([symbol])

        query = f"""
        Proposed trade: {action} {symbol}
        ML Model reasoning: {reasoning}

        Given current market context, is this a good trade?
        Consider risks and provide recommendation.
        """

        return self.analyst.analyze(query, context=context)

    def get_macro_regime(self) -> str:
        """Detect current macro regime"""
        macro = self.hub.get_macro()

        vix = macro['vix']['value']
        fed_funds = macro['fed_funds']['value']

        if vix > 25:
            return "high_volatility"
        elif vix < 15:
            return "low_volatility"
        elif fed_funds > 4.5:
            return "rising_rates"
        else:
            return "normal"

    def check_news_alerts(self, symbols: List[str]) -> List[Dict]:
        """Check for important news on symbols"""
        alerts = []

        for symbol in symbols:
            news = self.hub.news.search(symbol)

            # Check for key events
            for article in news:
                if any(word in article['title'].lower()
                       for word in ['earnings', 'downgrade', 'upgrade', 'fda', 'merger']):
                    alerts.append({
                        'symbol': symbol,
                        'title': article['title'],
                        'source': article['source'],
                        'url': article['url']
                    })

        return alerts

# Singleton instance
_bridge = None

def get_tiburtina_bridge() -> TiburtinaBridge:
    """Get or create Tiburtina bridge singleton"""
    global _bridge
    if _bridge is None:
        _bridge = TiburtinaBridge()
    return _bridge
```

**Usage in Cappuccino:**
```python
from integrations.tiburtina_helper import get_tiburtina_bridge

# In your trading logic
bridge = get_tiburtina_bridge()

# Before entering trade
context = bridge.get_market_context(['AAPL'])
analysis = bridge.analyze_trade('AAPL', 'LONG', 'ML signal: 0.85')

# Check macro regime
regime = bridge.get_macro_regime()
if regime == "high_volatility":
    position_size *= 0.5

# Monitor news
alerts = bridge.check_news_alerts(active_positions)
for alert in alerts:
    logger.warning(f"News alert: {alert['symbol']} - {alert['title']}")
```

---

## Monitoring & Debugging

### Check Integration Status

```python
def check_tiburtina_status():
    """Verify Tiburtina integration is working"""

    checks = {
        'data_dir': Path("/home/mrc/experiment/tiburtina/data").exists(),
        'db_exists': Path("/home/mrc/experiment/tiburtina/db/tiburtina.db").exists(),
        'can_import': False,
        'ollama_running': False,
        'data_fresh': False
    }

    try:
        sys.path.insert(0, "/home/mrc/experiment/tiburtina")
        from src.ingestion import get_hub
        checks['can_import'] = True

        hub = get_hub()
        quote = hub.get_quote("AAPL")
        checks['data_fresh'] = quote['price'] is not None

        from src.ai.ollama_client import OllamaClient
        client = OllamaClient()
        response = client.generate("Hello", max_tokens=10)
        checks['ollama_running'] = len(response) > 0

    except Exception as e:
        logger.error(f"Tiburtina check failed: {e}")

    return checks
```

---

## Performance Considerations

**AI Response Times (Local Mistral):**
- Simple query: 2-5 seconds
- Complex analysis: 5-15 seconds
- Market brief: 10-20 seconds

**Recommendations:**
1. Don't call AI in time-critical trading loops
2. Cache AI results (use TTL based on data freshness)
3. Run AI analysis async or in background
4. Fall back to pure ML if AI times out

---

## Migration Checklist

When working on Cappuccino with Tiburtina integration:

- [ ] Check if Tiburtina data is available before using
- [ ] Handle Tiburtina failures gracefully
- [ ] Log all AI suggestions for analysis
- [ ] Backtest AI-enhanced strategies separately
- [ ] Monitor performance impact
- [ ] Document any new integration points
- [ ] Update this file if adding new patterns

---

## Questions?

- **Tiburtina docs:** `/home/mrc/experiment/tiburtina/README.md`
- **Integration plan:** `/home/mrc/experiment/tiburtina/CAPPUCCINO_INTEGRATION.md`
- **Local LLM setup:** `/home/mrc/experiment/tiburtina/LOCAL_LLM_SETUP.md`

---

## Summary

**Key Points:**
1. ✅ No retraining required - integration is additive
2. ✅ Tiburtina provides data, analysis, monitoring
3. ✅ Cappuccino makes trading decisions
4. ✅ AI enhances but doesn't replace ML signals
5. ✅ Start with data sharing, add intelligence gradually

**Integration is live for:**
- Shared Alpaca API credentials
- Cross-database queries (if needed)

**Ready to implement:**
- Shared data directory
- AI-enhanced signals
- News monitoring
- Macro-aware trading

Remember: Tiburtina is intelligence/context, Cappuccino is execution. Use Tiburtina to make Cappuccino smarter, not to replace it.
