# Arena Market Benchmarks & Tiburtina Integration

## Summary

Added two major features to the Cappuccino trading system:
1. **Market Benchmarks** - Buy-and-hold portfolios to compare against DRL models
2. **Tiburtina Integration** - AI-powered market analysis in the dashboard

---

## 1. Market Benchmarks in Arena

### Overview
The Model Arena now tracks buy-and-hold benchmark portfolios alongside your DRL trading models, allowing you to see if active trading actually beats passive investing.

### Three Benchmark Strategies

1. **Equal Weight Portfolio**
   - Spreads $1000 equally across all 7 crypto assets
   - Rebalanced only at initialization
   - Shows diversification strategy performance

2. **BTC Only**
   - 100% Bitcoin allocation
   - Pure exposure to the largest crypto asset
   - Baseline for comparing against BTC

3. **60/40 BTC/ETH**
   - 60% Bitcoin, 40% Ethereum
   - Two largest cryptos weighted by preference
   - Common institutional-style allocation

### How It Works

**Automatic Initialization:**
- Benchmarks are created automatically when the arena first receives price data
- Each starts with $1000 virtual capital (same as DRL models)
- Holdings are bought at initial prices and held (no rebalancing)

**Performance Tracking:**
- Same metrics as DRL models: Return %, Sharpe Ratio, Max Drawdown
- Updated every hour when arena runs
- Saved in `arena_state/arena_state.json`

**Where to View:**
- **Arena Leaderboard** (`arena_state/leaderboard.txt`) - Shows benchmarks at the bottom
- **Dashboard Page 3** - Shows benchmark holdings and performance
- **Arena Status API** - Includes benchmark data in JSON format

### Example Output

```
================================================================================
MARKET BENCHMARKS (Buy & Hold)
================================================================================
Benchmark                      Return %     Sharpe      MaxDD %
--------------------------------------------------------------------------------
Equal Weight Portfolio            +12.50%      1.45        8.20%
BTC Only                          +15.30%      1.62        12.10%
60/40 BTC/ETH                     +13.80%      1.51        9.50%
================================================================================
```

### Files Modified

- `model_arena.py` - Added `MarketBenchmark` class and integration
- `dashboard.py` - Updated Page 3 to display benchmarks

---

## 2. Tiburtina AI Integration

### Overview
Tiburtina is an AI-powered financial research terminal. The integration brings macro data, crypto market overview, news, and AI analysis into the Cappuccino dashboard.

### New Dashboard Page 8: "Tiburtina AI"

Shows:
- **Macro Economic Snapshot** - Fed Funds rate, Treasury yields, unemployment, CPI, VIX
- **Top Crypto Markets** - Prices, 24h changes, market caps for top 10 cryptos
- **Latest Financial News** - Headlines from major financial news sources
- **About Section** - How to use Tiburtina terminal for full features

### Integration Module: `tiburtina_integration.py`

**TiburtinaClient class provides:**
- `get_macro_snapshot()` - Economic indicators
- `get_crypto_overview()` - Top crypto markets
- `get_news_summary()` - Latest headlines
- `get_portfolio_analysis()` - AI analysis of your positions
- `get_market_brief()` - AI-generated market overview
- `analyze_strategy()` - AI evaluation of trading strategies

### Error Handling

**Graceful Degradation:**
- All API calls wrapped in try/except blocks
- Errors displayed with helpful messages instead of crashes
- KeyboardInterrupt support (Ctrl+C to cancel slow calls)
- Timeout-friendly for external API calls

**Common Issues:**
- **FRED API** - May have pandas compatibility issues (shown as error)
- **API Keys** - Some features require Tiburtina API configuration
- **Network** - External API calls can be slow

### Files Created

- `tiburtina_integration.py` - Integration wrapper module
- Dashboard Page 8 - New page in `dashboard.py`

---

## Usage

### View Market Benchmarks

**In Terminal:**
```bash
# View arena leaderboard with benchmarks
cat arena_state/leaderboard.txt

# Or use arena status
python model_arena.py --show-status
```

**In Dashboard:**
```bash
python3 dashboard.py
# Press '3' to jump to Arena page
# Scroll down to see benchmarks
```

### Access Tiburtina Features

**Dashboard Page 8:**
```bash
python3 dashboard.py
# Press '8' to jump to Tiburtina AI page
# View macro data, crypto prices, and news
```

**Full Tiburtina Terminal:**
```bash
cd /home/mrc/experiment/tiburtina
python terminal/cli.py

# Commands:
/macro              # Full macro snapshot
/crypto             # Top 20 crypto markets
/news Tesla         # Search news
/brief              # AI market brief (slow, uses local LLM)
/compare BTC ETH    # AI comparison
```

---

## Configuration

### Tiburtina Setup (Optional)

If you want full Tiburtina features in the dashboard:

1. **Install Dependencies:**
   ```bash
   cd /home/mrc/experiment/tiburtina
   pip install -r requirements.txt
   ```

2. **Configure API Keys** (in Tiburtina's `.env`):
   ```env
   ALPACA_API_KEY=your_key
   ALPACA_SECRET_KEY=your_secret
   FRED_API_KEY=your_fred_key  # Optional
   NEWS_API_KEY=your_news_key  # Optional
   ```

3. **Restart Dashboard:**
   ```bash
   python3 dashboard.py
   ```

### Arena Benchmarks

No configuration needed - automatically enabled! Benchmarks initialize when arena first receives price data.

---

## Performance Analysis

### Comparing Models to Benchmarks

**Key Questions to Answer:**
- Is my DRL model beating simple buy-and-hold?
- How much better? Is the complexity worth it?
- Which market regime favors active trading?

**Metrics to Compare:**
- **Return %** - Absolute performance
- **Sharpe Ratio** - Risk-adjusted returns
- **Max Drawdown** - Worst-case loss

**Example Analysis:**
```
Model: trial_1234
  Return: +18.5%
  Sharpe: 2.1
  Max DD: 6.5%

BTC Only Benchmark:
  Return: +15.3%
  Sharpe: 1.62
  Max DD: 12.1%

Analysis: Model outperforms by 3.2% with better risk profile (higher Sharpe,
lower drawdown). Active trading adds value!
```

---

## Troubleshooting

### Page 8 Crashes or Shows Errors

**Expected behavior:** Page 8 now displays errors gracefully instead of crashing

**Common errors:**
- "FRED API compatibility issue" - Pandas/datetime incompatibility (safe to ignore)
- "Failed to fetch crypto data" - CoinGecko rate limit or network issue
- "No news available" - NewsAPI requires API key

**Solutions:**
- Press Ctrl+C to interrupt slow API calls
- Use Tiburtina terminal directly for full features
- Configure API keys in Tiburtina's `.env` file

### Benchmarks Not Showing

**Check:**
1. Is arena running? `./status_arena.sh`
2. Has arena received price data? Check `arena_state/arena_state.json`
3. Wait for next hourly update

**Fix:**
```bash
# Restart arena to trigger initialization
./stop_arena.sh
./start_arena.sh
```

---

## Future Enhancements

**Market Benchmarks:**
- [ ] More benchmark strategies (60% stocks / 40% bonds equivalent)
- [ ] Custom benchmark configuration
- [ ] Benchmark rebalancing options
- [ ] Performance attribution analysis

**Tiburtina Integration:**
- [ ] Async API calls with caching
- [ ] Portfolio-specific AI analysis
- [ ] Alert integration (news-driven trading signals)
- [ ] Strategy backtesting integration
- [ ] Macro regime detection for ensemble weight adjustment

---

## Summary

**Arena Benchmarks:**
- ✅ Three buy-and-hold strategies automatically tracked
- ✅ Compare DRL models against passive investing
- ✅ Visible in leaderboard and dashboard
- ✅ Same metrics as trading models

**Tiburtina Integration:**
- ✅ New Dashboard Page 8 with market data
- ✅ Graceful error handling (no crashes)
- ✅ Macro indicators, crypto prices, news
- ✅ Integration module for future enhancements
- ⚠️  Some features require API keys
- ⚠️  External API calls can be slow

**Impact:**
Now you can definitively answer: "Does my AI trading system beat buy-and-hold?"
And you have AI-powered market context to inform your trading decisions!
