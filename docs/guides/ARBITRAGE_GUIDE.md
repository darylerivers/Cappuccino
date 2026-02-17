# Crypto Arbitrage Scanner - Quick Guide

## Overview

The arbitrage scanner runs alongside your paper trading system, monitoring for **triangular arbitrage** opportunities across crypto-USD pairs.

## What is Triangular Arbitrage?

When cross-currency market prices deviate from implied rates calculated from USD pairs:

**Example:**
```
BTC/USD = $95,000
ETH/USD = $3,200
Implied BTC/ETH = 95000 / 3200 = 29.69 ETH per BTC

If actual BTC/ETH market = 29.50:
→ Arbitrage opportunity!
→ Profit = 0.19 ETH per BTC (~0.64% before fees)
```

**Execution Path:**
1. Buy BTC/ETH @ 29.50 (spend 1 BTC, get 29.50 ETH)
2. Sell BTC/USD @ 95,000 (sell 1 BTC, get $95,000)
3. Buy ETH/USD @ 3,200 (spend $95,000, get 29.69 ETH)
4. Net profit: 0.19 ETH (~$608)

## Quick Start

### Option 1: Run with Existing Paper Trading

```bash
# Stop current paper trading if running
pkill -f paper_trading_failsafe.sh

# Start paper trading + arbitrage scanner
./paper_trading_with_arbitrage.sh
```

This starts:
- **Paper trading** with RL model (main trading strategy)
- **Arbitrage scanner** monitoring for cross-currency opportunities

### Option 2: Run Scanner Standalone

```bash
# Monitor for 0.5% profit opportunities, scan every 5 minutes
python arbitrage_scanner.py --min-profit 0.005 --interval 300
```

## Configuration

### Minimum Profit Threshold

Default: **0.5%** (after transaction costs)

```bash
# More aggressive (0.3% = catch smaller opportunities)
./paper_trading_with_arbitrage.sh trial_3358_1h 300 0.003

# More conservative (1% = only large opportunities)
./paper_trading_with_arbitrage.sh trial_3358_1h 300 0.01
```

**Why 0.5%?**
- Each arbitrage requires 3 trades
- Transaction fee: 0.25% per trade
- Total costs: 0.75%
- Need 1.25%+ spread to profit 0.5%

### Scan Interval

Default: **300 seconds** (5 minutes)

```bash
# Scan every 1 minute (more opportunities, higher API usage)
./paper_trading_with_arbitrage.sh trial_3358_1h 60 0.005

# Scan every 10 minutes (conservative)
./paper_trading_with_arbitrage.sh trial_3358_1h 600 0.005
```

## Monitoring

### Dashboard

```bash
python dashboard.py
```

Shows:
- Arbitrage Scanner: ACTIVE/INACTIVE
- Recent opportunities count
- System health

### Logs

**Scanner activity:**
```bash
tail -f logs/arbitrage_scanner.log
```

**Opportunities found:**
```bash
tail -f logs/arbitrage_opportunities.json
```

Example opportunity:
```json
{
  "type": "triangular",
  "direction": "AVAX → BTC → USD → AVAX",
  "legs": [
    "Buy BTC/AVAX @ 6098.500000",
    "Sell BTC/USD @ 95830.08",
    "Buy AVAX/USD @ 15.71"
  ],
  "implied_rate": 6100.2,
  "actual_rate": 6098.5,
  "spread_pct": 0.028,
  "net_profit_pct": -0.722,
  "timestamp": "2025-11-15T23:40:12.123456"
}
```

## How It Works

### Detection Algorithm

For each pair of assets (A, B):

1. **Calculate implied rate** from USD pairs:
   ```
   implied_A_per_B = price_A_USD / price_B_USD
   ```

2. **Fetch actual cross-pair price** (if market exists):
   ```
   actual_A_per_B = get_market_price("A/B")
   ```

3. **Calculate spread**:
   ```
   spread = (implied_rate - actual_rate) / actual_rate
   ```

4. **Check profitability**:
   ```
   net_profit = spread - (3 × transaction_fee)

   if net_profit > min_profit_threshold:
       → OPPORTUNITY FOUND!
   ```

### Execution Strategy (Current: Logging Only)

**⚠️ Important:** The scanner currently only **detects and logs** opportunities. It does **not execute** trades automatically.

To enable execution, you would need to:
1. Add Alpaca trading API credentials
2. Implement order execution logic
3. Handle slippage and partial fills
4. Add risk management (position limits, max drawdown, etc.)

## Integration with RL Model

### Hybrid Approach

**RL Model (Primary):**
- Handles main portfolio management
- Uses technical indicators and price history
- Optimizes for beating equal-weight benchmark
- Trades USD pairs

**Arbitrage Scanner (Opportunistic):**
- Monitors for inefficiencies
- Executes when profitable (if enabled)
- Independent of RL model decisions
- Trades both USD and cross-pairs

### Why This Works

1. **Different Timescales**
   - RL: Minutes to hours (based on 1h candles)
   - Arbitrage: Seconds to minutes (real-time pricing)

2. **Different Strategies**
   - RL: Predictive (learns patterns)
   - Arbitrage: Reactive (exploits pricing errors)

3. **Complementary**
   - RL handles majority of trading
   - Arbitrage captures rare inefficiencies
   - Combined returns > either alone

## Real-World Considerations

### Why Arbitrage is Rare

1. **Efficient Markets**: Crypto markets are highly liquid and efficient
2. **Fast Bots**: Many arbitrage bots exist already
3. **Transaction Costs**: 0.75% round-trip is high
4. **Slippage**: Prices move during execution
5. **Latency**: Network delays matter

### When Opportunities Appear

- **High Volatility**: During price swings
- **Low Liquidity**: Thin order books on cross-pairs
- **Market Inefficiency**: News events, network issues
- **Flash Crashes**: Temporary pricing errors

### Expected Frequency

With 0.5% threshold:
- **Bear market**: 0-2 per day
- **Bull market**: 1-5 per day
- **High volatility**: 5-20 per day

Most are small (0.5-1% profit).

## Troubleshooting

### Scanner Not Finding Opportunities

**This is normal!** Crypto markets are efficient.

Try:
- Lower `--min-profit` to 0.003 (0.3%)
- Reduce `--interval` to 60 seconds
- Add more USD pairs (more combinations)

### Scanner Keeps Restarting

Check logs:
```bash
tail -50 logs/arbitrage_scanner.log | grep ERROR
```

Common issues:
- API rate limiting
- Network connectivity
- Invalid cross-pair symbols

### High CPU Usage

Scanner is lightweight, but frequent scans + many pairs = CPU usage.

Reduce:
- Increase `--interval`
- Reduce number of USD pairs in config

## Advanced: Adding Execution

To enable automatic arbitrage execution:

1. **Add trading logic** to `arbitrage_scanner.py`
2. **Configure Alpaca API** with trading permissions
3. **Implement order management**:
   - Market orders vs. limit orders
   - Partial fill handling
   - Timeout logic
4. **Add risk management**:
   - Maximum position size
   - Daily loss limits
   - Circuit breakers

**⚠️ Warning:** Automated execution requires careful testing!

## Status

### Current Implementation

✅ Triangular arbitrage detection
✅ Real-time price monitoring
✅ Profit calculation (after fees)
✅ Opportunity logging
✅ Dashboard integration
❌ Automatic execution (manual intervention required)
❌ Order management
❌ Slippage handling

### Roadmap

- [ ] Automatic execution engine
- [ ] Slippage estimation
- [ ] Historical backtesting
- [ ] Multi-hop arbitrage (4+ legs)
- [ ] Cross-exchange arbitrage

---

**Last Updated:** 2025-11-15
**Version:** 1.0.0 (Detection Only)
