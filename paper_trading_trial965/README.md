# Paper Trading - Trial #965

CGE-augmented trading model running in paper trading mode.

## Quick Start

```bash
./START.sh      # Start trading (default: 60 min)
./MONITOR.sh    # View live dashboard
./STATUS.sh     # Quick status check
./STOP.sh       # Stop trading
```

## Model Performance

**Training:**
- Sharpe: 0.0118
- Outperformance: 13.3x vs HODL

**Stress Test (200 CGE scenarios):**
- Mean Sharpe: 11.52
- Min Sharpe: 2.38 (worst case still profitable)
- Max Drawdown: -3.6%
- Win Rate: 84.8%
- Zero failures

**By Market Regime:**
- Bear Markets: Sharpe 4.30, 58% win rate
- Normal Markets: Sharpe 15.66, 100% win rate

## Configuration

**Trading Settings:**
- Initial Capital: $1,000
- Update Interval: 30 seconds
- Emergency Stop: -20% drawdown
- Assets: AAVE, AVAX, BTC, LINK, ETH, LTC, UNI

**Files:**
```
config/model_config.json      # Model hyperparameters
config/trading_config.json    # Trading settings
```

## Directory Structure

```
paper_trading_trial965/
├── config/          # Configuration files
├── models/          # Trained model files
├── logs/            # Trading logs
├── results/         # Performance CSVs
├── paper_trader.py  # Main trading bot
├── monitor.py       # Real-time dashboard
├── START.sh         # Start trading
├── MONITOR.sh       # Launch monitor
├── STATUS.sh        # Quick status
└── STOP.sh          # Stop trading
```

## Performance Tracking

Results are auto-saved to `results/performance_YYYYMMDD_HHMMSS.csv`:
- timestamp
- iteration
- portfolio_value
- cash
- market_return

View live logs: `tail -f logs/fast_*.log`

## Notes

- Updates every 30 seconds with immediate CSV saves
- Press Ctrl+C to stop early
- Emergency stop triggers at -20% drawdown
- Monitor updates every 10 seconds

**Last Updated:** 2026-01-26
