# Paper Trading Grading & Live Trading Promotion System

## Overview

A two-tier automated trading system that grades paper trading performance and promotes to live Coinbase trading when criteria are met.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PAPER TRADING (Alpaca)                       │
│  • Train models → Deploy to ensemble → Trade on paper          │
│  • Performance tracked continuously                              │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│              PERFORMANCE GRADING SYSTEM                          │
│  Script: performance_grader.py                                   │
│                                                                  │
│  Evaluation Criteria (all must pass):                           │
│  ✓ Minimum 7 days of trading history                           │
│  ✓ 80% win rate (profitable trades)                            │
│  ✓ Positive alpha vs market                                    │
│  ✓ Sharpe ratio > 0.5                                          │
│  ✓ Maximum drawdown < 15%                                      │
│  ✓ Minimum 20 trades for statistical significance              │
│  ✓ Overall positive return                                     │
│                                                                  │
│  Grading:                                                        │
│  • A: 95%+  (Excellent)                                        │
│  • B: 85%+  (Good)                                             │
│  • C: 70%+  (Fair)                                             │
│  • D: 50%+  (Poor)                                             │
│  • F: <50%  (Failed)                                           │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PROMOTION DECISION                             │
│  If grade ≥ 80% AND all criteria met:                          │
│  → Ready for promotion                                          │
│  → Promotion flag set in grading_state.json                    │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│              LIVE TRADING (Coinbase)                             │
│  Script: coinbase_live_trader.py                                 │
│                                                                  │
│  Safety Features:                                                │
│  ✓ Requires promotion verification                             │
│  ✓ Conservative position limits (10% max)                      │
│  ✓ 5% stop loss per position                                   │
│  ✓ 20% emergency portfolio stop                                │
│  ✓ Dry-run mode for testing                                    │
│  ✓ Complete audit trail                                        │
│                                                                  │
│  Authentication:                                                 │
│  • Coinbase CDP API (Ed25519)                                  │
│  • Key file: key/cdp_api_key.json                              │
│  • Portfolio: TestPortfolio                                    │
└─────────────────────────────────────────────────────────────────┘
```

## Current Status

### ✅ COMPLETED

1. **Performance Grader** (`performance_grader.py`)
   - Loads 7-day rolling window of trading history
   - Calculates comprehensive metrics (win rate, alpha, Sharpe, drawdown)
   - Grades performance (A-F scale)
   - Tracks promotion readiness
   - Saves state to `deployments/grading_state.json`

2. **Coinbase Live Trader** (`coinbase_live_trader.py`)
   - CDP API authentication (Ed25519)
   - Dry-run and live modes
   - Promotion verification (blocks live trading until promoted)
   - Emergency stop loss (20% portfolio drawdown)
   - Conservative risk management
   - Audit logging

### Current Paper Trader Performance

```
Trading Period: 6.7 days (need 7.0)
Total Trades: 65
Win Rate: 49.2% (need 80%)
Total Return: +3.66%
Alpha: +3.66% ✓
Sharpe Ratio: 2.12 ✓
Max Drawdown: -18.0% (need < -15%)

Grade: D (57.1%)
Status: ❌ NOT READY for live trading

Failing Criteria:
  • Trading period: 6.7 days < 7 days
  • Win rate: 49.2% < 80%
  • Max drawdown: -18% > -15%
```

## Usage

### 1. Check Paper Trading Grade

```bash
python performance_grader.py --check
```

This evaluates the last 7 days of trading and shows:
- Detailed performance metrics
- Grade (A-F)
- Pass/fail status
- Which criteria are failing

### 2. View Current Status

```bash
python performance_grader.py --status
```

Shows:
- Promotion status
- Last grade
- Grade history

### 3. Promote to Live Trading

After paper trader meets all criteria:

```bash
python performance_grader.py --promote
```

This sets the `promoted_to_live` flag.

### 4. Test Live Trader (Dry-Run)

```bash
python coinbase_live_trader.py --mode dry-run
```

Simulates live trading without executing real orders.

### 5. Start Live Trading

⚠️ **ONLY after promotion!**

```bash
python coinbase_live_trader.py --mode live --model-dir train_results/ensemble
```

This will:
- Verify promotion status
- Refuse to start if not promoted
- Execute real trades on Coinbase
- Monitor portfolio with emergency stops

## Integration with Automation

### Add to Watchdog

The grading system will be integrated into `system_watchdog.py` to:

1. **Hourly Grading** - Run `performance_grader.py --check` every hour
2. **Auto-Promotion** - When criteria met, promote automatically
3. **Live Trader Launch** - Start Coinbase live trader when promoted
4. **Continuous Monitoring** - Monitor both paper and live traders

### Configuration Files

**grading_state.json:**
```json
{
  "promoted_to_live": false,
  "ready_for_promotion": false,
  "promotion_date": null,
  "last_grade": {
    "grade": "D",
    "score": 57.1,
    "passed": false,
    "reason": "Failed: min_days, win_rate, drawdown"
  }
}
```

## Safety Features

### Paper Trading Safeguards
- Alpha decay monitoring (-3% threshold)
- Automatic model retraining
- Ensemble model updates
- Trailing stop losses

### Grading Safeguards
- Minimum 7 days prevents premature promotion
- 80% win rate ensures consistency
- Statistical significance (20+ trades)
- Risk-adjusted returns (Sharpe ratio)
- Drawdown limits protect capital

### Live Trading Safeguards
- **Promotion Verification** - Hard block on live trading until promoted
- **Conservative Limits** - 10% max position (vs 30% paper)
- **Stop Losses** - 5% per position (vs 10% paper)
- **Emergency Stop** - 20% portfolio drawdown triggers shutdown
- **Dry-Run Mode** - Test everything before going live
- **Audit Trail** - Complete logging of all decisions

## Coinbase API Setup

The system uses Coinbase CDP API with Ed25519 authentication:

**Key File Format** (`key/cdp_api_key.json`):
```json
{
  "id": "7ad324ed-85af-492d-b08a-74cd685d86ed",
  "privateKey": "Opn7ineK1Xk0ZzpyHonS0wDaQS/3RAKPjYaRxq9G7hMmPJhHOZWv2aev0a+Mtm4khpD0qv6EnGNDpQu/kyS4LA=="
}
```

**Note:** The API authentication needs to be verified with Coinbase Advanced Trade API.
Current status: 401 Unauthorized (signature method may need adjustment)

## Logs

All operations are logged:

```bash
logs/performance_grader.log       # Grading evaluations
logs/coinbase_live_dry-run_*.log  # Dry-run trading
logs/coinbase_live_live_*.log     # Live trading
deployments/grading_state.json    # Promotion state
```

## Commands Summary

```bash
# Check if ready for live trading
python performance_grader.py --check

# View promotion status
python performance_grader.py --status

# Promote to live (after passing criteria)
python performance_grader.py --promote

# Test live trader (safe)
python coinbase_live_trader.py --mode dry-run

# Start live trading (requires promotion)
python coinbase_live_trader.py --mode live

# Integrated automation (future)
./start_automation.sh  # Will include grading + live trading
```

## Next Steps

1. **Fix Coinbase API Authentication**
   - Verify signing method for Advanced Trade API
   - Test with actual API credentials
   - Validate permissions on TestPortfolio

2. **Integrate into Watchdog**
   - Add hourly grading checks
   - Auto-promote when criteria met
   - Launch live trader on promotion

3. **Complete Model Integration**
   - Load ensemble models into live trader
   - Implement signal generation
   - Execute trades based on model predictions

4. **Testing**
   - Run dry-run mode for 24 hours
   - Verify all safety stops work
   - Test emergency scenarios

