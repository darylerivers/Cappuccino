# Cappuccino - Critical Context

**DRL crypto trading system**: PPO + Ensemble + Alpaca API + Profit Protection

---

## Quick Start

```bash
# System status
./status_automation.sh

# View latest trades
tail -50 paper_trades/alpaca_session.csv

# Run tests
pytest tests/test_critical.py -v

# Check logs
tail -100 logs/paper_trading_live.log
```

---

## Architecture (3 Layers)

```
[Training] → Optuna DB (38MB) → [Auto-Deployer]
                                       ↓
                                 [Ensemble 10 models]
                                       ↓
                    [Paper Trader + Risk + Profit Protection]
```

---

## Critical Files (Priority Order)

1. **`paper_trader_alpaca_polling.py`** - Trading loop, risk management, NEW profit protection
2. **`environment_Alpaca.py`** - RL environment, reward function (COMPLEX - needs validation)
3. **`1_optimize_unified.py`** - Hyperparameter search
4. **`ultra_simple_ensemble.py`** - Model voting (10 models → 1 action)
5. **`constants.py`** - ALL magic numbers centralized (NEW)

---

## Core Logic

### Trading Loop (paper_trader_alpaca_polling.py:643-655)
```python
action = agent.act(state)
action = apply_portfolio_protection(action)  # NEW - locks in gains
action = apply_risk_management(action)        # Per-position limits
env.step(action)
```

### Profit Protection (NEW - TODAY)
- **Portfolio Trailing Stop** (1.5% from peak) → Sell ALL
- **Profit-Taking** (3% gain) → Sell 50%
- **Move-to-Cash** (optional, 5%+) → Liquidate + 24h cooldown

**⚠️ NEEDS TESTING**: Lines 812-922 added today, not validated on real data yet.

### Risk Management
- Stop-loss: 10% per position
- Max position: 30% per asset
- Transaction cost: 0.25% (eats ~50% of profits if trading too often)

---

## Current Issues (Top 5)

1. **No Baseline Comparison** 🔴
   `baselines/buy_and_hold.py` created but not run yet. DRL may not beat passive investing.

2. **Complex Reward Function** 🔴
   `environment_Alpaca.py:224-280` has 5+ components. Unvalidated if PPO can optimize this.

3. **No Tests** 🔴
   `tests/test_critical.py` created today. Run: `pytest tests/test_critical.py -v`

4. **Hyperparameter Search May Be Inefficient** 🟡
   20+ params, 150 trials. Check importance: `contexts/training_insights.md`

5. **Transaction Costs High** 🟡
   0.25% × 2 (buy/sell) = 0.5% per round-trip. Models trade 5-15x/day.

---

## Key Metrics

**Training** (Study: cappuccino_1year_20251121):
- Best Sharpe: 2.0-3.5 (validation)
- Training time: 20-40 min/trial
- Trials completed: 150+

**Paper Trading**:
- Poll interval: 60s
- Initial capital: $1000
- Assets: 7 crypto pairs (BTC, ETH, LTC, AAVE, AVAX, LINK, UNI)
- Observed: 2-4% gains over 24-48h, but gives back 1-2% during reversals
- **After Profit Protection**: Should lock in gains (needs validation)

---

## Configuration (constants.py)

```python
from constants import RISK, NORMALIZATION, TRADING

# Risk
RISK.STOP_LOSS_PCT = 0.10               # 10%
RISK.PORTFOLIO_TRAILING_STOP_PCT = 0.015  # 1.5%
RISK.PROFIT_TAKE_THRESHOLD_PCT = 0.03    # 3%

# Normalization
NORMALIZATION.NORM_CASH = 2**-11
NORMALIZATION.NORM_ACTION = 100

# Trading
TRADING.BUY_COST_PCT = 0.0025  # 0.25%
TRADING.INITIAL_CAPITAL = 1000
```

---

## Data Flow

```
Alpaca API (REST poll 60s)
    ↓
Add indicators (TA-Lib: MACD, RSI, CCI, DX)
    ↓
Environment.step(action)
    ↓
Log to CSV (paper_trades/alpaca_session.csv)
```

---

## Quick Fixes Needed

1. **Run baseline** (2h):
   ```bash
   python baselines/buy_and_hold.py --data data/price_array_val.npy
   ```

2. **Run tests** (10min):
   ```bash
   pytest tests/test_critical.py -v
   ```

3. **Validate profit protection** (4h):
   - Check logs: `tail -100 paper_trades/profit_protection.log`
   - Verify sells trigger at thresholds
   - No bugs in `paper_trader_alpaca_polling.py:812-922`

4. **Check hyperparameter importance** (2h):
   ```python
   import optuna
   study = optuna.load_study("cappuccino_1year_20251121", "sqlite:///databases/optuna_cappuccino.db")
   optuna.visualization.plot_param_importances(study).show()
   ```

---

## Detailed Context

See `contexts/` for focused documentation:
- `trading_logic.md` - Trading components
- `risk_management.md` - Risk systems
- `training_pipeline.md` - ML training
- `monitoring.md` - Automation/monitoring

---

## Emergency Commands

```bash
# Stop all trading
./stop_automation.sh

# Kill paper trader
pkill -f paper_trader_alpaca_polling

# Check for crashes
tail -200 logs/paper_trading_failsafe.log

# View profit protection events
tail -50 paper_trades/profit_protection.log
```

---

**Status**: ✅ Operational (paper trading active, no live capital)
**Last Updated**: 2025-11-24
**Full Docs**: `PROJECT_OVERVIEW_FOR_OPUS.md` (33KB)
