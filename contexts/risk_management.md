# Risk Management Context

## Two-Layer System

### Layer 1: Per-Position Risk Management
**File**: `paper_trader_alpaca_polling.py:699-773`

**Rules**:
1. **Stop-Loss** (default 10%)
   - Tracks `entry_prices` dict
   - Triggers when `(entry - current) / entry >= 0.10`
   - Force sells entire position

2. **Trailing Stop** (optional, disabled by default)
   - Tracks `high_water_mark` per position
   - Triggers when `(high - current) / high >= threshold`

3. **Position Size Limits** (default 30%)
   - Caps buy orders: `max_value = 0.30 * total_portfolio`
   - Ensures diversification (max 3-4 assets at full allocation)

4. **Action Dampening** (optional)
   - Scales all actions: `action *= dampening_factor`

**Logic**:
```python
def _apply_risk_management(action, prices, timestamp):
    for asset in assets:
        # Check stop-loss
        if loss_pct >= STOP_LOSS_PCT:
            action[asset] = -holdings[asset] * 1.1  # Sell all

        # Check position limits on buys
        if action[asset] > 0:
            if new_position_pct > MAX_POSITION_PCT:
                action[asset] = cap_to_limit()

    return action
```

### Layer 2: Portfolio-Level Profit Protection (NEW)
**File**: `paper_trader_alpaca_polling.py:812-922`
**Added**: 2025-11-24 (TODAY - NEEDS TESTING)

**Rules** (applied in order):
1. **Cash Mode Cooldown** (24h)
   - If in cooldown: Block ALL buys
   - After cooldown: Reset and resume trading

2. **Portfolio Trailing Stop** (1.5% from peak)
   - Tracks `portfolio_high_water_mark`
   - If `(peak - current) / peak >= 0.015`: **SELL EVERYTHING**
   - Prevents giving back gains

3. **Move-to-Cash** (optional, default disabled)
   - At high threshold (e.g., 5%): Liquidate 100%
   - Enter cooldown for 24h
   - Conservative profit locking

4. **Partial Profit-Taking** (3% trigger, sell 50%)
   - When `(current / initial - 1) >= 0.03`: Sell 50% of all holdings
   - Only triggers once per cycle
   - Locks in half, keeps upside

**Logic**:
```python
def _apply_portfolio_profit_protection(action, prices, timestamp):
    current_portfolio = cash + sum(holdings * prices)
    gain_from_start = (current_portfolio / initial_portfolio - 1)
    drawdown_from_peak = (peak - current_portfolio) / peak

    # Priority order matters!
    if in_cash_mode and hours < COOLDOWN:
        return block_all_buys(action)

    if drawdown_from_peak >= TRAILING_STOP_PCT:
        return sell_everything(action)

    if gain_from_start >= MOVE_TO_CASH_PCT:
        enter_cash_mode()
        return sell_everything(action)

    if gain_from_start >= PROFIT_TAKE_PCT and not profit_taken:
        return sell_partial(action, 0.5)

    return action
```

## Execution Order (CRITICAL)

```python
# In _process_new_bar() at line 643-651
raw_action = agent.act(state)
action = apply_portfolio_protection(raw_action)  # Portfolio level FIRST
action = apply_risk_management(action)           # Per-position SECOND
env.step(action)
```

**Why this order?**
- Portfolio protection overrides everything (e.g., "sell all now")
- Per-position limits then refine (e.g., "but respect min qty")

## Configuration

**CLI Arguments**:
```bash
# Per-position
--max-position-pct 0.30
--stop-loss-pct 0.10
--trailing-stop-pct 0.0  # Disabled

# Portfolio-level
--portfolio-trailing-stop-pct 0.015  # 1.5%
--profit-take-threshold-pct 0.03     # 3%
--profit-take-amount-pct 0.50        # Sell 50%
--move-to-cash-threshold-pct 0.0     # Disabled
--cooldown-after-cash-hours 24

# Disable all
--no-risk-management
--no-profit-protection
```

**From constants.py**:
```python
from constants import RISK
RISK.STOP_LOSS_PCT = 0.10
RISK.PORTFOLIO_TRAILING_STOP_PCT = 0.015
```

## Logs

**Profit Protection Events**:
```bash
tail -f paper_trades/profit_protection.log
```

Example:
```
[2025-11-24T12:00:00Z] Initial portfolio value: $1000.00
[2025-11-24T14:30:00Z] New portfolio high: $1030.00 (+3.0% from start)
[2025-11-24T15:00:00Z] PROFIT TAKING: Portfolio up 3.1% - selling 50% of positions
[2025-11-24T16:00:00Z] PORTFOLIO TRAILING STOP: Down 1.6% from peak ($1030.00 → $1013.50)
```

## Known Issues

1. **Thresholds May Be Too Tight** 🟡
   - 1.5% trailing stop = tight for crypto volatility
   - BTC can swing 2% in minutes
   - **Investigation**: Backtest optimal thresholds on historical data

2. **No Asset-Specific Thresholds** 🟡
   - BTC vs. altcoins have different volatility
   - Consider dynamic thresholds based on realized vol

3. **Profit Protection Logic Untested** 🔴
   - Added today (2025-11-24)
   - **TODO**: Run tests, validate on historical paper trading logs

## Testing

**Unit Tests**: `tests/test_critical.py`
```bash
pytest tests/test_critical.py::TestProfitProtection -v
pytest tests/test_critical.py::TestStopLoss -v
pytest tests/test_critical.py::TestPositionLimits -v
```

## Quick Fixes

**Adjust Thresholds**:
```python
# In paper_trading_failsafe.sh or direct call
python paper_trader_alpaca_polling.py \
    --portfolio-trailing-stop-pct 0.02 \  # Loosen to 2%
    --profit-take-threshold-pct 0.05 \    # Wait for 5%
    ...
```

**Disable Profit Protection Temporarily**:
```bash
python paper_trader_alpaca_polling.py \
    --no-profit-protection \
    ...
```
