# Function Index - Quick Reference

## Trading Core

### paper_trader_alpaca_polling.py
- `run()` - Main trading loop (line 948)
- `_process_new_bar()` - Execute trades (line 583)
- `_apply_risk_management()` - Per-position limits (line 699)
- `_apply_portfolio_profit_protection()` - Portfolio protection (line 812) **NEW**
- `_select_action()` - Get model prediction (line 677)
- `_fetch_latest_bars()` - Poll Alpaca API (line 544)

### environment_Alpaca.py
- `step()` - Execute action, return reward (line 127)
- `reset()` - Initialize episode (line 115)
- `get_state()` - Build state vector (line 250)
- **Reward calculation** - Lines 224-280 (COMPLEX)

## Ensemble & Models

### ultra_simple_ensemble.py
- `act()` - Average 10 model predictions (line 90)
- `get_voting_breakdown()` - Individual votes (line 120)
- `get_required_hyperparameters()` - Consensus config (line 150)

### adaptive_ensemble_agent.py
- `act()` - Weighted voting (line 85)
- `update_scores()` - Track model performance (line 140)
- `eliminate_worst()` - Remove poor models (line 180)

## Training

### 1_optimize_unified.py
- `objective()` - Optuna trial function (line 200)
- `create_hyperparameter_space()` - Define search (line 350)
- `run_training()` - Train single trial (line 450)

### 2_validate.py
- `validate_model()` - Out-of-sample eval (line 50)
- `calculate_metrics()` - Sharpe, returns, DD (line 150)

## Automation

### auto_model_deployer.py
- `select_top_models()` - Query Optuna DB (line 120)
- `deploy_ensemble()` - Copy checkpoints (line 200)
- `check_for_improvement()` - Compare performance (line 280)

### system_watchdog.py
- `check_process_health()` - Monitor PIDs (line 80)
- `restart_process()` - Auto-restart (line 150)

## Metrics & Analysis

### function_finance_metrics.py
- `sharpe_iid()` - Calculate Sharpe ratio (line 63)
- `max_drawdown_ndarray()` - Max DD (line 67)
- `compute_data_points_per_year()` - Annualization (line 36)

## Baselines (NEW)

### baselines/buy_and_hold.py
- `run()` - Execute strategy (line 30)
- `_calculate_metrics()` - Performance (line 60)

## Tests (NEW)

### tests/test_critical.py
- `TestStopLoss` - Validate stop-loss (line 18)
- `TestPositionLimits` - Validate limits (line 40)
- `TestEnsembleVoting` - Validate averaging (line 70)
- `TestProfitProtection` - Validate NEW logic (line 100)
- `TestStateNormalization` - Validate consistency (line 180)

## Configuration

### constants.py (NEW)
- `RISK` - Risk parameters (line 11)
- `NORMALIZATION` - State scaling (line 27)
- `TRADING` - Trading config (line 43)
- `TRAINING` - Optimization config (line 65)

### config_main.py
- Global settings (tickers, timeframes, windows)
- `TICKER_LIST` - Assets to trade (line 105)
- `TIMEFRAME` - Bar size (line 56)
- `TRAIN_WINDOW_HOURS` / `VAL_WINDOW_HOURS` (lines 67-68)
