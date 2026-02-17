# Cappuccino - Automated Crypto Trading with Deep Reinforcement Learning

## Project Overview

Cappuccino is an end-to-end automated cryptocurrency trading system using Deep Reinforcement Learning (DRL). The system trains DRL agents via Optuna hyperparameter optimization, evaluates them through paper trading, and can deploy them for live trading on Alpaca Markets.

**Current Status:** Production-ready with 500 trained models, active paper trading of top performers.

## Core Architecture

### 1. Training Pipeline (Optuna + DRL)
- **Framework:** Optuna for hyperparameter search, PyTorch for models
- **Algorithms:** DDQN (primary), A2C, PPO
- **Search Space:** 20+ hyperparameters including learning rates, network architecture, lookback periods
- **Studies:** 5 parallel studies (cappuccino_3workers_1h, cappuccino_alpaca_v2, cappuccino_ft_transformer, etc.)
- **Database:** SQLite (databases/pipeline_v2.db, optuna_*.db files)
- **Training Data:** Alpaca crypto market data (OHLCV + 9 technical indicators)

### 2. Agent Architecture
- **State Space:** Cash + Holdings + Technical Indicators × Lookback
  - Formula: `state_dim = 1 (cash) + num_tickers + (num_tickers × 14 indicators) × lookback`
  - Example: 7 tickers, lookback=10 → state_dim = 988
- **Action Space:** Continuous actions per ticker (buy/sell/hold proportions)
- **Networks:**
  - Baseline: 3-layer MLP (configurable hidden sizes)
  - FT-Transformer: Feature Transformer encoder with frozen/trainable options
- **Technical Indicators (14/ticker):** OHLCV, MACD, RSI, CCI, DX, ATR regime shift, range breakout volume, trend reacceleration

### 3. Ensemble Voting System
- **Conservative Ensemble:** Top-N models vote, requires 60-80% agreement for trades
- **Aggressive Ensemble:** Lower voting threshold, more active trading
- **Model Selection:** Best Sharpe ratio models from training campaigns

### 4. Paper Trading (Alpaca Markets)
- **Polling Mode:** Fetches 1h bars, executes trades at bar close
- **State Construction:** Matches training environment exactly (critical for performance)
- **CSV Logging:** All trades, positions, rewards logged to paper_trades/*.csv
- **Performance Monitoring:** Live Sharpe ratio calculation, alert thresholds

## File Structure

```
cappuccino/
├── scripts/
│   ├── training/
│   │   ├── 1_optimize_unified.py          # Main Optuna training
│   │   ├── robust_training_wrapper.py     # Fault-tolerant wrapper
│   │   └── 2_backtest_best_trial.py       # Evaluate trained models
│   ├── deployment/
│   │   ├── paper_trader_alpaca_polling.py # Live paper trading
│   │   └── deploy_top_models.py           # Batch deployment
│   └── automation/
│       ├── auto_train.py                  # Continuous training orchestrator
│       └── watchdog_training.py           # Crash recovery
├── monitoring/
│   ├── live_performance_monitor.py        # Real-time performance tracking
│   ├── dashboard.py                       # TUI monitoring dashboard
│   └── training_monitor_discord.py        # Discord notifications
├── databases/
│   ├── pipeline_v2.db                     # Model registry
│   └── optuna_*.db                        # Optuna study databases
├── train_results/
│   └── cwd_*/trial_*_1h/                  # Trained model checkpoints
├── paper_trades/
│   ├── trial*_session.csv                 # Paper trading logs
│   └── positions_state.json               # Current positions
├── logs/                                  # All system logs
├── environment_Alpaca.py                  # Training environment
└── agent_ddqn.py                          # DRL agent implementations
```

## Key Components

### Training Environment (environment_Alpaca.py)
- Custom OpenAI Gym environment
- Simulates crypto portfolio management
- Reward: Portfolio value change (can be log, sharpe-based, or simple returns)
- Episode: One pass through training data (~120 bars for 5-day period)
- Normalization: Cash, stocks, and technical indicators normalized to [0,1] range

### Paper Trading (paper_trader_alpaca_polling.py)
- Loads trained model checkpoint
- Fetches live market data from Alpaca API
- Constructs state matching training environment
- Inference through model → trading actions
- Executes trades via Alpaca paper trading API
- **Critical:** Lookback parameter must match training or state dimensions mismatch

### Performance Monitoring (live_performance_monitor.py)
- Calculates rolling Sharpe ratio from paper trading CSV
- Compares live performance vs backtest target
- Alert thresholds: Warning (-0.5 vs backtest), Critical (<-1.0), Emergency (<-2.0 for 24h)
- Sends notifications on performance degradation

## Current Active Traders (Feb 2026)

### Trial #100 (Ensemble Conservative)
- **Backtest Sharpe:** 0.1768
- **Architecture:** 3-layer MLP, ensemble voting
- **Lookback:** 5 bars (5 hours)
- **State Dim:** 498
- **Status:** Paper trading active since Feb 9 22:42

### Trial #91 (FT-Transformer)
- **Backtest Sharpe:** 0.1784 (BEST model)
- **Architecture:** Feature Transformer encoder
- **Lookback:** 10 bars (10 hours)
- **State Dim:** 988
- **Status:** Paper trading active since Feb 9 22:44

### Trial #250 (Legacy)
- **Backtest Sharpe:** 0.1803
- **Live Sharpe:** -9.55 (FAILING)
- **Status:** Degraded performance, likely to be retired

## Recent Technical Challenges & Solutions

### State Dimension Mismatch (Feb 2026)
**Problem:** Models trained with lookback=10 but database stored lookback=3 → state dimension mismatch on deployment
**Solution:** Auto-detection algorithm in paper_trader_alpaca_polling.py (lines 362-386) reads model checkpoint input dimension and calculates correct lookback

### Database-Model Parameter Discrepancy
**Problem:** Optuna trial params didn't always reflect actual training hyperparameters (especially with --force-ft flag)
**Solution:** Manual trial parameter correction, improved parameter logging in training scripts

## Hyperparameter Search Space (Optuna)

- **Agent Type:** DDQN, A2C, PPO
- **Learning Rate:** 1e-5 to 1e-3 (log scale)
- **Batch Size:** 64, 128, 256, 512
- **Gamma (discount):** 0.95 to 0.9999
- **Network Hidden Sizes:** [128,128] to [2048,1408,1408]
- **Lookback:** 1 to 20 bars
- **Reward Type:** returns, log_returns, sharpe
- **FT-Transformer:** use_ft_encoder, frozen_encoder flags
- **Training Episodes:** 50-500
- **Target Update Period:** 50-500 steps

## Performance Metrics

- **Primary:** Annualized Sharpe Ratio
- **Secondary:** Total return, max drawdown, win rate
- **Backtest Period:** 5 days of 1h bars (~120 bars)
- **Success Criteria:** Sharpe > 0.15 considered good, > 0.20 excellent

## Trading Tickers (Alpaca Crypto)
AAVE/USD, AVAX/USD, BTC/USD, LINK/USD, ETH/USD, LTC/USD, UNI/USD

## Automation Features

1. **Continuous Training:** auto_train.py runs indefinite Optuna studies
2. **Crash Recovery:** watchdog_training.py restarts failed processes
3. **Auto-Deployment:** Top models automatically deployed to paper trading
4. **Performance Monitoring:** Alerts on degradation, auto-stop on emergency threshold
5. **Model Registry:** SQLite tracking of all trained models with metadata

## Integration Points

- **Data Source:** Alpaca Markets API (free tier, crypto data)
- **Model Storage:** Local filesystem (train_results/)
- **Monitoring:** Discord webhooks, terminal dashboard, CSV logs
- **Optional:** Tiburtina AI for market sentiment (not currently active)

## Current Goals

1. ✓ Complete 500-trial training campaign (494/500 done)
2. ✓ Deploy top 2 models for comparison (Trial #91, #100)
3. ⏳ Evaluate live performance over 7-14 days
4. ⏳ Retire underperforming models (Trial #250)
5. 🔄 Continuous improvement via ongoing training

## System Requirements

- Python 3.8+
- PyTorch, NumPy, Pandas, Optuna
- Alpaca API credentials (paper trading or live)
- ~10GB disk space for models and databases
- GPU optional but recommended for training

## Notes for LLM Assistants

- **State dimensions are critical** - mismatch will cause model loading failures
- **Lookback parameter** must be consistent between training and deployment
- **CSV files** in paper_trades/ are the source of truth for live performance
- **Optuna databases** contain all trial metadata, use for model selection
- **Auto-detection logic** (lines 362-386 in paper_trader_alpaca_polling.py) is the permanent fix for lookback mismatches
- **Performance monitors** need 10+ bars before calculating Sharpe ratio
