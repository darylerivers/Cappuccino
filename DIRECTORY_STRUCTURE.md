# Cappuccino Directory Structure

**Last Updated:** February 2026
**Structure:** `scripts/` organized tree with core modules at root

---

## Overview

Cappuccino uses a hybrid structure that balances organization with ease of imports:
- **Core modules** (environment, agents, config) remain at root for simple imports
- **Utility scripts** are organized by purpose under `scripts/`
- **Supporting modules** are grouped by function (`utils/`, `processors/`, etc.)

---

## Root Level

Core system files that are imported frequently:

```
├── environment_Alpaca.py          # Main RL environment for Alpaca trading
├── config_main.py                 # Primary configuration (imports from constants)
├── constants.py                   # Centralized constants (canonical source)
├── 2_validate.py                  # Study validation and analysis
├── paper_trading_failsafe.sh      # Auto-restart wrapper for paper trading
└── drl_agents/                    # Deep RL agent implementations
    ├── agents/                    # Individual agent classes (PPO, DDPG, etc.)
    ├── elegantrl_models.py        # Model factory and training interface
    └── ...
```

---

## \`scripts/\` - Organized Utilities

### \`scripts/data/\`
Data download and preprocessing scripts:
```
├── 0_dl_trainval_data.py          # Download training/validation data
├── 0_dl_fred_data.py              # Download FRED economic data
└── 0_dl_trade_data_chunked.py    # Download trade data in chunks
```

### \`scripts/training/\`
Training and optimization workflows:
```
├── 1_optimize_unified.py          # Main Optuna CPCV training pipeline
├── rerun_best_trial_v2.py         # Re-run specific trial from study
└── ...
```

### \`scripts/deployment/\`
Paper trading and live deployment:
```
├── paper_trader_alpaca_polling.py # Main paper trading bot (polling mode)
└── ...
```

### \`scripts/optimization/\`
Backtesting and performance analysis:
```
├── 4_backtest.py                  # Backtest trained models
├── 5_pbo.py                       # Probability of Backtest Overfitting
└── ...
```

### \`scripts/automation/\`
System automation and control:
```
├── start_training.sh              # Launch training workers
├── start_automation.sh            # Start full automation suite
├── stop_automation.sh             # Stop all automation
├── status_automation.sh           # Check automation status
├── training_control.sh            # Dynamic worker scaling
└── ...
```

---

## Supporting Modules

### \`utils/\`
Reusable utility functions:
```
├── function_CPCV.py               # Combinatorial Purged K-Fold CV
├── function_train_test.py         # Training and testing functions
├── function_finance_metrics.py    # Financial performance metrics
├── fee_tier_manager.py            # Dynamic fee tier tracking
└── timeframe_constraint.py        # Trading hour constraints
```

### \`processors/\`
Data processing pipelines:
```
├── preprocessor.py                # Base data preprocessing
├── processor_Alpaca.py            # Alpaca-specific processing
├── processor_MultiAsset.py        # Multi-asset data handling
├── processor_FRED.py              # FRED economic data processing
└── processor_CGE.py               # CGE sentiment data
```

### \`models/\`
Trained models and ensemble systems:
```
├── adaptive_ensemble_agent.py     # Game-theory voting ensemble
└── ...
```

### \`config/\`
Configuration files:
```
├── discord.py                     # Discord bot configuration
├── pipeline_config.json           # Pipeline settings
└── pipeline_v2_config.json        # V2 pipeline settings
```

---

## Data Directories

### \`data/\`
Training and validation data (gitignored):
```
├── 1h_1680/                       # 1-hour candles, 1680-hour lookback
└── ...
```

### \`train_results/\`
Model checkpoints and studies (gitignored):
```
├── ensemble/                      # Top 10 models for ensemble
├── adaptive_ensemble/             # Adaptive ensemble state
├── cwd_tests/                     # Individual trial results
│   └── trial_XXXX_1h/            # Per-trial model checkpoints
└── ...
```

### \`paper_trades/\`
Paper trading logs and state (gitignored):
```
├── alpaca_session.csv             # Trade execution log
├── ensemble_votes.json            # Voting record
└── profit_protection.log          # Risk management events
```

---

## Monitoring & Infrastructure

### \`monitoring/\`
System health and performance:
```
├── check_traders_status.sh        # Check paper trader health
├── check_trial_complete.sh        # Monitor training progress
└── ...
```

### \`infrastructure/\`
Deployment and integration guides:
```
├── amd_migration/                 # AMD GPU migration scripts
├── tiburtina_integration/         # Tiburtina sentiment integration
└── local_coding_bot/              # Local AI assistant setup
```

### \`logs/\`
System logs (gitignored):
```
├── paper_trading_live.log         # Current paper trading output
├── paper_trading_failsafe.log     # Failsafe wrapper log
├── training_worker_*.log          # Per-worker training logs
└── ...
```

---

## Archive

### \`archive/\`
Deprecated scripts and old documentation:
```
├── deprecated_scripts/            # Old training/data scripts
├── training_variants/             # Historical training approaches
├── test_scripts/                  # One-off test scripts
└── old_documentation/             # Outdated guides
```

---

## Key Files Reference

### Training Workflow
1. Download data: \`scripts/data/0_dl_trainval_data.py\`
2. Optimize: \`scripts/training/1_optimize_unified.py\`
3. Backtest: \`scripts/optimization/4_backtest.py\`
4. Deploy: \`scripts/deployment/paper_trader_alpaca_polling.py\`

### Automation
- Start: \`scripts/automation/start_automation.sh\`
- Stop: \`scripts/automation/stop_automation.sh\`
- Status: \`scripts/automation/status_automation.sh\`

### Configuration
- Data constants: \`constants.py\` (canonical source)
- Training config: \`config_main.py\` (computed values)
- Discord: \`config/discord.py\`

---

## Import Patterns

### Core Modules (at root)
\`\`\`python
from environment_Alpaca import CryptoEnvAlpaca
from config_main import TICKER_LIST, ALPACA_LIMITS
from constants import RISK, TRADING, NORMALIZATION
from drl_agents.elegantrl_models import get_model
\`\`\`

### Utils and Processors
\`\`\`python
from utils.function_CPCV import CombPurgedKFoldCV
from utils.function_train_test import train_and_test
from processors.processor_Alpaca import AlpacaProcessor
\`\`\`

### Models
\`\`\`python
from models.adaptive_ensemble_agent import AdaptiveEnsembleAgent
\`\`\`

### Scripts (when running from root)
\`\`\`bash
python scripts/training/1_optimize_unified.py
python scripts/deployment/paper_trader_alpaca_polling.py
\`\`\`

---

## Notes

- **Why not a full package?** The \`scripts/\` tree provides organization without the overhead of package imports. Core modules stay at root for convenience.
- **Archive vs Deletion:** Scripts are archived rather than deleted to preserve historical context and enable rollback if needed.
- **Config Hierarchy:** \`constants.py\` is the canonical source; \`config_main.py\` computes derived values and references constants.
- **Import Hygiene:** Always import from canonical locations. Use \`from utils.X import Y\` not \`from X import Y\`.

---

## Migration Notes (February 2026)

The codebase was reorganized from a flat root structure to this organized tree:
- Old flat files → \`scripts/\` organized by purpose
- Duplicate configs → Single \`constants.py\` source of truth
- \`Cappuccino_V-0.1/\` stale package → Deleted
- All shell scripts updated to reference \`scripts/\` paths
- All Python imports updated to use \`utils/\`, \`processors/\`, etc.
