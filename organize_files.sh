#!/bin/bash
# Organize Cappuccino file structure
# Safe: Does not move actively used files

set -e

echo "Creating organized folder structure..."

# Create main folders
mkdir -p scripts/{training,data,optimization,deployment,automation}
mkdir -p logs/{training,data,archive,system}
mkdir -p docs/{guides,reports,status}
mkdir -p processors
mkdir -p monitoring
mkdir -p models/{checkpoints,exports,arena}
mkdir -p databases
mkdir -p infrastructure/{docker,integration}
mkdir -p tests
mkdir -p utils

echo "✓ Folders created"

# Move training scripts
echo "Moving training scripts..."
mv -n 1_optimize_*.py scripts/training/ 2>/dev/null || true
mv -n two_phase_scheduler.py scripts/training/ 2>/dev/null || true
mv -n continuous_training.py scripts/training/ 2>/dev/null || true
mv -n pipeline*.py scripts/training/ 2>/dev/null || true
mv -n rerun_best_trial*.py scripts/training/ 2>/dev/null || true

# Move data scripts  
echo "Moving data scripts..."
mv -n 0_dl_*.py scripts/data/ 2>/dev/null || true
mv -n update_data_*.py scripts/data/ 2>/dev/null || true
mv -n prepare_*.py scripts/data/ 2>/dev/null || true
mv -n augment_*.py scripts/data/ 2>/dev/null || true

# Move deployment scripts
echo "Moving deployment scripts..."
mv -n auto_model_deployer.py scripts/deployment/ 2>/dev/null || true
mv -n deploy_*.py scripts/deployment/ 2>/dev/null || true
mv -n export_trial_*.py scripts/deployment/ 2>/dev/null || true
mv -n paper_trader_*.py scripts/deployment/ 2>/dev/null || true
mv -n coinbase_live_trader.py scripts/deployment/ 2>/dev/null || true

# Move automation scripts
echo "Moving automation scripts..."
mv -n start_*.sh scripts/automation/ 2>/dev/null || true
mv -n stop_*.sh scripts/automation/ 2>/dev/null || true
mv -n status_*.sh scripts/automation/ 2>/dev/null || true
mv -n verify_*.sh scripts/automation/ 2>/dev/null || true
mv -n watch_*.sh scripts/automation/ 2>/dev/null || true
mv -n watch_*.py scripts/automation/ 2>/dev/null || true
mv -n kill_switch.sh scripts/automation/ 2>/dev/null || true
mv -n training_control.sh scripts/automation/ 2>/dev/null || true

# Move analysis/optimization
echo "Moving analysis scripts..."
mv -n analyze_*.py scripts/optimization/ 2>/dev/null || true
mv -n backtest*.py scripts/optimization/ 2>/dev/null || true
mv -n validate_*.py scripts/optimization/ 2>/dev/null || true
mv -n 4_backtest.py scripts/optimization/ 2>/dev/null || true
mv -n 5_pbo.py scripts/optimization/ 2>/dev/null || true

# Move processors
echo "Moving processors..."
mv -n processor_*.py processors/ 2>/dev/null || true
mv -n preprocessor.py processors/ 2>/dev/null || true

# Move monitoring
echo "Moving monitoring scripts..."
mv -n monitor_*.sh monitoring/ 2>/dev/null || true
mv -n check_*.sh monitoring/ 2>/dev/null || true
mv -n show_*.py monitoring/ 2>/dev/null || true
mv -n performance_*.py monitoring/ 2>/dev/null || true
mv -n system_*.py monitoring/ 2>/dev/null || true
mv -n dashboard.py monitoring/ 2>/dev/null || true
mv -n paper_trading_dashboard.py monitoring/ 2>/dev/null || true

# Move model-related
echo "Moving model files..."
mv -n ensemble_*.py models/ 2>/dev/null || true
mv -n adaptive_*.py models/ 2>/dev/null || true
mv -n simple_ensemble_*.py models/ 2>/dev/null || true
mv -n multi_timeframe_*.py models/ 2>/dev/null || true
mv -n model_*.py models/ 2>/dev/null || true
mv -n arena_*.py models/ 2>/dev/null || true

# Move utilities
echo "Moving utilities..."
mv -n redundancy_*.py utils/ 2>/dev/null || true
mv -n archive_*.py utils/ 2>/dev/null || true
mv -n fee_tier_*.py utils/ 2>/dev/null || true
mv -n timeframe_*.py utils/ 2>/dev/null || true
mv -n trade_history_*.py utils/ 2>/dev/null || true
mv -n alert_*.py utils/ 2>/dev/null || true
mv -n market_analysis.py utils/ 2>/dev/null || true
mv -n portfolio_forecaster.py utils/ 2>/dev/null || true
mv -n tiburtina_integration.py utils/ 2>/dev/null || true
mv -n path_detector.py utils/ 2>/dev/null || true
mv -n code_janitor.py utils/ 2>/dev/null || true

# Move function files
echo "Moving function libraries..."
mv -n function_*.py utils/ 2>/dev/null || true

# Move infrastructure
echo "Moving infrastructure files..."
mv -n Dockerfile infrastructure/docker/ 2>/dev/null || true
mv -n docker-compose.yml infrastructure/docker/ 2>/dev/null || true
mv -n Makefile infrastructure/docker/ 2>/dev/null || true

# Move databases
echo "Moving databases..."
mv -n *.db databases/ 2>/dev/null || true

# Move logs
echo "Moving logs..."
mv -n *.log logs/training/ 2>/dev/null || true
mv -n training_workers.pids logs/training/ 2>/dev/null || true

# Move documentation
echo "Moving documentation..."
mv -n *.md docs/guides/ 2>/dev/null || true
mv -n README* docs/guides/ 2>/dev/null || true

# Move config files
echo "Moving config files..."
mv -n config_*.py . 2>/dev/null || true  # Keep in root
mv -n constants.py . 2>/dev/null || true  # Keep in root

# Move tests
echo "Moving test files..."
mv -n test*.py tests/ 2>/dev/null || true

echo ""
echo "✓ File organization complete!"
echo ""
echo "Summary:"
find scripts -type f | wc -l | xargs echo "  Scripts:"
find logs -type f | wc -l | xargs echo "  Logs:"
find docs -type f | wc -l | xargs echo "  Docs:"
find processors -type f | wc -l | xargs echo "  Processors:"
find monitoring -type f | wc -l | xargs echo "  Monitoring:"
find models -type f | wc -l | xargs echo "  Models:"
find utils -type f | wc -l | xargs echo "  Utils:"
find databases -type f | wc -l | xargs echo "  Databases:"

