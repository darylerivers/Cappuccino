# Automated Trading Pipeline

Complete automation of model lifecycle from training to live trading.

## Overview

```
Training (Optuna)
    ↓
Gate 1: Backtesting (Adaptive thresholds)
    ↓
Gate 2: CGE Stress Testing (200 scenarios)
    ↓
Gate 3: Paper Trading (Model Arena, 7+ days)
    ↓
Live Trading (Automatic promotion)
```

## Quick Start

### 1. Start the Pipeline Orchestrator

**Daemon mode (recommended for production):**
```bash
cd /opt/user-data/experiment/cappuccino
python pipeline_orchestrator.py --daemon
```

**Single check (for testing):**
```bash
python pipeline_orchestrator.py --once
```

**Dry-run mode (no actual deployments):**
```bash
python pipeline_orchestrator.py --dry-run --once
```

### 2. Check Pipeline Status

```bash
python pipeline_orchestrator.py --status
```

### 3. Monitor Logs

```bash
tail -f logs/pipeline_orchestrator.log
```

## Configuration

Edit `config/pipeline_config.json` to customize:

- **Check interval**: How often to check for new trials (default: 30 minutes)
- **Gate thresholds**: Criteria for each validation gate
- **Auto-promotion**: Enable/disable automatic promotion to live trading
- **Notifications**: Desktop alerts, log files, email

## Validation Gates

### Gate 1: Backtesting

**Adaptive thresholds based on training progress:**

- **Early (<50 trials)**: Very lenient, max loss -50%
- **Mid (50-200 trials)**: Max loss -20%, Sharpe > -0.5
- **Late (200-500 trials)**: Max loss -10%, Sharpe > 0
- **Mature (500+ trials)**: Must be profitable, Sharpe > 0.3, drawdown < 15%

### Gate 2: CGE Stress Testing

Tests model across 200 diverse economic scenarios:

- ≥40% of scenarios must be profitable
- Median Sharpe > 0
- Max drawdown < 25%
- No catastrophic failures (>-90% loss)

### Gate 3: Paper Trading (Model Arena)

Minimum 7 days of live paper trading evaluation:

- Win rate > 80%
- Sharpe > 0.5
- Max drawdown < 15%
- Positive alpha vs market
- No emergency stops triggered

## Emergency Controls

### Emergency Stop

Create file to halt all pipeline operations:
```bash
touch deployments/pipeline_emergency_stop
```

Remove file to resume:
```bash
rm deployments/pipeline_emergency_stop
```

### Manual Override

Edit `deployments/pipeline_state.json` to manually adjust trial stages.

## File Structure

```
cappuccino/
├── pipeline_orchestrator.py       # Main entry point
├── config/
│   └── pipeline_config.json      # Configuration
├── pipeline/                      # Pipeline modules
│   ├── state_manager.py          # State tracking
│   ├── gates.py                  # Validation logic
│   ├── backtest_runner.py        # Automate backtests
│   ├── cge_runner.py             # Automate stress tests
│   └── notifications.py          # Alerts
├── deployments/
│   ├── pipeline_state.json       # Trial progress tracking
│   └── pipeline_emergency_stop   # Kill switch (optional)
├── stress_test_results/          # CGE test results
│   └── trial_XXX_cge.csv
└── logs/
    ├── pipeline_orchestrator.log # Main log
    └── pipeline_notifications.log# Notification log
```

## Monitoring

### Desktop Notifications

The pipeline sends desktop notifications for key events:

- ✅ Gate passed
- ❌ Gate failed
- 🚀 Model deployed
- ⚠️ Errors

### Log Files

- `logs/pipeline_orchestrator.log` - Main pipeline activity
- `logs/pipeline_notifications.log` - Notification history
- `deployments/pipeline_failures.log` - Error tracking

### Pipeline State

Check current state:
```bash
cat deployments/pipeline_state.json | jq
```

View trials by stage:
```bash
cat deployments/pipeline_state.json | jq '.trials | group_by(.current_stage) | .[] | {stage: .[0].current_stage, count: length}'
```

## Integration with Existing Systems

### Auto Model Deployer

The pipeline integrates with `auto_model_deployer.py`:

- Trials must pass Gates 1 & 2 before paper trading deployment
- Paper trading uses Model Arena for evaluation
- Automatic promotion after 7-day validation

### Performance Grader

The `performance_grader.py` evaluates paper trading performance:

- Grades: A, B, C, D, F
- Only grades A or B qualify for live trading
- Automatic promotion when criteria met

## Troubleshooting

### No trials being processed

1. Check Optuna database has completed trials
2. Verify study name in config matches `.env.training`
3. Check pipeline logs for errors

### Backtest failures

1. Verify model files exist in `train_results/cwd_tests/trial_XXX_1h/`
2. Check data availability in `data/1h_1680/`
3. Review retry count in pipeline state

### CGE stress test failures

1. Verify CGE scenario data exists in `data/cge_synthetic/`
2. Check available disk space for results
3. Increase timeout in `pipeline/cge_runner.py`

### Stuck trials

1. Check `pipeline_state.json` for retry counts
2. Review error messages in stage status
3. Manually mark stage as failed to skip:
   ```bash
   # Edit pipeline_state.json and set stage status to "failed"
   ```

## Advanced Usage

### Custom Thresholds

Edit gate thresholds in `config/pipeline_config.json`:

```json
{
  "gates": {
    "backtest": {
      "thresholds": {
        "mature": {
          "max_loss": -0.05,
          "min_sharpe": 0.5,
          "max_drawdown": 0.10
        }
      }
    }
  }
}
```

### Running as systemd Service

Create `/etc/systemd/system/pipeline-orchestrator.service`:

```ini
[Unit]
Description=Trading Pipeline Orchestrator
After=network.target

[Service]
Type=simple
User=mrc
WorkingDirectory=/opt/user-data/experiment/cappuccino
ExecStart=/usr/bin/python3 pipeline_orchestrator.py --daemon
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable pipeline-orchestrator
sudo systemctl start pipeline-orchestrator
sudo systemctl status pipeline-orchestrator
```

### Integration with Training

Add to training script to trigger pipeline after trial completion:

```python
# At end of training
import subprocess
subprocess.run(['python', 'pipeline_orchestrator.py', '--once'], check=False)
```

## Performance Metrics

After 30 days of operation, expect:

- ✅ 90%+ of new best trials automatically processed
- ✅ <5% false negatives (good models blocked)
- ✅ <1% false positives (bad models pass gates)
- ✅ Zero manual interventions needed
- ✅ Average time training → live: 8-10 days

## Support

For issues or questions:

1. Check logs: `logs/pipeline_orchestrator.log`
2. Review pipeline state: `deployments/pipeline_state.json`
3. Enable debug logging in configuration
4. Review this README for troubleshooting steps
