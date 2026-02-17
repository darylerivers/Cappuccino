# Autonomous AI Advisor

Self-improving hyperparameter optimization system powered by local Ollama models.

## What It Does

The autonomous advisor runs continuously in the background and:

1. **Monitors** your training progress (checks every 5 minutes by default)
2. **Analyzes** results when enough new trials complete (every 50 trials)
3. **Generates** new parameter configurations using AI insights
4. **Tests** its discoveries automatically when GPU is free
5. **Learns** from results and tracks best discoveries

This creates a **self-improving loop** where the AI continuously optimizes your training.

## Quick Start

### Start the Advisor

```bash
# Start with defaults (monitors current study)
./start_autonomous_advisor.sh

# Or specify study
./start_autonomous_advisor.sh cappuccino_3workers_20251102_2325

# Custom intervals
./start_autonomous_advisor.sh \
  cappuccino_3workers_20251102_2325 \
  qwen2.5-coder:7b \
  50 \   # Analyze every 50 trials
  300 \  # Check every 300 seconds (5 min)
  10     # Run 10 trials per test config
```

### Check Status

```bash
./status_autonomous_advisor.sh
```

Shows:
- Running status and PID
- Current state (trials analyzed, configs tested)
- Best discovered value
- Recent activity

### Stop the Advisor

```bash
./stop_autonomous_advisor.sh
```

Gracefully shuts down (waits for current operations to complete).

## How It Works

### The Autonomous Loop

```
┌─────────────────────────────────────────────────────────────┐
│                    AUTONOMOUS AI LOOP                        │
└─────────────────────────────────────────────────────────────┘

Every 5 minutes:
  │
  ├─> Check: How many new trials since last check?
  │
  ├─> If >= 50 new trials:
  │     │
  │     ├─> Run AI analysis (Ollama analyzes all trials)
  │     │     └─> Identifies impactful parameters
  │     │
  │     ├─> Generate 3 new configurations
  │     │     └─> AI suggests promising combinations
  │     │
  │     └─> If GPU is idle (< 20% utilization):
  │           │
  │           ├─> Test each config (10 trials each)
  │           │
  │           └─> Track results
  │                 └─> If new best found: 🎯 Log it!
  │
  └─> Save state and repeat
```

### Smart Testing

The advisor only tests when:
- No training processes are running (`1_optimize_unified.py`)
- GPU utilization is < 20%
- Analysis has generated suggestions

This means it **won't interfere** with your main training runs.

### State Persistence

All state is saved to `analysis_reports/advisor_state.json`:
- Last trial count
- Number of analyses performed
- All tested configurations and their results
- Best discovered value

If you restart the advisor, it picks up where it left off.

## Configuration Options

### Analysis Interval

```bash
--analysis-interval 50  # Run analysis every 50 new trials
```

**Recommendations:**
- `25-50`: Frequent analysis, good for exploration
- `50-100`: Balanced (default)
- `100+`: Less frequent, for mature studies

### Check Interval

```bash
--check-interval 300  # Check every 5 minutes
```

**Recommendations:**
- `180` (3 min): Responsive, more overhead
- `300` (5 min): Balanced (default)
- `600` (10 min): Less overhead

### Test Trials

```bash
--max-test-trials 10  # Run 10 trials per config
```

**Recommendations:**
- `5`: Quick validation
- `10`: Good balance (default)
- `20+`: More reliable results, takes longer

### Disable Auto-Testing

```bash
--no-auto-test  # Only analyze, don't test
```

Use this if you want the advisor to only analyze and generate suggestions without automatically testing them.

## Usage Scenarios

### Scenario 1: Overnight Self-Improvement

```bash
# Before bed: Start main training + advisor
./train_alpaca_model.sh 200 3
./start_autonomous_advisor.sh

# The advisor will:
# - Monitor the 3 training workers
# - Analyze every 50 trials
# - When training completes, test AI suggestions
# - You wake up to analysis reports + tested configs
```

### Scenario 2: Continuous Optimization

```bash
# Long-running exploration
./start_autonomous_advisor.sh \
  cappuccino_3workers_20251102_2325 \
  qwen2.5-coder:7b \
  30 \   # Analyze frequently
  180 \  # Check every 3 minutes
  15     # Thorough testing

# Let it run for days/weeks
# It will continuously improve your hyperparameters
```

### Scenario 3: Analysis Only

```bash
# Just want insights, not auto-testing
python ollama_autonomous_advisor.py \
  --study cappuccino_3workers_20251102_2325 \
  --analysis-interval 100 \
  --no-auto-test \
  --daemon
```

## Monitoring

### Watch Live Activity

```bash
# Follow the log
tail -f logs/autonomous_advisor.log

# Or console output
tail -f logs/autonomous_advisor_console.log
```

### Check State

```bash
# View current state
cat analysis_reports/advisor_state.json | jq '.'

# See tested configurations
cat analysis_reports/advisor_state.json | jq '.tested_configs'

# Check best discovered value
cat analysis_reports/advisor_state.json | jq '.best_discovered_value'
```

### Generated Reports

The advisor creates timestamped reports:

```
analysis_reports/
├── advisor_state.json                      # Current state
├── ollama_analysis_*_TIMESTAMP.txt         # Analysis reports
└── ollama_suggestions_*_TIMESTAMP.json     # Generated configs
```

## What Gets Logged

### Log Levels

- **INFO**: Normal operations (checks, analysis start/complete)
- **SUCCESS**: New best values discovered
- **WARNING**: Timeouts, GPU busy
- **ERROR**: Failed operations

### Example Log

```
[2025-11-08 15:45:00] [INFO] Check: 785 total trials (53 new since last check)
[2025-11-08 15:45:00] [INFO] Threshold reached (53 >= 50)
[2025-11-08 15:45:00] [INFO] Running AI analysis on cappuccino_3workers_20251102_2325...
[2025-11-08 15:46:30] [INFO] ✓ Analysis completed successfully
[2025-11-08 15:46:30] [INFO] Generating 3 new configurations...
[2025-11-08 15:47:15] [INFO] ✓ Generated 3 configurations
[2025-11-08 15:47:15] [INFO] GPU is idle. Testing AI-suggested configurations...
[2025-11-08 15:47:15] [INFO] Testing configuration 1/10...
[2025-11-08 15:47:15] [INFO]   Rationale: Explores the high correlation of lr_schedule_factor
[2025-11-08 15:47:15] [INFO]   Testing: learning_rate=0.0001, batch_size=256, gamma=0.99...
[2025-11-08 16:02:45] [INFO] ✓ Test completed. Result: 0.082345
[2025-11-08 16:02:45] [SUCCESS] 🎯 NEW BEST discovered by AI: 0.082345!
```

## Performance Impact

### Resource Usage

- **CPU**: Minimal during monitoring (~0.1%)
- **CPU**: ~100-200% during analysis (2 minutes every 50 trials)
- **Memory**: ~200MB base + model size during analysis
- **GPU**: Only used when idle and testing

### Timing

- **Check cycle**: < 1 second
- **Analysis**: 30-90 seconds
- **Config generation**: 30-90 seconds
- **Testing (per config)**: 10-30 minutes (depends on max_test_trials)

## Integration with Existing Workflow

The advisor works alongside your existing training:

```bash
# Terminal 1: Main training (3 workers)
./train_alpaca_model.sh 300 3

# Terminal 2: Paper trading
python paper_trader_alpaca_polling.py ...

# Terminal 3: Autonomous advisor
./start_autonomous_advisor.sh

# All three run simultaneously!
```

The advisor is smart enough to:
- Not interfere with active training
- Wait for GPU to be free
- Test during idle periods

## Advanced Usage

### One-Time Analysis

```bash
# Just run one cycle (don't loop)
python ollama_autonomous_advisor.py \
  --study cappuccino_3workers_20251102_2325 \
  --once
```

### Custom Model

```bash
./start_autonomous_advisor.sh \
  cappuccino_3workers_20251102_2325 \
  mistral:latest  # Use Mistral instead of Qwen
```

### Multiple Studies

You can run separate advisors for different studies:

```bash
# Terminal 1
python ollama_autonomous_advisor.py \
  --study cappuccino_3workers_20251102_2325 \
  --daemon &

# Terminal 2
python ollama_autonomous_advisor.py \
  --study cappuccino_alpaca_v2 \
  --daemon &
```

## Troubleshooting

### Advisor Not Testing

**Check:**
1. Is training running? `ps aux | grep 1_optimize_unified`
2. Is GPU busy? `nvidia-smi`
3. Are suggestions generated? `ls analysis_reports/ollama_suggestions*`

**Fix:**
- Wait for training to complete
- Increase check interval
- Check logs for errors

### Analysis Failing

**Check:**
1. Is Ollama running? `ollama list`
2. Is the model available? `ollama list | grep qwen2.5-coder`

**Fix:**
```bash
ollama serve  # Start Ollama
ollama pull qwen2.5-coder:7b  # Pull model
```

### State File Corruption

**Fix:**
```bash
# Backup and reset state
mv analysis_reports/advisor_state.json analysis_reports/advisor_state.json.bak
# Advisor will create new state on next run
```

## Best Practices

1. **Start Small**: Begin with `--analysis-interval 100` and `--max-test-trials 5`
2. **Monitor First Day**: Watch logs to ensure it's working as expected
3. **Check Results Weekly**: Review `tested_configs` to see what AI discovered
4. **Adjust Intervals**: Fine-tune based on your trial completion rate
5. **Track Best Values**: The `best_discovered_value` shows if AI is improving

## Example: Full Setup

```bash
# 1. Start training
./train_alpaca_model.sh 300 3

# 2. Start paper trading
python paper_trader_alpaca_polling.py \
  --model-dir train_results/cwd_tests/trial_141_1h \
  --tickers BTC/USD ETH/USD LTC/USD \
  --timeframe 1h &

# 3. Start autonomous advisor
./start_autonomous_advisor.sh

# 4. Monitor everything
watch -n 30 '
  echo "=== Training ==="
  tail -3 logs/parallel_training/worker_1.log
  echo ""
  echo "=== AI Advisor ==="
  tail -3 logs/autonomous_advisor.log
  echo ""
  echo "=== GPU ==="
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
'
```

## Files Created

```
cappuccino/
├── ollama_autonomous_advisor.py          # Main daemon
├── start_autonomous_advisor.sh           # Start script
├── stop_autonomous_advisor.sh            # Stop script
├── status_autonomous_advisor.sh          # Status checker
├── logs/
│   ├── autonomous_advisor.log            # Structured log
│   ├── autonomous_advisor_console.log    # Console output
│   ├── autonomous_advisor.pid            # Process ID
│   └── ai_test_*.log                     # Test run logs
└── analysis_reports/
    ├── advisor_state.json                # Persistent state
    ├── ollama_analysis_*.txt             # Analysis reports
    └── ollama_suggestions_*.json         # AI suggestions
```

---

**Status**: Production Ready
**Generated**: 2025-11-08
**Dependencies**: Ollama, pandas, numpy, requests
