# Ollama AI Training Advisor Guide

AI-powered training analysis system using locally-run Ollama models to analyze your DRL training results and suggest hyperparameter improvements.

## Overview

Two powerful tools are now available:

1. **`ollama_training_advisor.py`** - Analyzes training results and provides insights
2. **`ollama_param_suggester.py`** - Generates new hyperparameter configurations to try

Both tools use your local Ollama models (no cloud API needed) to analyze 700+ completed trials and provide actionable recommendations.

## Available Ollama Models

Your system has these models installed:

- `mistral:latest` (4.4 GB) - General purpose, good for analysis
- `qwen2.5-coder:7b` (4.7 GB) - **Recommended** - Coding-focused, excellent for technical analysis
- `llama2:7b` (3.8 GB) - Alternative general-purpose model

## Quick Start

### 1. Analyze Training Results

```bash
# Using recommended qwen2.5-coder model
python ollama_training_advisor.py \
  --study cappuccino_3workers_20251102_2325 \
  --model qwen2.5-coder:7b

# Using mistral
python ollama_training_advisor.py \
  --study cappuccino_3workers_20251102_2325 \
  --model mistral:latest
```

**What it does:**
- Loads 700+ completed trials from the database
- Analyzes hyperparameter patterns and correlations
- Identifies which parameters have the biggest impact
- Provides 5 key recommendations:
  1. Key insights about parameter patterns
  2. Potential issues in training
  3. Specific recommendations (3-5 actionable changes)
  4. Search space refinement suggestions
  5. Alternative algorithmic approaches

**Output:** Prints analysis to terminal + saves to `analysis_reports/`

### 2. Generate New Parameter Configurations

```bash
# Generate 3 new configurations
python ollama_param_suggester.py \
  --study cappuccino_3workers_20251102_2325 \
  --generate 3 \
  --model qwen2.5-coder:7b

# Generate 10 configurations
python ollama_param_suggester.py \
  --study cappuccino_3workers_20251102_2325 \
  --generate 10 \
  --model qwen2.5-coder:7b
```

**What it does:**
- Analyzes top 20% performers vs. all trials
- Calculates parameter correlations with performance
- Uses AI to generate new configs that:
  - Exploit regions near best performers
  - Explore promising new parameter combinations
  - Balance exploration vs. exploitation

**Output:**
- Prints configurations with rationale
- Saves to `analysis_reports/ollama_suggestions_*.json`

## Example Output

### Analysis Output

```
KEY INSIGHTS:
- base_break_step and norm_action show significant differences between
  top and bottom performers
- net_dimension, base_target_step, eval_time_gap are also highly impactful
- PPO epochs show slight variation

RECOMMENDATIONS:
1. base_break_step: Explore range around 123000 (top performers cluster here)
2. norm_action: Increase exploration from 17900 to ~30000
3. net_dimension: Try values closer to 1542 (bottom performers' avg)
4. base_target_step: Keep close to 806, explore small variations
5. eval_time_gap: Narrow to 30-90 seconds range

ALGORITHMIC SUGGESTIONS:
- Try A2C algorithm for potentially more stable training
- Implement learning rate decay
- Consider L2 regularization or dropout
```

### Parameter Suggestions Output

```
Configuration 1:
Rationale: High correlation with performance, exploration of new region

  learning_rate: 0.00015
  batch_size: 256
  gamma: 0.995
  ppo_epochs: 8
  net_dimension: 1400
  use_lr_schedule: 0.0
```

## Advanced Usage

### With Web Search (Experimental)

```bash
python ollama_training_advisor.py \
  --study cappuccino_3workers_20251102_2325 \
  --model qwen2.5-coder:7b \
  --web-search
```

Enhances analysis with latest DRL research (requires internet).

### Analyze Different Studies

```bash
# List available studies
sqlite3 databases/optuna_cappuccino.db \
  "SELECT study_name FROM studies"

# Analyze specific study
python ollama_training_advisor.py \
  --study cappuccino_alpaca \
  --model qwen2.5-coder:7b
```

### Custom Ollama Host

```bash
python ollama_training_advisor.py \
  --study cappuccino_3workers_20251102_2325 \
  --ollama-host http://192.168.1.100:11434
```

## Integration with Training

### Workflow Recommendation

1. **Run initial training batch** (100-200 trials)
   ```bash
   ./train_alpaca_model.sh 100 3
   ```

2. **Analyze results with AI**
   ```bash
   python ollama_training_advisor.py \
     --study cappuccino_3workers_20251102_2325
   ```

3. **Generate new configurations**
   ```bash
   python ollama_param_suggester.py \
     --study cappuccino_3workers_20251102_2325 \
     --generate 5
   ```

4. **Review and refine search space** based on AI recommendations

5. **Run focused training** on promising regions

### Continuous Improvement Loop

```bash
# Every 100 trials, analyze and adjust
while true; do
  # Wait for 100 more trials to complete
  sleep 7200  # ~2 hours for 100 trials

  # Get AI analysis
  python ollama_training_advisor.py \
    --study cappuccino_3workers_20251102_2325 \
    --model qwen2.5-coder:7b \
    > analysis_reports/latest_analysis.txt

  # Review and decide on next steps
  cat analysis_reports/latest_analysis.txt
done
```

## Performance

### Processing Time

- **Analysis**: ~30-60 seconds (depends on Ollama model)
- **Suggestions**: ~45-90 seconds (generating 5 configs)

### Resource Usage

- CPU: ~100-200% (single core inference)
- RAM: ~4-6 GB (model loaded in memory)
- No GPU needed (runs on CPU)

## Tips & Best Practices

### 1. Model Selection

- **qwen2.5-coder:7b** - Best for technical hyperparameter analysis
- **mistral:latest** - Good for general insights and explanations
- **llama2:7b** - Fallback option

### 2. Iteration Strategy

- Analyze after every 100-200 trials
- Generate 3-5 new configs at a time (manageable to test)
- Focus on top 3 recommended parameter changes first

### 3. Parameter Interpretation

The AI looks at:
- **Correlation**: How parameter changes affect performance
- **Distribution**: Where top performers cluster
- **Variance**: How sensitive performance is to each parameter

### 4. Combining with Human Insight

AI suggestions are a starting point:
- Review rationale for each suggestion
- Consider domain knowledge (trading dynamics)
- Test suggestions incrementally
- Keep successful patterns

## Troubleshooting

### Ollama Not Running

```bash
# Start Ollama service
ollama serve

# In another terminal, test
ollama list
```

### Model Not Available

```bash
# Pull the recommended model
ollama pull qwen2.5-coder:7b

# Or try another model
ollama pull mistral:latest
```

### No Completed Trials

```bash
# Check study progress
sqlite3 databases/optuna_cappuccino.db \
  "SELECT s.study_name, COUNT(*) as completed
   FROM studies s
   JOIN trials t ON s.study_id = t.study_id
   WHERE t.state = 'COMPLETE'
   GROUP BY s.study_name"
```

### Slow Response

- Ollama inference on CPU takes 30-90 seconds
- Use `--model mistral:latest` for faster results (smaller model)
- Ensure other processes aren't maxing out CPU

## File Structure

```
cappuccino/
├── ollama_training_advisor.py      # Main analysis script
├── ollama_param_suggester.py       # Config generator
├── analysis_reports/               # Generated reports
│   ├── ollama_analysis_*.txt       # Analysis reports
│   └── ollama_suggestions_*.json   # Parameter configs
├── databases/
│   └── optuna_cappuccino.db        # Training database
└── OLLAMA_AI_ADVISOR_GUIDE.md      # This file
```

## Example Session

```bash
# 1. Check current training status
sqlite3 databases/optuna_cappuccino.db \
  "SELECT COUNT(*) FROM trials WHERE study_id=13 AND state='COMPLETE'"
# Output: 719

# 2. Run analysis
python ollama_training_advisor.py \
  --study cappuccino_3workers_20251102_2325 \
  --model qwen2.5-coder:7b

# 3. Read the recommendations
cat analysis_reports/ollama_analysis_*.txt | tail -50

# 4. Generate new configs
python ollama_param_suggester.py \
  --study cappuccino_3workers_20251102_2325 \
  --generate 5

# 5. Review suggested configs
cat analysis_reports/ollama_suggestions_*.json | jq '.'

# 6. Implement top recommendations in your Optuna config
# Edit 1_optimize_unified.py to narrow search ranges
```

## Current Training Study

Your active study: `cappuccino_3workers_20251102_2325`
- **Completed trials**: 719
- **Running trials**: 3 (workers active)
- **Best performance**: 0.071889
- **Worst performance**: -0.090025
- **Mean**: ~0.035

The AI has analyzed all 719 trials and identified the most impactful parameters.

## Next Steps

1. Run the advisor on your current study
2. Review the top 3 recommendations
3. Generate 3-5 new parameter configs
4. Test the most promising suggestions
5. Iterate based on results

## Questions?

The scripts are self-contained and well-documented. Check:
- Script help: `python ollama_training_advisor.py --help`
- Inline code comments in the scripts
- Optuna documentation for parameter meanings

---

**Generated**: 2025-11-08
**Status**: Production Ready
**Dependencies**: Ollama, pandas, numpy, requests, sqlite3
