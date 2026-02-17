# How to Actually Use Aider (Working Methods)

Aider's interactive mode is buggy. Use these approaches instead:

## Method 1: Single-Shot Questions (Most Reliable)

Ask one question at a time without interactive mode:

```bash
# Ask about code
aider --message "Explain how the paper trader works" --no-auto-commits

# Ask to find files
aider --message "What files implement the ensemble voting system?" --no-auto-commits

# Ask for analysis
aider --message "Find any bugs in the fee calculation logic" --no-auto-commits
```

## Method 2: Single-Shot Edits

Make specific edits without interactive mode:

```bash
# Add a file and make a change
aider constants.py --message "Add a comment explaining what INITIAL_CAPITAL is used for" --no-auto-commits

# Edit multiple files
aider environment_Alpaca.py constants.py --message "Increase default initial capital to 1000" --no-auto-commits

# Refactor code
aider function_CPCV.py --message "Add more descriptive variable names to improve readability" --no-auto-commits
```

## Method 3: Use --yes Flag (Semi-Interactive)

Accept all changes automatically:

```bash
aider constants.py --yes
```

Then type your request and it will apply changes without asking.

## Method 4: Use Python API

For more complex tasks:

```python
# Create a file: aider_task.py
from aider.coders import Coder
from aider.models import Model

# Setup
model = Model("ollama/codellama:7b-instruct")
coder = Coder.create(
    main_model=model,
    fnames=["constants.py"],
    auto_commits=False
)

# Ask question
response = coder.run("What is the initial capital setting?")
print(response)

# Make edit
coder.run("Add a comment explaining INITIAL_CAPITAL")
```

Then: `python aider_task.py`

## Method 5: Fix Interactive Mode (Advanced)

If you really want interactive mode:

```bash
# Set environment variable
export OLLAMA_API_BASE=http://localhost:11434

# Use different input method
python -c "import aider; aider.main.main()" --model ollama/codellama:7b-instruct
```

## Best Workflow for Your Project

### For Questions:
```bash
# Quick questions
aider --message "Where is X implemented?" --no-auto-commits

# Detailed analysis
aider --message "Analyze the training pipeline and suggest optimizations" \
      --no-auto-commits > analysis.txt
```

### For Code Changes:
```bash
# Small edits
aider <file> --message "Add error handling for GPU OOM" --no-auto-commits

# Review changes
git diff

# Commit if good
git add <file>
git commit -m "Add GPU OOM error handling"
```

### For Multi-File Changes:
```bash
# Let Aider find the files
aider --message "Fix the concentration limit bug across all relevant files" \
      --no-auto-commits

# Or specify files
aider env*.py constants.py --message "Standardize fee calculations" --no-auto-commits
```

## Common Tasks

### Understand Code
```bash
aider --message "Explain the CPCV cross-validation system" --no-auto-commits
```

### Find Bugs
```bash
aider --message "Analyze paper_trader_alpaca_polling.py for potential bugs" \
      paper_trader_alpaca_polling.py --no-auto-commits
```

### Add Comments
```bash
aider 1_optimize_unified.py --message "Add docstrings to all functions" --no-auto-commits
```

### Refactor
```bash
aider environment_Alpaca.py --message "Extract fee calculation into a helper function" \
      --no-auto-commits
```

### Add Features
```bash
aider dashboard.py --message "Add a new page showing recent trade history" --no-auto-commits
```

## Why Interactive Mode Fails

The issue is:
- Aider loads the repo map
- Tries to start interactive prompt
- Terminal input is broken/not detected
- Exits immediately

Single-shot mode avoids this entirely.

## Recommended Daily Workflow

```bash
# Morning: Check what needs work
aider --message "Review the codebase and suggest improvements" --no-auto-commits > todo.txt

# During work: Make specific changes
aider <file> --message "Fix the bug where..." --no-auto-commits
git diff  # Review
git commit -am "Fix: ..."

# End of day: Analysis
aider --message "Analyze today's changes and suggest tests" --no-auto-commits
```

## Troubleshooting

**Still getting errors?**
```bash
# Disable repo map entirely
aider --no-show-repo-map --message "Your question here"

# Use whole file editing (simpler)
aider --edit-format whole --message "Your question here"

# Maximum compatibility
aider --model ollama/codellama:7b-instruct \
      --no-show-repo-map \
      --no-auto-commits \
      --edit-format whole \
      --message "Your question here"
```

**Model too slow?**
```bash
# Use smaller/faster model
aider --model ollama/mistral --message "Your question here"
```

## Summary

✅ **USE:** Single-shot mode with `--message`
❌ **AVOID:** Interactive mode (buggy)

**Basic pattern:**
```bash
aider [files] --message "what you want" --no-auto-commits
```

This works reliably every time!
