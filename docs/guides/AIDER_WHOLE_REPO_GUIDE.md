# Aider Whole Repository Guide

## Working with the Entire Cappuccino Codebase

Aider can understand your entire repository without manually adding each file!

## Quick Start Commands

### Start Aider in Whole-Repo Mode
```bash
cd /opt/user-data/experiment/cappuccino
aider
```

Now Aider has access to the **repository map** - it can see all your files!

### Ask Questions About Any File
```bash
aider
> Where is the paper trading logic implemented?
> How does the ensemble voting system work?
> What files handle the training pipeline?
```

Aider will search the repo map and automatically add relevant files to context!

### Add Specific Files or Patterns
```bash
aider
> /add *.py                    # Add all Python files (careful - lots of files!)
> /add drl_agents/**/*.py      # Add all files in drl_agents/
> /add *_trader*.py            # Add all trader files
> /add 1_optimize_unified.py dashboard.py constants.py
```

### Smart Context - Let Aider Find Files
```bash
aider
> /ask What files implement the PPO algorithm?

# Aider will search and tell you, then you can add them:
> /add drl_agents/agents/AgentPPO.py drl_agents/elegantrl_models.py
```

## Common Workflows

### 1. Fix a Bug Across Multiple Files
```bash
aider
> The concentration limit isn't working correctly. Find all files that handle
  position concentration and fix the calculation.

# Aider will:
# 1. Search the repo map for relevant files
# 2. Add them to context automatically
# 3. Propose fixes across all files
# 4. Show you diffs before applying
```

### 2. Refactor Related Code
```bash
aider
> /add environment_Alpaca.py environment_Alpaca_phase2.py
> Consolidate these two environment files - they have duplicate code
```

### 3. Understand System Architecture
```bash
aider --read-only
> Explain the flow from data download to model training to paper trading
> What's the difference between the arena and ensemble systems?
```

### 4. Add Feature Across Multiple Components
```bash
aider
> Add a stop-loss feature that:
  1. Tracks max drawdown in environment_Alpaca.py
  2. Adds stop-loss config to constants.py
  3. Implements logic in paper_trader_alpaca_polling.py
  4. Shows stop-loss status in dashboard.py
```

Aider will automatically add all 4 files and coordinate changes!

## Useful Aider Commands

### File Management
```bash
/add <file>              # Add file to context
/drop <file>             # Remove file from context
/ls                      # List files in current context
/clear                   # Clear all files from context

# Wildcards
/add *.py                # Add all .py files in current dir
/add **/*.py             # Add all .py files recursively
/add drl_agents/*.py     # Add all .py in drl_agents/
```

### Code Search
```bash
/ask <question>          # Ask about code without editing
/search <term>           # Search for term in repo
/help                    # Show all commands
```

### Changes & History
```bash
/diff                    # Show pending changes
/undo                    # Undo last change
/commit                  # Commit current changes
/tokens                  # Show token usage
```

### Repository Map
```bash
/map                     # Show repository structure
/map-refresh             # Refresh the repo map
```

## Smart Features

### Auto File Discovery
Aider automatically finds relevant files when you ask questions:

```bash
> How does the CPCV cross-validation work?
# Aider searches, finds function_CPCV.py, and explains it

> Fix the fee calculation bug
# Aider finds environment_Alpaca.py, constants.py, analyzes them
```

### Multi-File Editing
```bash
# Aider can edit multiple files in one request:
> Move the TRADING constants from constants.py into a separate
  config/trading.py file and update all imports

# Aider will:
# 1. Create config/trading.py
# 2. Move constants
# 3. Update imports in all files that use them
# 4. Show you all changes before applying
```

### Intelligent Context
Aider uses a "repository map" to understand your codebase:
- Sees all file names
- Sees all function/class definitions
- Understands relationships between files
- Adds relevant files automatically

## Configuration

### .aiderignore File
Controls what Aider can see:
```bash
# Edit to customize what's ignored
nano .aiderignore
```

Currently ignoring:
- `archive/` - Old scripts
- `*.db`, `*.log`, `*.pth` - Data/model files
- `train_results/`, `deployments/` - Large directories
- Most `.md` files - Documentation
- Test scripts - `test_*.py`

### ~/.aider.conf.yml
Your global Aider settings:
- Model: qwen2.5-coder:7b
- Shows repository map by default
- No auto-commits (you control git)
- Pretty output with diffs

## Performance Tips

### Start Small, Expand as Needed
```bash
aider
> /add constants.py environment_Alpaca.py
> Explain how initial capital is configured

# If you need more context:
> /add paper_trader_alpaca_polling.py
> Now show me how it's used in paper trading
```

### Use Read-Only for Questions
```bash
aider --read-only
> What's the architecture of this system?
> How does training optimization work?
```
Faster responses, no accidental edits.

### Token Management
```bash
# Check how much context you're using
> /tokens

# If hitting limits, drop files:
> /drop old_file.py
```

Qwen2.5-Coder 7B has ~4K token context, but Aider uses repository map efficiently.

## Example Sessions

### Example 1: Investigate Paper Trading Issue
```bash
$ aider
> The paper trader isn't respecting position limits. Find and fix this.

# Aider searches repo map, finds:
# - paper_trader_alpaca_polling.py
# - environment_Alpaca.py
# - constants.py
# Adds them automatically, proposes fix, shows diff

> Apply the changes
# Changes applied to all 3 files

> /commit
# Creates git commit with descriptive message
```

### Example 2: Add New Feature
```bash
$ aider
> Add a portfolio risk metric that calculates:
  1. Value at Risk (VaR)
  2. Conditional VaR (CVaR)

  Add this to environment_Alpaca.py and show it in dashboard.py

# Aider will:
# - Add both files to context
# - Implement VaR/CVaR calculations
# - Add dashboard display
# - Show you all changes
# - Wait for your approval
```

### Example 3: Understand System
```bash
$ aider --read-only
> Explain the full flow: data → training → deployment → trading

# Aider will search and explain:
# - 0_dl_trainval_data.py downloads data
# - 1_optimize_unified.py trains with Optuna
# - auto_model_deployer.py deploys best models
# - paper_trader_alpaca_polling.py trades live
# - dashboard.py monitors everything
```

## Advanced: Multi-File Refactoring

```bash
aider
> The codebase has duplicate code for fee calculations in environment_Alpaca.py
  and paper_trader_alpaca_polling.py. Extract this into a shared utility
  function in a new utils/fee_calculator.py file and update all callers.

# Aider will:
# 1. Create utils/fee_calculator.py
# 2. Extract shared logic
# 3. Update environment_Alpaca.py to use it
# 4. Update paper_trader_alpaca_polling.py to use it
# 5. Show you diffs for all 3 files
# 6. Wait for approval before applying
```

## Safety Features

1. **Always shows diffs** - You see every change before it's applied
2. **/undo command** - Revert last change instantly
3. **No auto-commits** - You control when changes are committed
4. **Repository map** - Aider understands context, won't make random changes
5. **.aiderignore** - Protects data files, logs, models from being edited

## Tips for Best Results

### Be Specific
✅ Good: "Add error handling for GPU OOM in the training loop of 1_optimize_unified.py"
❌ Bad: "Make it better"

### Let Aider Search
✅ Good: "Find where we calculate Sharpe ratio and optimize it"
❌ Bad: "Edit this file" (without context)

### Review Diffs
Always check `/diff` before accepting changes:
```bash
> /diff
# Review changes carefully
> Apply the changes  # Only if diff looks good
```

### Use /ask First
```bash
> /ask How does the ensemble voting work?
# Understand the code first
> Now modify the voting to weight by recent Sharpe ratio
# Make informed changes
```

## Getting Started

**Try this now:**
```bash
cd /opt/user-data/experiment/cappuccino
aider

# In Aider:
> /map
> /ask What are the core Python files in this project?
> /add 1_optimize_unified.py
> Explain the main training loop
```

You're now working with the whole repository efficiently!

## Troubleshooting

**Aider is slow:**
- Use `/tokens` to check context size
- Drop unnecessary files with `/drop`
- Use `--read-only` for questions

**Can't find a file:**
```bash
> /map                    # See repository structure
> /map-refresh           # Rebuild map if needed
```

**Made a mistake:**
```bash
> /undo                  # Revert last change
> /clear                 # Clear all pending changes
```

**Want to start fresh:**
```bash
> /exit
$ rm .aider.chat.history.md
$ aider
```

## Summary

You can now:
- ✅ Work with entire repository without adding each file
- ✅ Let Aider auto-discover relevant files
- ✅ Ask questions about any part of the codebase
- ✅ Make coordinated changes across multiple files
- ✅ Safely review all changes before applying

**Start with:** `aider` (no files specified)
**Then ask:** Questions about your codebase
**Aider will:** Find and add relevant files automatically!
