# Aider - Simple Single-File Mode

## Basic Usage

Work on **one file at a time** - simple and reliable.

### Ask About a File
```bash
aider <filename> --message "Explain what this does" --no-auto-commits
```

### Make Changes to a File
```bash
aider <filename> --message "Add better comments" --no-auto-commits
```

## Examples

### Understand Code
```bash
aider constants.py --message "Explain all the constants" --no-auto-commits

aider 1_optimize_unified.py --message "What does this training script do?" --no-auto-commits

aider dashboard.py --message "How does this dashboard work?" --no-auto-commits
```

### Make Simple Edits
```bash
aider constants.py --message "Add a comment explaining INITIAL_CAPITAL" --no-auto-commits

aider environment_Alpaca.py --message "Add error handling for GPU OOM" --no-auto-commits

aider paper_trader_alpaca_polling.py --message "Add logging for each trade" --no-auto-commits
```

### Review Changes
```bash
# After Aider makes changes
git diff <filename>

# Commit if good
git add <filename>
git commit -m "Description of change"

# Undo if bad
git checkout <filename>
```

## Multiple Files (If Needed)

```bash
# Specify 2-3 files max
aider file1.py file2.py --message "Make them consistent" --no-auto-commits
```

## That's It!

Just: `aider <file> --message "what you want" --no-auto-commits`

Simple, focused, reliable.
