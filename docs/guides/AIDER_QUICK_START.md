# Aider Quick Start - Reliable Local Coding Assistant

## Why Aider Instead of Continue?

Your Continue extension has critical bugs:
- ❌ Deletes entire files when editing
- ❌ Websocket disconnection issues
- ❌ No responses from Ollama
- ❌ Old version with broken MeiliSearch

Aider is:
- ✅ Stable and reliable
- ✅ Works perfectly with Ollama
- ✅ Won't delete your files
- ✅ Command-line based (works anywhere)

## Quick Commands

### Start Aider in Current Directory
```bash
cd /opt/user-data/experiment/cappuccino
aider
```

### Ask Aider to Edit a File
```bash
aider 1_optimize_unified.py
```
Then ask: "Add more detailed comments to the main training loop"

### Ask About Multiple Files
```bash
aider dashboard.py constants.py environment_Alpaca.py
```

### Ask Questions Without Editing
```bash
aider --read-only
```
Then ask: "Explain how the ensemble voting works"

## Example Session

```bash
$ cd /opt/user-data/experiment/cappuccino
$ aider 1_optimize_unified.py

# Aider starts, you can now type:
> Add error handling for GPU OOM errors in the training loop

# Aider will:
# 1. Analyze the file
# 2. Show you the proposed changes
# 3. Ask if you want to apply them
# 4. Apply changes safely

# To exit:
> /exit
```

## Useful Aider Commands

**In an Aider session:**
- `/add <file>` - Add another file to context
- `/drop <file>` - Remove file from context
- `/ls` - List files in context
- `/diff` - Show pending changes
- `/undo` - Undo last change
- `/exit` - Exit Aider
- `/help` - Show all commands

## Configuration

Aider is already configured at `~/.aider.conf.yml`:
- Model: qwen2.5-coder:7b (Ollama)
- Auto-commits: Disabled (you control git)
- Shows diffs before applying changes
- Pretty output with syntax highlighting

## Safety Features

1. **Shows diffs first** - You see changes before they're applied
2. **No auto-commits** - You control when to commit
3. **Undo available** - Can revert changes with `/undo`
4. **Read-only mode** - Use `--read-only` to just ask questions

## Common Tasks

### Fix a Bug
```bash
aider paper_trader_alpaca_polling.py
> Fix the concentration limit calculation to account for existing positions
```

### Refactor Code
```bash
aider function_CPCV.py
> Refactor this to use more descriptive variable names
```

### Add Feature
```bash
aider environment_Alpaca.py
> Add a method to calculate portfolio Sharpe ratio
```

### Understand Code
```bash
aider --read-only
> /add drl_agents/agents/AgentPPO.py
> Explain how the PPO algorithm updates the policy network
```

## Performance

- Qwen2.5-Coder 7B runs fast on CPU
- Responses typically in 5-15 seconds
- Much faster than GPT-4/Claude API calls
- No API costs!

## Tips

1. **Be specific** - "Add error handling for GPU OOM" is better than "improve this"
2. **One file at a time** - Easier for the model to understand
3. **Use /diff** - Always review changes before accepting
4. **Save often** - Aider won't auto-commit, so git commit regularly
5. **Read-only for questions** - Use `--read-only` when just exploring

## Troubleshooting

**Aider slow or hanging:**
```bash
# Check Ollama is running
pgrep -f ollama

# Restart Ollama if needed
sudo systemctl restart ollama
```

**Model not found:**
```bash
# Verify model exists
ollama list | grep qwen2.5-coder
```

**Wrong model being used:**
```bash
# Specify model explicitly
aider --model ollama/qwen2.5-coder:7b
```

## Alternative: Use a Different Model

If qwen2.5-coder is slow or not working well:

```bash
# Use CodeLlama (also good for code)
aider --model ollama/codellama:7b-instruct

# Use Mistral (general purpose)
aider --model ollama/mistral
```

## Next Steps

1. Try the basic example above
2. Start with read-only mode to get comfortable
3. Then try editing a simple file
4. Gradually use it for more complex tasks

**Aider is your safe, reliable coding assistant - no more file deletions!**
