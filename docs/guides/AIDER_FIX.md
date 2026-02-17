# Aider + Ollama Fixes

## Error: "cannot schedule new futures after interpreter shutdown"

This is a bug in Aider's chat history summarization when using Ollama.

### Fix Applied

Updated `~/.aider.conf.yml` to:
- ✅ Disable `cache-prompts`
- ✅ Disable `restore-chat-history`
- ✅ Disable auto-summarization
- ✅ Removed chat history file

### Now Use Aider Like This

**Start fresh session (no history):**
```bash
cd /opt/user-data/experiment/cappuccino
aider
```

**For longer sessions, use --no-auto-commits:**
```bash
aider --no-auto-commits
```

**If you still get errors:**
```bash
# Clear any leftover state
rm -f .aider*

# Restart Ollama
sudo systemctl restart ollama

# Try again
aider --no-stream
```

## Alternative: Use a More Stable Model

If qwen2.5-coder keeps having issues, try:

### Option 1: Use CodeLlama (more stable with Aider)
```bash
aider --model ollama/codellama:7b-instruct
```

### Option 2: Use Mistral (general purpose, very stable)
```bash
aider --model ollama/mistral
```

### Option 3: Set permanent alternative
```bash
# Edit ~/.aider.conf.yml and change:
model: ollama/codellama:7b-instruct
```

## Quick Test

Try this to verify it's working:
```bash
cd /opt/user-data/experiment/cappuccino
aider --no-stream constants.py

# In Aider, type:
> /ls
> What's in this file?
> /exit
```

## If Still Broken

### Nuclear Option: Minimal Aider
```bash
# Completely minimal config
cat > ~/.aider.conf.yml << 'EOF'
model: ollama/codellama:7b-instruct
auto-commits: false
stream: false
show-diffs: true
EOF

# Clear everything
rm -f .aider*

# Try again
aider
```

## Working Configuration

Your current config now has:
- No chat history (prevents summarization errors)
- No caching (prevents asyncio issues)
- Smaller map tokens (less memory)
- Streaming disabled for summarization (can re-enable once working)

## Common Causes

1. **Ollama connection issues** - Restart Ollama
2. **Model context too small** - Use a different model
3. **Chat history corruption** - Deleted with fix above
4. **Python asyncio bug** - Disabled problematic features

## Verify Fix Works

```bash
# Should work now without errors:
aider

# Test in Aider:
> /help
> /map
> /exit
```

If you see output without "cannot schedule new futures" error, it's fixed!
