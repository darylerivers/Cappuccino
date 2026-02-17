# Fix Continue Extension (Optional)

## Current Problem

Your Continue extension is a **very old version** with critical bugs:
- Deletes entire files when editing
- Websocket disconnection issues
- Broken MeiliSearch integration
- No Ollama responses

## Option 1: Uninstall Continue (Recommended)

Just use Aider instead - it's more reliable:
```bash
# Uninstall Continue extension
code --uninstall-extension continue.continue

# Remove config
rm -rf ~/.continue/

# Use Aider instead (see AIDER_QUICK_START.md)
```

## Option 2: Upgrade Continue to Modern Version

If you really want to keep using Continue in VSCode:

### Step 1: Uninstall Old Version
```bash
code --uninstall-extension continue.continue
rm -rf ~/.continue/
```

### Step 2: Install Modern Continue
```bash
# Install latest Continue extension
code --install-extension continue.continue

# Wait for installation to complete
```

### Step 3: Create Modern Config (JSON Format)

Modern Continue uses `~/.continue/config.json`:

```bash
mkdir -p ~/.continue
cat > ~/.continue/config.json << 'EOF'
{
  "models": [
    {
      "title": "Qwen2.5-Coder-7B",
      "provider": "ollama",
      "model": "qwen2.5-coder:7b",
      "apiBase": "http://localhost:11434"
    },
    {
      "title": "CodeLlama-7B",
      "provider": "ollama",
      "model": "codellama:7b-instruct",
      "apiBase": "http://localhost:11434"
    }
  ],
  "tabAutocompleteModel": {
    "title": "Qwen2.5-Coder",
    "provider": "ollama",
    "model": "qwen2.5-coder:7b",
    "apiBase": "http://localhost:11434"
  },
  "embeddingsProvider": {
    "provider": "ollama",
    "model": "nomic-embed-text",
    "apiBase": "http://localhost:11434"
  },
  "allowAnonymousTelemetry": false,
  "disableIndexing": true
}
EOF
```

### Step 4: Download Embedding Model (Optional)
```bash
# Only needed if you want code search
ollama pull nomic-embed-text
```

### Step 5: Restart VSCode
```bash
# Close all VSCode windows
killall code

# Start VSCode
code /opt/user-data/experiment/cappuccino
```

### Step 6: Test Continue
1. Open any Python file
2. Press `Ctrl+L` (or Cmd+L on Mac)
3. Type: "Explain this file"
4. Should get a response from Qwen2.5-Coder

## Modern Continue Features

If you upgrade to modern Continue:
- ✅ JSON config (easier to edit)
- ✅ Better Ollama integration
- ✅ Tab autocomplete
- ✅ Inline editing (won't delete files)
- ✅ Code search (optional)
- ✅ More stable overall

## Our Recommendation

**Use Aider for now:**
- More reliable
- Better for actual code editing
- Won't delete your files
- Easier to control what changes

**Use Continue later:**
- Good for quick questions
- Tab autocomplete in editor
- When modern version is stable

## If You Keep Getting Issues

Continue is still in active development and can be buggy. Aider is battle-tested and stable for production use. For critical work on your trading system, Aider is the safer choice.

See: AIDER_QUICK_START.md
