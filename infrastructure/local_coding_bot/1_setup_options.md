# Local Coding Bot Setup Options

## Overview

You have several excellent options for local AI coding assistants. All run on your hardware with no API costs.

---

## Option 1: Continue.dev (RECOMMENDED)

**Best for:** VSCode/Cursor users, easiest setup, great UI

### Features
- ✅ VSCode/Cursor extension
- ✅ Inline code suggestions
- ✅ Chat interface in sidebar
- ✅ Works with local Ollama models
- ✅ Git-aware
- ✅ Free and open source

### Setup
```bash
# 1. Install Continue extension in VSCode
code --install-extension continue.continue

# 2. Configure to use Ollama
# VSCode: Ctrl+Shift+P → "Continue: Open Config"
# Edit ~/.continue/config.json:
{
  "models": [{
    "title": "Mistral 7B",
    "provider": "ollama",
    "model": "mistral"
  }],
  "tabAutocompleteModel": {
    "title": "DeepSeek Coder",
    "provider": "ollama",
    "model": "deepseek-coder:6.7b"
  }
}

# 3. Pull coding-specific models
ollama pull deepseek-coder:6.7b
ollama pull codellama:7b

# 4. Start coding!
# Press Ctrl+L to chat
# Press Tab for autocomplete
```

### Pros & Cons
✅ Best UX - integrated directly in IDE
✅ Fast autocomplete
✅ Context-aware (sees your open files)
✅ Free forever

⚠️ Requires VSCode/Cursor
⚠️ Limited to IDE environment

**Estimated setup time:** 10 minutes

---

## Option 2: Aider (POWERFUL CLI)

**Best for:** Terminal users, autonomous coding, large refactors

### Features
- ✅ CLI-based (works everywhere)
- ✅ Can edit multiple files autonomously
- ✅ Git integration (auto-commits)
- ✅ Works with local models via Ollama
- ✅ Great for big refactors

### Setup
```bash
# 1. Install Aider
pip install aider-chat

# 2. Configure for Ollama
export OLLAMA_API_BASE=http://localhost:11434

# 3. Start Aider in your project
cd /opt/user-data/experiment/cappuccino
aider --model ollama/mistral

# 4. Give it tasks
# Example:
# > Add sentiment analysis to paper_trader_alpaca_polling.py
# > Refactor environment_Alpaca.py to use constants
```

### Pros & Cons
✅ Autonomous - can edit multiple files
✅ Git-aware - auto commits changes
✅ Works in terminal (no IDE needed)
✅ Great for refactoring

⚠️ No autocomplete
⚠️ Can make mistakes on large tasks
⚠️ CLI-only (no GUI)

**Estimated setup time:** 5 minutes

---

## Option 3: LM Studio + Open WebUI

**Best for:** Interactive chat, model comparison, non-coding tasks

### Features
- ✅ Beautiful GUI
- ✅ Easy model management
- ✅ Chat interface like ChatGPT
- ✅ API server for other tools
- ✅ Model comparison

### Setup
```bash
# 1. Download LM Studio
# Visit: https://lmstudio.ai/
# Install via AppImage or package manager

# 2. Load a model
# In LM Studio GUI:
# - Browse models
# - Download DeepSeek Coder 6.7B or CodeLlama 7B
# - Start local server

# 3. Use in chat or via API
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a Python function"}],
    "temperature": 0.7
  }'
```

### Pros & Cons
✅ Beautiful UI
✅ Easy to use
✅ Good for experimentation

⚠️ Not IDE-integrated
⚠️ Slower for coding tasks
⚠️ Requires GUI app

**Estimated setup time:** 15 minutes

---

## Option 4: Cline (Claude in VSCode)

**Best for:** If you have Claude API credits

### Features
- ✅ Uses Claude 3.5 Sonnet (best coding model)
- ✅ VSCode extension
- ✅ Can execute commands, edit files
- ✅ Autonomous task completion

### Setup
```bash
# 1. Install Cline extension
code --install-extension saoudrizwan.claude-dev

# 2. Add API key
# Get key from: https://console.anthropic.com/

# 3. Configure
# VSCode: Settings → Cline → API Key
```

### Pros & Cons
✅ Best model (Claude 3.5 Sonnet)
✅ Very capable
✅ Great for complex tasks

⚠️ Requires API credits ($)
⚠️ Not fully local
⚠️ Costs money per request

**Estimated setup time:** 5 minutes

---

## Recommended Setup: Continue + Aider

**Why both?**
- **Continue** for day-to-day coding (autocomplete, quick questions)
- **Aider** for big refactors and autonomous tasks

### Combined Setup
```bash
# 1. Install Continue extension
code --install-extension continue.continue

# 2. Install Aider
pip install aider-chat

# 3. Pull models
ollama pull mistral              # General purpose
ollama pull deepseek-coder:6.7b  # Autocomplete
ollama pull codellama:7b         # Alternative coder

# 4. Configure Continue
# ~/.continue/config.json:
{
  "models": [{
    "title": "Mistral",
    "provider": "ollama",
    "model": "mistral"
  }, {
    "title": "DeepSeek Coder",
    "provider": "ollama",
    "model": "deepseek-coder:6.7b"
  }],
  "tabAutocompleteModel": {
    "title": "DeepSeek Coder",
    "provider": "ollama",
    "model": "deepseek-coder:6.7b"
  }
}

# 5. Usage patterns
# Continue (Ctrl+L): "How do I add sentiment to this function?"
# Aider: "Refactor environment_Alpaca.py to support 16GB VRAM"
```

---

## Quick Setup Script

Run this to set up Continue + Aider:

```bash
./infrastructure/local_coding_bot/2_install_coding_bot.sh
```

---

## Model Recommendations

### For Autocomplete (Fast, Lightweight)
- **deepseek-coder:6.7b** (Best - optimized for code completion)
- **codellama:7b** (Alternative)
- **starcoder:7b** (Specialized for Python)

### For Chat/Refactoring (Smarter, Slower)
- **mistral:7b** (Good general purpose)
- **codellama:13b** (Better but slower)
- **deepseek-coder:33b** (Best quality, requires 16GB+ VRAM)

### After RX 7900 GRE Arrives
With 16GB VRAM, you can run:
- **codellama:33b** (excellent coding)
- **deepseek-coder:33b** (state-of-the-art)
- **mixtral:8x7b** (very capable, 47B parameters)

---

## Comparison Table

| Feature | Continue | Aider | LM Studio | Cline |
|---------|----------|-------|-----------|-------|
| **Setup Time** | 10min | 5min | 15min | 5min |
| **IDE Integration** | ✅ | ❌ | ❌ | ✅ |
| **Autocomplete** | ✅ | ❌ | ❌ | ❌ |
| **Multi-file Edit** | ⚠️ | ✅ | ❌ | ✅ |
| **Local (No API)** | ✅ | ✅ | ✅ | ❌ |
| **Cost** | Free | Free | Free | $$ |
| **Best For** | Daily coding | Refactoring | Experimentation | Complex tasks |

---

## Next Steps

1. **Decide which option** (I recommend Continue + Aider)
2. **Run setup script**: `./infrastructure/local_coding_bot/2_install_coding_bot.sh`
3. **Test with**: "Add type hints to environment_Alpaca.py"

The setup script is ready to run!
