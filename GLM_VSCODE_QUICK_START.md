# GLM-4 VSCode Quick Start Card

## 🚀 Getting Started (30 seconds)

1. **Open VSCode**: `code /opt/user-data/experiment/cappuccino`
2. **Open Continue Panel**: Press `Ctrl+Shift+L`
3. **Verify GLM-4 is selected**: Look for "GLM-4.7-Flash" in model dropdown
4. **Start chatting**: Type a question and press Enter

---

## 📍 Essential Shortcuts

| Action | Shortcut | Description |
|--------|----------|-------------|
| **Open Continue** | `Ctrl+Shift+L` | Open AI assistant panel |
| **Inline Chat** | `Ctrl+L` | Chat about highlighted code |
| **Quick Question** | `Ctrl+Shift+M` | Ask without opening panel |
| **Accept AI Edit** | `Tab` | Accept suggested code |
| **Reject AI Edit** | `Esc` | Reject suggested code |

---

## 💬 Custom Commands (Type `/` in chat)

### 🎯 Most Useful for Training Optimization:
- **`/optimize`** - Analyze code for performance improvements (GPU, memory)
- **`/drl`** - DRL/Trading specific analysis (rewards, states, actions)
- **`/debug`** - Find bugs and edge cases

### 🛠️ General Coding:
- **`/test`** - Generate unit tests
- **`/explain`** - Explain complex code
- **`/refactor`** - Improve code quality
- **`/comment`** - Add code comments
- **`/edit`** - Edit highlighted code
- **`/cmd`** - Generate shell command

---

## 🎓 Example Usage

### Optimize Training Speed (Your Current Task!)
```
1. Open: scripts/training/1_optimize_unified.py
2. Highlight the study.optimize() section
3. Press: Ctrl+L
4. Type: /optimize
5. Follow GLM's suggestions
```

### Understand DRL Rewards
```
1. Highlight reward calculation code
2. Press: Ctrl+L
3. Type: /drl
4. Ask: "Is this reward function properly shaped?"
```

### Debug Paper Trading
```
1. Open: scripts/deployment/paper_trader_alpaca_polling.py
2. Highlight problematic function
3. Press: Ctrl+L
4. Type: /debug
```

### Quick Question (Without Opening Files)
```
1. Press: Ctrl+Shift+M
2. Ask: "How do I add Optuna pruning to speed up hyperparameter search?"
3. GLM responds with code example
```

---

## 🔥 Pro Tips

### 1. Add Context with @
```
@terminal What's this error?
@diff Explain these changes
@file utils/function_train_test.py How does this work?
```

### 2. Chain Commands
```
1. /explain  (understand code first)
2. /optimize (then optimize it)
3. /test     (then write tests)
```

### 3. Be Specific
❌ "Make this faster"
✅ "Optimize this loop for GPU. Focus on batch processing and memory efficiency."

### 4. Highlight First
Always highlight relevant code before pressing `Ctrl+L`

### 5. Use Custom Commands
`/optimize` and `/drl` are specifically tuned for your trading project!

---

## 📊 For Your Training Optimization Task

**You have these files ready for GLM:**
- `OPTIMIZE_TRAINING_PROMPT.md` - Full problem description
- `OPTIMIZE_TRAINING_QUICK.txt` - Quick version
- `TRAINING_SPEEDUP_GUIDE.md` - Implementation guide
- `best_trial_hyperparams.json` - Current best hyperparameters

**How to use them:**
```
1. Open VSCode
2. Press Ctrl+Shift+L
3. Type: @file OPTIMIZE_TRAINING_QUICK.txt
4. Ask: "Implement these optimizations in scripts/training/1_optimize_unified.py"
5. GLM will show you the exact code changes
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| GLM not responding | Check Ollama: `pgrep -f "ollama serve"` |
| Slow responses | Switch to Qwen2.5-Coder (faster model) |
| Wrong suggestions | Add more context with @file, @terminal |
| Can't see Continue panel | Press `Ctrl+Shift+L` |

---

## 📈 What's Different from Claude Code?

| Feature | GLM-4 (Continue) | Claude Code |
|---------|------------------|-------------|
| **Speed** | 3-5 sec ⚡ | 1-2 sec ⚡⚡ |
| **Privacy** | 100% local 🔒 | Cloud ☁️ |
| **Cost** | Free 💰 | Paid 💳 |
| **Offline** | ✅ Yes | ❌ No |
| **Quality** | Good for coding ⭐⭐⭐⭐ | Excellent ⭐⭐⭐⭐⭐ |

**Strategy:** Use GLM for quick iterations, Claude for complex refactoring

---

## 🎯 Your Next Step

**Open VSCode and try this RIGHT NOW:**

1. Press `Ctrl+Shift+L`
2. Type: "Read OPTIMIZE_TRAINING_QUICK.txt and tell me the 3 main optimizations to implement"
3. Follow GLM's guidance to speed up your training!

---

**Need help?** Full guide at: `~/.continue/GLM_VSCODE_GUIDE.md`

**Test setup:** `python3 test_glm_vscode.py`
