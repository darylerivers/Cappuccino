# GLM Code Assistant - Your Local Claude Code Alternative

This guide shows how to use GLM-4 as an interactive coding assistant similar to Claude Code, but running locally.

## What It Does

GLM Code Assistant makes GLM-4 function like our current conversation:
- ✅ Answers coding questions
- ✅ Analyzes files and code
- ✅ Helps debug issues
- ✅ Suggests improvements
- ✅ Explains architecture
- ✅ Maintains conversation context
- ✅ Provides step-by-step help

## How It Works

### The Architecture

```
┌─────────────┐
│     You     │
└──────┬──────┘
       │ Question: "Why isn't my paper trader working?"
       ▼
┌─────────────────┐
│ GLM Assistant   │ ← System prompt (behaves like Claude Code)
│  (Wrapper)      │ ← Conversation history (context)
└──────┬──────────┘
       │ "I need to see the paper trader code"
       ▼
┌─────────────────┐
│  File Reader    │ ← Reads scripts/deployment/paper_trader.py
│  Command Runner │ ← Runs: grep "error" logs/
│  Code Searcher  │ ← Searches: grep -r "execute_trade"
└──────┬──────────┘
       │ Provides file contents, command output
       ▼
┌─────────────────┐
│      GLM-4      │ ← Analyzes information
└──────┬──────────┘
       │ "The issue is in line 145 - missing API key check"
       ▼
┌─────────────────┐
│     You         │ ← Gets actionable answer
└─────────────────┘
```

**Key Insight:** GLM-4 doesn't have tools like Claude Code, but the wrapper provides them by:
1. Reading files when GLM asks
2. Running commands when needed
3. Searching code on request
4. Maintaining conversation context

## Usage

### Interactive Mode (Like chatting with Claude)

```bash
# Start interactive session
./glm-assist

# Or
python3 integrations/glm_code_assistant.py
```

This starts a chat interface where you can ask questions naturally:

```
You: How does the training script work?

GLM: The training script (scripts/training/1_optimize_unified.py) is the main
entry point for training DRL agents. It uses Optuna for hyperparameter
optimization and supports both exploitation and exploration modes...

You: Can you show me the main training loop?

GLM: I need to see scripts/training/1_optimize_unified.py to show you the
exact code. Can you load it?

You: /read scripts/training/1_optimize_unified.py

GLM: Thanks! Looking at the code, the main training loop is in the objective()
function starting at line 156. Here's how it works:

1. Lines 156-180: Initialize environment and agent
2. Lines 182-220: Training loop (rollout + PPO update)
3. Lines 222-250: Evaluation and metrics calculation
...
```

### Non-Interactive Mode (Single question)

```bash
# Ask a quick question
./glm-assist --question "What models are currently being trained?"

# Ask about a specific file
./glm-assist --file constants.py --question "What are the main configuration settings?"
```

### Available Commands

While in interactive mode, you can use these commands:

```
/read <file>      - Read a file and add to context
                    Example: /read scripts/training/1_optimize_unified.py

/run <command>    - Run a shell command
                    Example: /run ls -la scripts/

/search <pattern> - Search code for a pattern
                    Example: /search "def train"

/ls <directory>   - List files in a directory
                    Example: /ls drl_agents/agents/

/clear            - Clear conversation history (fresh start)

/help             - Show command help

/quit             - Exit assistant
```

## Example Conversations

### Example 1: Debugging

```
You: My paper trader isn't executing any trades. Can you help?

GLM: I'll help you debug this. First, I need to see the paper trader code
and recent logs. Can you run:
  /read scripts/deployment/paper_trader_alpaca_polling.py
  /run tail -50 paper_trader.log

You: /run tail -50 paper_trader.log

GLM: I see the error. Line 19 shows "the following arguments are required:
--model-dir". The paper trader needs a model directory to load the DRL agent.

Looking at the error, you need to provide the --model-dir argument when
starting the trader. Try:
  python scripts/deployment/paper_trader_alpaca_polling.py --model-dir <path>

Or use the convenience script in paper_trading_trial965/:
  cd paper_trading_trial965 && ./START.sh

You: Thanks! That worked.

GLM: Great! The trader should now be running. You can monitor it with:
  tail -f paper_trading_trial965/logs/fast_*.log
```

### Example 2: Understanding Code

```
You: Can you explain how the FT-Transformer works in the PPO agent?

GLM: I need to see the agent code to explain the FT-Transformer integration.
Can you load:
  /read drl_agents/agents/AgentPPO_FT.py

You: /read drl_agents/agents/AgentPPO_FT.py

GLM: Perfect! Looking at the code, the FT-Transformer integration works like this:

**Architecture** (lines 45-120):
1. Input Processing (lines 45-60):
   - Market data (OHLCV) is normalized
   - Categorical features are embedded
   - Continuous features are projected

2. FT-Transformer Encoder (lines 62-85):
   - Uses multi-head attention to capture feature interactions
   - Each feature gets its own embedding
   - Attention weights learn which features matter most

3. Policy/Value Heads (lines 87-120):
   - Transformer output feeds into policy network (actor)
   - Separate value network (critic) estimates state value
   - Both share the transformer backbone

**Key Innovation**:
The FT-Transformer is better than MLPs for tabular data because it learns
feature interactions automatically rather than relying on manual feature
engineering.

You: What hyperparameters control the transformer size?

GLM: Looking at lines 156-180, the key hyperparameters are:
- `n_heads` (line 162): Number of attention heads (default: 4-8)
- `n_blocks` (line 164): Transformer depth (default: 2-4)
- `d_ffn_factor` (line 166): Feed-forward network size multiplier
- `attention_dropout` (line 168): Dropout for attention weights
- `ffn_dropout` (line 170): Dropout for feed-forward layers

These are optimized by Optuna during training (see lines 266-285 in
1_optimize_unified.py).
```

### Example 3: Architecture Review

```
You: I want to add a new feature to the trading environment. What files do I need to modify?

GLM: To add a new feature to the trading environment, you'll need to modify:

1. **environment_Alpaca_phase2.py** - Main environment class
   - Add feature to observation space (line ~120)
   - Calculate feature in _get_observation() (line ~450)

2. **constants.py** - If the feature needs configuration
   - Add any new constants or settings

3. **drl_agents/agents/AgentPPO_FT.py** (maybe)
   - Update input dimensions if feature changes obs space size
   - Modify preprocessing if needed (line ~200)

4. **scripts/training/1_optimize_unified.py** (maybe)
   - Add hyperparameters if feature needs tuning

Would you like me to look at a specific file to show you exactly where to add
the feature?

You: Yes, show me where to add it in the environment.

GLM: I need to see the environment code:
  /read environment_Alpaca_phase2.py

You: /read environment_Alpaca_phase2.py

GLM: Perfect! Here's exactly what to do:

**Step 1: Define observation space** (around line 120)
```python
# Add your new feature dimension
self.observation_space = gym.spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(original_features + 1,),  # +1 for your new feature
    dtype=np.float32
)
```

**Step 2: Calculate feature** (around line 450 in _get_observation())
```python
def _get_observation(self):
    # ... existing features ...

    # Add your new feature
    my_new_feature = self._calculate_my_feature()

    # Append to observation
    obs = np.concatenate([existing_obs, [my_new_feature]])
    return obs
```

**Step 3: Add calculation method** (around line 600)
```python
def _calculate_my_feature(self):
    # Your feature logic here
    # Example: RSI, custom indicator, etc.
    return feature_value
```

What feature are you planning to add?
```

## Comparison to Claude Code

| Feature | Claude Code | GLM Assistant |
|---------|-------------|---------------|
| File Reading | ✅ Automatic | ✅ Manual (`/read`) |
| Command Execution | ✅ Automatic | ✅ Manual (`/run`) |
| Code Search | ✅ Automatic | ✅ Manual (`/search`) |
| Conversation Context | ✅ Automatic | ✅ Automatic |
| Project Understanding | ✅ Built-in | ✅ System prompt |
| Code Suggestions | ✅ Yes | ✅ Yes |
| Interactive Chat | ✅ Yes | ✅ Yes |
| Tool Use | ✅ Native | ⚠️ Simulated |
| File Editing | ✅ Native | ❌ Manual (suggests changes) |
| Speed | ✅ Fast | ⚠️ Slower (3-10s per response) |
| Always Available | ❌ Needs connection | ✅ Local |
| Cost | ❌ API costs | ✅ Free (local) |

## Tips for Best Results

### 1. Be Specific

**Good:**
```
You: Why is my training using only 2GB of GPU VRAM when I have 16GB available?
```

**Less Good:**
```
You: GPU not working
```

### 2. Provide Context

**Good:**
```
You: I'm getting this error when starting training:
[paste error]

I just upgraded to an AMD GPU from NVIDIA.
```

**Less Good:**
```
You: Error
```

### 3. Use Commands to Help GLM

When GLM asks to see a file or run a command, use the built-in commands:

```
GLM: I need to see the training script to help debug this.

You: /read scripts/training/1_optimize_unified.py

GLM: Thanks! I can see the issue now...
```

### 4. Break Down Complex Questions

**Instead of:**
```
You: Help me understand the entire training pipeline, how Optuna works,
why the agent architecture was chosen, and how to improve performance.
```

**Do:**
```
You: Can you explain the overall training pipeline?
GLM: [explains]

You: How does Optuna fit into this?
GLM: [explains]

You: Why use PPO with FT-Transformer?
GLM: [explains]
```

### 5. Load Context First

For questions about specific files:

```
# Load file into context first
You: /read drl_agents/agents/AgentPPO_FT.py

# Then ask questions about it
You: How does the policy network work in this agent?
```

## Advanced Usage

### Loading Multiple Files

```
You: /read scripts/training/1_optimize_unified.py
You: /read drl_agents/agents/AgentPPO_FT.py
You: /read environment_Alpaca_phase2.py

You: How do these three components work together?

GLM: [analyzes all three files in context]
```

### Combining Commands

```
You: /search "class AgentPPO"
You: [See results]
You: /read drl_agents/agents/AgentPPO_FT.py
You: Explain the AgentPPO class architecture
```

### Debugging Workflow

```
1. You: /run python my_script.py
   [Error occurs]

2. You: I got this error: [paste]

3. GLM: I need to see the script:
   You: /read my_script.py

4. GLM: The issue is on line 45. Here's the fix...

5. You: [Make the change]

6. You: /run python my_script.py
   [Success!]
```

## Limitations

### What GLM Assistant CAN'T Do

1. **Edit files directly** - It suggests changes, but you make them
2. **Execute Python code** - It can suggest code to run via `/run`
3. **Access external APIs** - Limited to local files and commands
4. **See real-time data** - Can only see what you provide
5. **Remember between sessions** - Each session starts fresh (no persistent memory)

### What GLM Assistant CAN Do

1. ✅ Understand your codebase
2. ✅ Debug issues
3. ✅ Suggest solutions
4. ✅ Explain complex code
5. ✅ Review architecture
6. ✅ Provide code examples
7. ✅ Help plan changes
8. ✅ Answer technical questions

## Performance

- **Response time**: 3-10 seconds per message (vs 1-2s for Claude Code)
- **Context window**: ~10 messages (older messages dropped)
- **File size limit**: Works best with files <1000 lines
- **Memory usage**: ~6GB when GLM is running

## Keyboard Shortcuts

- **Up/Down arrows**: Browse command history
- **Ctrl+C**: Exit assistant
- **Ctrl+D**: Exit assistant
- **Tab**: Auto-complete file paths (in some terminals)

## Examples of Good Questions

```
✓ "How does the paper trader execute trades?"
✓ "Why am I getting 'module not found' error?"
✓ "What's the difference between AgentPPO and AgentPPO_FT?"
✓ "How can I increase GPU utilization during training?"
✓ "Walk me through what happens when I run ./resume_main_training.sh"
✓ "What files do I need to modify to add a new trading indicator?"
✓ "Explain the reward calculation in the trading environment"
✓ "How does the FT-Transformer process market data?"
```

## Troubleshooting

### GLM is slow

- Use shorter, more focused questions
- Clear history with `/clear` periodically
- Don't load very large files

### GLM asks for files repeatedly

- Load all relevant files at the start of conversation
- Files remain in context for ~10 messages

### GLM gives wrong answers

- Provide more context
- Load relevant files so it can see actual code
- Ask for clarification or alternatives

### Connection errors

- Make sure Ollama is running: `ps aux | grep ollama`
- Check if GLM-4 is loaded: `ollama list | grep glm4`
- Restart Ollama if needed: `ollama serve`

## Summary

GLM Code Assistant gives you a **local Claude Code alternative** that:
- Understands your project
- Helps debug issues
- Explains code
- Suggests improvements
- Maintains conversation context

It's not as seamless as Claude Code (you need to manually provide files/run commands), but it's:
- ✅ Free
- ✅ Local (works offline)
- ✅ Private (your code stays on your machine)
- ✅ Available anytime

**Usage:**
```bash
./glm-assist                    # Interactive mode
./glm-assist -q "your question" # Quick question
```

Enjoy your local AI coding assistant!
