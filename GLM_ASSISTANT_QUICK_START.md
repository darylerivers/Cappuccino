# GLM Code Assistant - Quick Start

Your local alternative to Claude Code when you don't have Claude access.

## Start It

```bash
./glm-assist
```

## Use It

```
You: How does the training script work?

GLM: [explains the training process]

You: /read scripts/training/1_optimize_unified.py

GLM: [analyzes the file and provides detailed explanation]

You: How can I increase GPU usage?

GLM: [suggests increasing batch sizes, provides code changes]
```

## Commands

```bash
/read <file>      # Load file into context
/run <command>    # Run shell command
/search <pattern> # Search code
/ls <dir>         # List directory
/clear            # Clear history
/quit             # Exit
```

## Quick Questions (No Interactive Mode)

```bash
# Ask single question
./glm-assist -q "Why isn't my paper trader working?"

# Ask about specific file
./glm-assist -f constants.py -q "What are the key settings?"
```

## Example Session

```
$ ./glm-assist

You: I'm getting a ModuleNotFoundError when running training

GLM: I need to see the error. Can you run:
  /run python scripts/training/1_optimize_unified.py

You: /run python scripts/training/1_optimize_unified.py

GLM: The error shows missing 'torch'. Make sure you're in the right
environment:
  source activate_rocm_env.sh

Then retry the training.

You: That worked! Now how do I monitor GPU usage?

GLM: Use this command:
  /run rocm-smi --showuse

Or for continuous monitoring:
  watch -n 1 'rocm-smi --showuse'

You: Thanks!
```

## Tips

1. **Load files first** for questions about code:
   ```
   /read the_file.py
   Now ask about the_file.py
   ```

2. **Be specific** in questions:
   ✓ "Why is training only using 2GB VRAM?"
   ✗ "GPU problem"

3. **Provide context**:
   ✓ "I just upgraded to AMD GPU and training won't start [error]"
   ✗ "Error"

4. **Break down complex questions**:
   Instead of asking everything at once, have a conversation.

## Comparison to Claude Code

**Same as Claude:**
- Understands your project
- Helps debug
- Explains code
- Suggests fixes
- Maintains conversation

**Different:**
- Slower (5-15s vs 1-2s)
- Manual file loading (`/read` vs automatic)
- Suggests edits (doesn't make them)
- But: FREE and WORKS OFFLINE!

## When to Use

- ✅ Claude access expired
- ✅ Working offline
- ✅ Need quick coding help
- ✅ Want to understand code
- ✅ Debugging issues
- ✅ Privacy concerns (code stays local)

## Full Guide

See `GLM_ASSISTANT_GUIDE.md` for complete documentation.

## Troubleshooting

**Slow responses?**
- Use shorter questions
- `/clear` periodically
- Don't load huge files

**GLM asks for same file twice?**
- Load it early in conversation
- Files stay in context for ~10 messages

**Wrong answers?**
- Provide more context
- Load relevant files with `/read`
- Ask for clarification

Start using: `./glm-assist`
