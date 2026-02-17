# Making GLM-4 Behave Like Claude Code

This guide shows how to use prompt engineering to make GLM-4 function like an intelligent agent (similar to Claude Code) for your trading system.

## Key Differences

| Feature | Claude Code | GLM-4 (Base) | GLM-4 (With Prompts) |
|---------|-------------|--------------|----------------------|
| File Access | ✅ Native | ❌ No | ❌ No (but can analyze provided content) |
| Code Execution | ✅ Native | ❌ No | ❌ No |
| Tool Use | ✅ Native | ⚠️ Limited | ✅ Via structured prompts |
| Context Memory | ✅ Automatic | ❌ No | ✅ Manual implementation |
| Structured Output | ✅ Native | ❌ No | ✅ Via JSON prompts |
| Specialized Behavior | ✅ Built-in | ❌ Generic | ✅ Via system prompts |

## How It Works

### 1. System Prompts (The Key Technique)

System prompts define the AI's behavior, expertise, and output format. Think of it as "programming" the AI's personality and capabilities.

**Basic pattern:**
```
You are a [role] with expertise in [domain].

# Your Capabilities
- Capability 1
- Capability 2

# Your Behavior
- How to approach problems
- Communication style
- Output format

# Guidelines
- Rule 1
- Rule 2
```

### 2. Context Management

Unlike Claude Code (which has automatic context), GLM-4 needs manual context management:

```python
# Keep conversation history
conversation_history = [
    {"role": "user", "content": "Question 1"},
    {"role": "assistant", "content": "Answer 1"},
    {"role": "user", "content": "Question 2"}  # References Answer 1
]

# Include in each prompt
full_prompt = f"{system_prompt}\n\nHistory:\n{format_history(conversation_history)}\n\nNew: {new_question}"
```

### 3. Structured Outputs

Force JSON outputs for reliable parsing:

```python
payload = {
    "model": "glm4",
    "prompt": "Analyze this...",
    "format": "json"  # Forces JSON response
}
```

## Quick Start

### Installation

```bash
# Ollama already installed, GLM-4 already downloaded
ollama list | grep glm4  # Should show glm4:latest
```

### Basic Usage

```python
from integrations.glm_agent import GLMAgent

# Create agent with default trading analyst prompt
agent = GLMAgent()

# Analyze sentiment
result = agent.analyze_sentiment(
    "Bitcoin surges 8% on ETF approval news"
)
print(result)  # Returns structured JSON

# Multi-turn conversation (maintains context)
response1 = agent.chat("What's your BTC outlook?")
response2 = agent.chat("Should I increase my position?")  # Knows previous context
```

### Advanced: Custom Personas

```python
from integrations.glm_agent import GLMAgent
from integrations.glm_prompts import get_prompt

# Use specialized persona
agent = GLMAgent(
    system_prompt_file="/tmp/my_prompt.txt",
    temperature=0.2  # Lower = more consistent
)

# Or use pre-built personas
from pathlib import Path
prompt_file = Path("/tmp/analyst.txt")
prompt_file.write_text(get_prompt('analyst'))

agent = GLMAgent(system_prompt_file=str(prompt_file))
```

## Pre-Built Personas

### 1. Market Analyst
**Use for:** Sentiment analysis, price predictions, market commentary

```python
from integrations.glm_prompts import get_prompt
analyst_prompt = get_prompt('analyst')

# Provides structured analysis with:
# - Current state assessment
# - Context and catalysts
# - Price implications
# - Risk factors
# - Actionable recommendations
```

**Example:**
```python
agent = GLMAgent(system_prompt=analyst_prompt)
result = agent.analyze_sentiment("BTC hits new ATH")
# Returns JSON with sentiment, confidence, risk_level, etc.
```

### 2. Risk Manager
**Use for:** Portfolio review, position sizing, risk assessment

```python
risk_prompt = get_prompt('risk')

# Evaluates:
# - Position sizing appropriateness
# - Concentration risk
# - Market risk scenarios
# - Stop-loss recommendations
```

**Example:**
```python
agent = GLMAgent(system_prompt=risk_prompt)
result = agent.risk_assessment(portfolio_data, market_data)
# Returns risk level, recommended actions, position adjustments
```

### 3. Signal Validator
**Use for:** Pre-trade checks before executing DRL agent signals

```python
validator_prompt = get_prompt('validator')

# Validates:
# - Signal alignment with market
# - Position size appropriateness
# - Risk/reward ratio
# - Identifies red flags
```

**Example:**
```python
agent = GLMAgent(system_prompt=validator_prompt)
result = agent.evaluate_trade_signal(drl_signal, market_data)
# Returns approved/rejected, confidence, warnings
```

### 4. News Analyzer
**Use for:** Real-time news impact prediction

```python
news_prompt = get_prompt('news')

# Analyzes:
# - Impact magnitude (1-10)
# - Direction (bullish/bearish)
# - Timeframe (immediate to long-term)
# - Affected assets
# - Price predictions
```

**Example:**
```python
agent = GLMAgent(system_prompt=news_prompt)
result = agent.analyze_news_impact(headlines, tickers)
# Returns impact scores and price predictions per ticker
```

### 5. Claude Code Style
**Use for:** Debugging, code review, technical explanations

```python
claude_prompt = get_prompt('claude')

# Provides:
# - Systematic problem analysis
# - Step-by-step solutions
# - Concrete examples
# - Acknowledges uncertainty
```

**Example:**
```python
agent = GLMAgent(system_prompt=claude_prompt)
response = agent.chat("Why is my backtest better than paper trading?")
# Returns thoughtful analysis with specific suggestions
```

## Integration with Trading System

### Real-time Sentiment Filter

```python
# In your paper trader
from integrations.glm_agent import GLMAgent
from integrations.glm_prompts import get_prompt

# Initialize once
validator = GLMAgent(system_prompt=get_prompt('validator'))

def execute_trade(signal, market_data):
    # Validate with GLM-4 before executing
    validation = validator.evaluate_trade_signal(signal, market_data)

    if not validation.get('approved', False):
        print(f"Signal rejected: {validation['reasons']}")
        return None

    if validation.get('suggested_modifications'):
        # Adjust position size based on GLM-4 recommendation
        signal['position_size'] = validation['suggested_modifications']['position_size']

    # Execute trade
    return place_order(signal)
```

### News-Based Trading Filter

```python
# Check news before market open
news_agent = GLMAgent(system_prompt=get_prompt('news'))

def should_trade_today(recent_headlines, tickers):
    analysis = news_agent.analyze_news_impact(recent_headlines, tickers)

    # Don't trade if major negative news
    for ticker in tickers:
        impact = analysis.get(ticker, {})
        if impact.get('impact_score', 0) > 8 and impact.get('direction') == 'bearish':
            print(f"Pausing {ticker} trading due to major negative news")
            return False

    return True
```

### Portfolio Review

```python
# Daily risk check
risk_agent = GLMAgent(system_prompt=get_prompt('risk'))

def daily_risk_review(portfolio, market_conditions):
    assessment = risk_agent.risk_assessment(portfolio, market_conditions)

    if assessment['risk_level'] in ['high', 'critical']:
        # Auto-reduce positions
        for ticker, action in assessment['position_adjustments'].items():
            if action == 'reduce':
                reduce_position(ticker, 0.5)  # Cut in half

    return assessment
```

## Performance Tips

### 1. Temperature Settings

```python
# Use case specific temperatures
agent = GLMAgent(
    temperature=0.1   # Risk assessment (very consistent)
    temperature=0.2   # News analysis (mostly consistent)
    temperature=0.3   # General chat (balanced)
    temperature=0.5   # Creative analysis (more varied)
)
```

### 2. Token Optimization

```python
# Prefer JSON for structured data (parseable)
result = agent.analyze_sentiment(text)  # Returns JSON

# Use free-form only when needed
response = agent.chat(question)  # Returns text
```

### 3. Context Window Management

```python
# The agent auto-trims history to last 10 turns
# But you can manually control:

agent.clear_history()  # Reset context
stats = agent.get_stats()  # Check token usage
```

### 4. Batch Processing

```python
# Analyze multiple news items efficiently
headlines = fetch_latest_news()  # Get all at once
result = agent.analyze_news_impact(headlines, ['BTC', 'ETH'])

# Instead of:
# for headline in headlines:  # Slower, more tokens
#     agent.analyze_sentiment(headline)
```

## Testing

Run the test suite to see all personas in action:

```bash
cd /opt/user-data/experiment/cappuccino
python integrations/test_glm_agent.py
```

This will show:
- Market analyst sentiment analysis
- Risk manager portfolio assessment
- Signal validator trade evaluation
- News analyzer impact predictions
- Claude-style problem solving

## API Reference

### GLMAgent Class

```python
GLMAgent(
    model: str = "glm4",
    ollama_url: str = "http://localhost:11434",
    system_prompt_file: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 2000
)
```

**Methods:**
- `analyze_sentiment(text, context=None)` → Dict
- `evaluate_trade_signal(signal, market_data, portfolio=None)` → Dict
- `analyze_news_impact(headlines, tickers)` → Dict
- `risk_assessment(portfolio, market_conditions)` → Dict
- `chat(message, temperature=None)` → str
- `clear_history()` → None
- `get_stats()` → Dict

## Comparison to Claude Code

**What GLM-4 Agent CAN do:**
- ✅ Structured analysis with confidence scores
- ✅ Multi-turn conversations with context
- ✅ Specialized domain expertise (via prompts)
- ✅ JSON outputs for reliable parsing
- ✅ Multiple personas for different tasks
- ✅ Trade signal validation
- ✅ Risk assessment
- ✅ News impact analysis

**What GLM-4 Agent CANNOT do (vs Claude Code):**
- ❌ Read/write files directly
- ❌ Execute code
- ❌ Use tools (git, grep, etc.)
- ❌ Browse file systems
- ❌ Automatic context management
- ❌ Interactive debugging

**Workaround:** Pass file contents and outputs as strings in prompts

## Best Practices

1. **Use specific personas for specific tasks** - Don't use market analyst for code review
2. **Lower temperature for critical decisions** - Risk assessment, trade validation
3. **Request JSON for structured data** - Easier to parse and use programmatically
4. **Provide context in prompts** - More context = better analysis
5. **Track token usage** - Use `get_stats()` to optimize
6. **Clear history between unrelated tasks** - Prevents context pollution
7. **Validate JSON responses** - Always handle parsing errors

## Example: Complete Trading Workflow

```python
from integrations.glm_agent import GLMAgent
from integrations.glm_prompts import get_prompt

# Initialize specialized agents
news_agent = GLMAgent(system_prompt=get_prompt('news'), temperature=0.2)
validator = GLMAgent(system_prompt=get_prompt('validator'), temperature=0.1)
risk_mgr = GLMAgent(system_prompt=get_prompt('risk'), temperature=0.1)

# 1. Morning: Check news
headlines = fetch_news()
news_impact = news_agent.analyze_news_impact(headlines, ['BTC', 'ETH'])

if any(h['impact_score'] > 9 for h in news_impact.values()):
    print("Major news detected, pausing trading")
    sys.exit(0)

# 2. During trading: Validate signals
drl_signal = agent.get_action(state)
validation = validator.evaluate_trade_signal(drl_signal, market_data)

if validation['approved']:
    execute_trade(drl_signal)
else:
    print(f"Signal rejected: {validation['reasons']}")

# 3. End of day: Risk review
portfolio_state = get_portfolio()
risk_report = risk_mgr.risk_assessment(portfolio_state, market_conditions)

if risk_report['risk_level'] == 'critical':
    send_alert(risk_report)
    reduce_exposure(risk_report['position_adjustments'])
```

## Summary

GLM-4 can't fully replicate Claude Code's tool-using capabilities, but with proper prompt engineering it can:
- Provide specialized domain expertise
- Maintain conversation context
- Generate structured, parseable outputs
- Validate trading decisions
- Analyze market conditions
- Assess risk

The key is using **system prompts** to define behavior and **context management** to maintain coherence across multiple interactions.
