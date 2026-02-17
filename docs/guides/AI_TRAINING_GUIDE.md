# TinyLlama AI Training System for Cappuccino

## 🎯 Overview

Train a custom TinyLlama model on your actual trading data to create an AI advisor that learns from YOUR system's real performance.

**Key Features:**
- **Tiny VRAM**: Only ~650MB (runs alongside DRL training)
- **Continuous Learning**: Gets smarter as you trade more
- **Domain-Specific**: Trained on YOUR trading patterns and market conditions
- **$0 Cost**: Uses local Ollama + open-source models

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
chmod +x ai_training/setup.sh
./ai_training/setup.sh
```

This installs Unsloth (optimized training) and required packages.

### Step 2: Collect Training Data

Run your paper trader to start collecting trade data:

```bash
./start_automation.sh
```

The system automatically logs:
- Trade decisions with full context (signals, news, macro)
- Outcomes (profit/loss, duration)
- Market conditions at time of trade

**Minimum recommended:** 50+ completed trades for good fine-tuning

### Step 3: Check Your Data

```bash
python3 -c "from ai_training.trade_logger import get_trade_logger; \
            import json; print(json.dumps(get_trade_logger().get_training_stats(), indent=2))"
```

Output:
```json
{
  "total_trades": 127,
  "trades_with_outcomes": 89,
  "training_examples": 89,
  "winning_trades": 62,
  "losing_trades": 27,
  "avg_profit_pct": 2.3
}
```

### Step 4: Train Your Model

**Basic training (3 epochs, ~30 minutes):**
```bash
python3 ai_training/train_tinyllama.py
```

**Advanced options:**
```bash
python3 ai_training/train_tinyllama.py \
  --epochs 5 \
  --batch-size 4 \
  --lr 3e-4 \
  --export-ollama
```

### Step 5: Deploy to Ollama

```bash
cd ai_training/models/tinyllama_trading_latest
ollama create cappuccino-ai -f Modelfile
```

Test it:
```bash
ollama run cappuccino-ai "BTC showing bullish signal but VIX is at 28. What do you think?"
```

---

## 📊 How It Works

### Data Collection Pipeline

Every trade is logged with full context:

```python
{
  "trade_id": "BTC_20251209_143022",
  "symbol": "BTC/USD",
  "action": "BUY",
  "agent_signal": 0.73,
  "market_context": {
    "price": 43250,
    "volume": 1.2M,
    "signal": "bullish"
  },
  "macro_context": {
    "vix": 15.4,
    "regime": "normal",
    "multiplier": 1.0
  },
  "news_summary": {
    "recommendation": "proceed",
    "bullish_count": 4,
    "bearish_count": 0
  },
  "outcome": {
    "profit_loss_pct": 2.3,
    "hold_duration_hours": 6.5,
    "success": true
  }
}
```

### Training Process

1. **Load Base Model**: TinyLlama-1.1B (open-source, 1.1 billion parameters)
2. **Add LoRA Adapters**: Only train ~2% of parameters (efficient!)
3. **Fine-tune**: Learn from your trade outcomes
4. **Export**: Save as Ollama-compatible model

### What The Model Learns

- **Pattern Recognition**: "Similar setup to trade #47 which gained +3.2%"
- **Risk Assessment**: "High VIX + bearish news = reduce position 50%"
- **Timing**: "This pattern typically resolves in 4-6 hours"
- **Context Integration**: Combines technical, fundamental, and macro signals

---

## 🔄 Continuous Learning

### Manual Retraining

Re-train weekly/monthly as you collect more data:

```bash
python3 ai_training/train_tinyllama.py --epochs 3
cd ai_training/models/tinyllama_trading_latest
ollama create cappuccino-ai -f Modelfile
```

### Performance Tracking

Compare model versions:

```bash
python3 ai_training/test_model.py \
  --model1 models/run_20251201_120000 \
  --model2 models/run_20251208_120000
```

---

## 💡 Example Queries

Once deployed, ask your fine-tuned model:

```bash
ollama run cappuccino-ai "ETH signal is 0.85 but VIX just spiked to 30. Trade or wait?"

ollama run cappuccino-ai "Analyze this: BTC +2.5% today, news shows 3 bullish signals, but treasury yields up 0.3%"

ollama run cappuccino-ai "What were the key factors in our most profitable BTC trades?"
```

---

## 📈 Training Data Quality

### Good Training Data

✅ Diverse market conditions (bull, bear, sideways)
✅ Mix of winning and losing trades
✅ Minimum 50-100 completed trades
✅ Rich context (news, macro, technical)
✅ Clear outcomes (profit/loss)

### What Improves Results

- **More data**: 200+ trades >> 50 trades
- **Longer hold times**: Better outcome attribution
- **Varied conditions**: Different VIX levels, news types
- **Balanced outcomes**: Not 100% winners (model needs to learn mistakes)

---

## 🛠️ Advanced Usage

### Custom Training Data

Add your own examples manually:

```python
from ai_training.trade_logger import get_trade_logger

logger = get_trade_logger()

# Log a historical trade
trade_id = logger.log_trade_decision(
    symbol="BTC/USD",
    action="BUY",
    agent_signal=0.82,
    position_size=1000,
    market_context={"price": 42000, "signal": "strong_buy"},
    macro_context={"vix": 14.2, "regime": "low_volatility"},
    news_summary={"recommendation": "proceed", "bullish_count": 5},
    reasoning="Strong technical + macro + sentiment alignment"
)

# Add outcome later
logger.log_trade_outcome(
    trade_id=trade_id,
    profit_loss=75.50,
    profit_loss_pct=7.55,
    hold_duration_hours=8.0,
    exit_reason="target_reached"
)
```

### CPU-Only Training

Force CPU if GPU busy:

```bash
CUDA_VISIBLE_DEVICES="" python3 ai_training/train_tinyllama.py
```

Slower (~2-3x) but doesn't conflict with DRL training.

### Hyperparameter Tuning

```bash
# Longer training
python3 ai_training/train_tinyllama.py --epochs 10

# Smaller batches (less VRAM)
python3 ai_training/train_tinyllama.py --batch-size 1

# Higher learning rate (faster convergence, more risky)
python3 ai_training/train_tinyllama.py --lr 5e-4
```

---

## 🔍 Troubleshooting

### "Out of memory" during training

```bash
# Reduce batch size
python3 ai_training/train_tinyllama.py --batch-size 1

# Or use CPU
CUDA_VISIBLE_DEVICES="" python3 ai_training/train_tinyllama.py
```

### "Not enough training data"

Need minimum 10 trades with outcomes. Keep trading to collect more data!

### Model gives generic responses

- Need more training data (50+ recommended)
- Try more epochs: `--epochs 5`
- Check data quality: Are outcomes clearly attributed?

### Ollama import fails

Make sure Ollama is installed:
```bash
curl https://ollama.ai/install.sh | sh
```

---

## 📊 Expected Performance

| Training Data | Model Quality | Response Time |
|--------------|---------------|---------------|
| 10-50 trades | Basic | ~0.5s |
| 50-200 trades | Good | ~0.5s |
| 200+ trades | Excellent | ~0.5s |
| 1000+ trades | Expert-level | ~0.5s |

**Note**: TinyLlama is fast (~0.5s per response) even on CPU!

---

## 🎯 Next Steps

1. ✅ **Setup complete** - Dependencies installed
2. 🔄 **Collect data** - Run paper trader for 1-2 weeks
3. 🎓 **First training** - Train when you hit 50+ trades
4. 🚀 **Deploy** - Create Ollama model
5. 🔁 **Retrain monthly** - Continuous improvement

---

## 💰 Cost Analysis

| Component | Cost | Notes |
|-----------|------|-------|
| TinyLlama Base Model | $0 | Open-source |
| Training (GPU) | $0 | Your hardware |
| Inference (CPU/GPU) | $0 | Local Ollama |
| Data Storage | ~1MB | Negligible |
| **Total** | **$0** | Completely free! |

---

## ✅ Benefits vs Generic LLM

| Feature | Generic Mistral | Fine-tuned TinyLlama |
|---------|----------------|---------------------|
| Domain Knowledge | General | **Your trading system** |
| VRAM Usage | 4GB | 650MB |
| Learns From Mistakes | ❌ | ✅ |
| Knows Your Data | ❌ | ✅ |
| Response Time | ~3s | ~0.5s |
| Continuous Improvement | ❌ | ✅ |
| Cost | Free | Free |

**Bottom Line**: A 1B parameter model trained on YOUR data beats a 7B generic model for YOUR specific use case!

---

## 📚 Additional Resources

- **Unsloth Documentation**: https://github.com/unslothai/unsloth
- **TinyLlama Paper**: https://arxiv.org/abs/2401.02385
- **LoRA Fine-tuning**: https://arxiv.org/abs/2106.09685
- **Ollama Documentation**: https://ollama.ai/docs

---

*Built with ❤️ for the Cappuccino Trading System*
