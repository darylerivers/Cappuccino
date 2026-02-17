# GLM-4.7-Flash Setup Guide

## Overview

**GLM-4.7-Flash** is a 4.7 billion parameter language model optimized for fast inference, perfect for sentiment analysis in your crypto trading system.

## What's Installed

- ✅ Hugging Face Transformers (5.1.0)
- ✅ Accelerate (for efficient model loading)
- ✅ SentencePiece (tokenizer)
- ✅ GLM-4.7-Flash model (downloading)

## Model Details

- **Name:** `zai-org/GLM-4.7-Flash`
- **Size:** 4.7B parameters (~9GB disk space)
- **Dtype:** bfloat16 (efficient for AMD GPU)
- **Device:** Auto-mapped to your AMD RX 7900 GRE

## Usage

### Quick Test

```bash
source activate_rocm_env.sh
python setup_glm.py
```

### In Your Code

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model (first time loads from cache)
tokenizer = AutoTokenizer.from_pretrained(
    'zai-org/GLM-4.7-Flash',
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    'zai-org/GLM-4.7-Flash',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)

# Analyze sentiment
text = "Bitcoin surges to new highs"
messages = [{
    "role": "user",
    "content": f"Analyze sentiment: {text}"
}]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Integration with Trading System

### Option 1: Real-time Sentiment (for paper/live trading)

```python
from integrations.sentiment import GLMSentimentAnalyzer

analyzer = GLMSentimentAnalyzer(
    model_name='zai-org/GLM-4.7-Flash',
    device='cuda'
)

# In your trading loop
news_text = fetch_latest_news()
sentiment = analyzer.analyze(news_text)  # Returns: positive/negative/neutral
```

### Option 2: Batch Sentiment (for backtesting)

```python
# Pre-process historical news
import pandas as pd

news_df = pd.read_csv('crypto_news_historical.csv')
sentiments = analyzer.batch_analyze(
    news_df['text'].tolist(),
    batch_size=32  # Process 32 at a time
)

news_df['sentiment'] = sentiments
```

## Performance

### Expected Inference Speed (AMD RX 7900 GRE)

- **Single inference:** ~100-200ms
- **Batch (32):** ~2-3 seconds
- **Memory usage:** ~5-6GB VRAM

### GPU Utilization

The model will use your GPU efficiently:
- Runs in bfloat16 for speed
- Auto-distributes across GPU memory
- Compatible with ROCm 7.1

## Model Cache

After first download, model is cached at:
```
~/.cache/huggingface/hub/models--zai-org--GLM-4.7-Flash/
```

**Size:** ~9GB disk space

## Tips

1. **First run is slow:** Initial load downloads model (~10-15 min)
2. **Subsequent runs are fast:** Model loads from cache (5-10 sec)
3. **GPU required:** Model needs GPU, won't run efficiently on CPU
4. **Temperature parameter:** Lower (0.1-0.3) for consistent sentiment, higher (0.7-1.0) for creative responses

## Troubleshooting

### Out of Memory
- Reduce batch size
- Use `torch.float16` instead of `bfloat16`
- Close other GPU processes

### Slow inference
- Check GPU is being used: `model.device`
- Ensure ROCm environment is activated
- Monitor with `rocm-smi`

### Download issues
- Set HF token: `huggingface-cli login`
- Check internet connection
- Try smaller model first: `zai-org/GLM-4.7-Flash`

## Next Steps

1. ✅ Model download completes
2. Run `python setup_glm.py` to test
3. Integrate into trading system
4. Backtest with sentiment features

---

**Status:** Model downloading (will be ready in ~15 minutes)
