#!/usr/bin/env python3
"""
Setup and test GLM-4.7-Flash model for sentiment analysis
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_glm_model():
    """Load GLM-4.7-Flash model"""
    model_name = 'zai-org/GLM-4.7-Flash'

    print('='*70)
    print('Loading GLM-4.7-Flash Model')
    print('='*70)

    print('\n1. Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    print('   ✓ Tokenizer loaded')

    print('\n2. Loading model...')
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map='auto'  # Automatically use GPU
    )
    print('   ✓ Model loaded')

    # Model info
    device = next(model.parameters()).device
    total_params = sum(p.numel() for p in model.parameters())

    print(f'\n3. Model Info:')
    print(f'   Device: {device}')
    print(f'   Parameters: {total_params/1e9:.2f}B')
    print(f'   Dtype: {next(model.parameters()).dtype}')

    return tokenizer, model

def test_sentiment_analysis(tokenizer, model):
    """Test sentiment analysis on crypto news"""
    print('\n4. Testing Sentiment Analysis:')
    print('='*70)

    test_texts = [
        "Bitcoin surges to new all-time high as institutional adoption increases",
        "Crypto market crashes amid regulatory concerns and exchange failures",
        "Ethereum maintains steady growth with successful network upgrades"
    ]

    for i, text in enumerate(test_texts, 1):
        print(f'\nTest {i}: "{text[:60]}..."')

        messages = [{
            "role": "user",
            "content": f"Analyze the sentiment (positive/negative/neutral) of this crypto news: {text}"
        }]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=100,
                temperature=0.3,
                do_sample=True
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the response part
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()

        print(f'Sentiment: {response[:200]}...')

def main():
    print('\n🚀 GLM-4.7-Flash Setup\n')

    # Load model
    tokenizer, model = load_glm_model()

    # Test it
    test_sentiment_analysis(tokenizer, model)

    print('\n' + '='*70)
    print('✓ GLM-4.7-Flash Setup Complete!')
    print('='*70)
    print('\nThe model is cached and ready to use.')
    print('Model location: ~/.cache/huggingface/hub/')

if __name__ == '__main__':
    main()
