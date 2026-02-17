#!/usr/bin/env python3
"""
Load GLM-4.7-Flash with 8-bit quantization to reduce memory usage
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

print('='*70)
print('Loading GLM-4.7-Flash with 8-bit Quantization')
print('='*70)

model_name = 'zai-org/GLM-4.7-Flash'

# Configure 8-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

print('\n1. Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)
print('   ✓ Tokenizer loaded')

print('\n2. Loading model with 8-bit quantization...')
print('   (This reduces memory by 50% - from ~20GB to ~10GB)')

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map='auto',
        low_cpu_mem_usage=True
    )
    print('   ✓ Model loaded successfully!')

    # Model info
    device = next(model.parameters()).device
    print(f'\n3. Model Info:')
    print(f'   Device: {device}')
    print(f'   Quantization: 8-bit')
    print(f'   Memory usage: ~10GB (vs 20GB unquantized)')

    # Test
    print(f'\n4. Testing inference...')
    messages = [{"role": "user", "content": "Is Bitcoin bullish or bearish today?"}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=50,
            temperature=0.3
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f'   Response: {response}')

    print('\n' + '='*70)
    print('✓ GLM-4.7-Flash (8-bit) setup complete!')
    print('='*70)

except RuntimeError as e:
    if 'out of memory' in str(e).lower():
        print('\n✗ Still out of memory with 8-bit quantization')
        print('\nRecommendation: Use a smaller model or increase system RAM')
        print('Alternative: Try GLM-4V-2B (2 billion params) or similar smaller model')
    else:
        print(f'\n✗ Error: {e}')

except Exception as e:
    print(f'\n✗ Error loading model: {e}')
    print('\nTroubleshooting:')
    print('  1. Check available RAM: free -h')
    print('  2. Close other applications')
    print('  3. Try smaller model: zai-org/GLM-4V-2B')
