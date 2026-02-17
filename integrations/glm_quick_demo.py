#!/usr/bin/env python3
"""
Quick GLM-4 demo - Fast responses for trading decisions
"""

import requests
import time

def quick_ask(question, max_words=50):
    """Ask GLM-4 with very short, focused prompts for fast responses"""
    start = time.time()

    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': 'glm4',
            'prompt': f"{question}\n\nAnswer in {max_words} words or less.",
            'stream': False,
            'options': {
                'temperature': 0.2,
                'num_predict': max_words * 2  # ~2 tokens per word
            }
        },
        timeout=30
    )

    elapsed = time.time() - start
    answer = response.json()['response']

    return answer, elapsed


if __name__ == "__main__":
    print("=== GLM-4 Quick Demo ===\n")

    # Test 1: Sentiment
    print("1. Quick Sentiment Check:")
    answer, t = quick_ask("Bitcoin up 5% on institutional buying. Sentiment: bullish or bearish?", 10)
    print(f"   Q: BTC up 5% on institutional buying. Sentiment?")
    print(f"   A: {answer.strip()}")
    print(f"   Time: {t:.1f}s\n")

    # Test 2: Risk check
    print("2. Risk Assessment:")
    answer, t = quick_ask("I want to buy BTC at 60% of my portfolio. Too risky?", 20)
    print(f"   Q: Buy BTC at 60% of portfolio. Too risky?")
    print(f"   A: {answer.strip()}")
    print(f"   Time: {t:.1f}s\n")

    # Test 3: Signal validation
    print("3. Signal Validation:")
    answer, t = quick_ask("Should I buy ETH now? Market is volatile but trending up.", 15)
    print(f"   Q: Buy ETH now? Market volatile but trending up.")
    print(f"   A: {answer.strip()}")
    print(f"   Time: {t:.1f}s\n")

    print("=== Key Insight ===")
    print("Short, focused questions = Fast responses (5-15s)")
    print("Long system prompts + JSON = Slow but structured (30-120s)")
    print("\nUse the right tool for the job!")
