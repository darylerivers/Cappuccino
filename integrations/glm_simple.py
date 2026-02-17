#!/usr/bin/env python3
"""
Simplified GLM-4 wrapper for faster responses
Uses shorter prompts and no JSON formatting for speed
"""

import requests
from typing import Optional, Dict, Any


class SimpleGLM:
    """
    Lightweight GLM-4 wrapper optimized for speed

    Trade-offs:
    - Faster responses (no JSON parsing)
    - Simpler prompts (less overhead)
    - Still maintains conversation context
    - Good for quick sentiment checks
    """

    def __init__(
        self,
        model: str = "glm4",
        url: str = "http://localhost:11434",
        temperature: float = 0.3
    ):
        self.model = model
        self.url = url
        self.temperature = temperature
        self.history = []

    def ask(self, question: str, system_context: Optional[str] = None) -> str:
        """
        Ask GLM-4 a question (fast, no JSON)

        Args:
            question: Your question
            system_context: Optional role/context (keep it SHORT!)

        Returns:
            GLM-4's response as plain text
        """
        # Build prompt
        if system_context:
            prompt = f"{system_context}\n\n{question}"
        else:
            prompt = question

        # Add recent history (last 2 turns only for speed)
        if self.history:
            context = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in self.history[-2:]
            ])
            prompt = f"Previous:\n{context}\n\nNow: {prompt}"

        # Call Ollama
        try:
            response = requests.post(
                f"{self.url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": 300  # Shorter responses = faster
                    }
                },
                timeout=30  # 30 second timeout
            )

            result = response.json()
            answer = result.get('response', '')

            # Update history
            self.history.append({"role": "user", "content": question})
            self.history.append({"role": "assistant", "content": answer})

            # Keep history small
            if len(self.history) > 6:
                self.history = self.history[-6:]

            return answer

        except Exception as e:
            return f"Error: {e}"

    def sentiment(self, text: str) -> str:
        """Quick sentiment analysis (bullish/bearish/neutral)"""
        return self.ask(
            f"Sentiment for crypto market: {text}\nAnswer in 1-2 sentences.",
            system_context="You are a crypto analyst. Be brief and direct."
        )

    def validate_signal(self, action: str, ticker: str, reason: str) -> str:
        """Quick signal validation"""
        return self.ask(
            f"Should I {action} {ticker}? Reason: {reason}\nGive quick yes/no with 1 sentence why.",
            system_context="You are a conservative risk manager."
        )

    def clear(self):
        """Clear conversation history"""
        self.history = []


# Quick usage example
if __name__ == "__main__":
    import time

    glm = SimpleGLM()

    print("=== Simple GLM-4 Test (Optimized for Speed) ===\n")

    # Test 1: Quick sentiment
    start = time.time()
    result = glm.sentiment("Bitcoin surges 8% on ETF approval news")
    elapsed = time.time() - start
    print(f"1. Sentiment Analysis ({elapsed:.1f}s):")
    print(f"   {result}\n")

    # Test 2: Signal validation
    start = time.time()
    result = glm.validate_signal("buy", "BTC", "strong momentum + positive news")
    elapsed = time.time() - start
    print(f"2. Signal Validation ({elapsed:.1f}s):")
    print(f"   {result}\n")

    # Test 3: Multi-turn (uses context)
    start = time.time()
    r1 = glm.ask("What's your BTC outlook for next week?",
                 system_context="You are a crypto analyst.")
    print(f"3. Outlook ({time.time() - start:.1f}s):")
    print(f"   {r1}\n")

    start = time.time()
    r2 = glm.ask("Should I increase my position?")  # Remembers previous context
    print(f"4. Follow-up ({time.time() - start:.1f}s):")
    print(f"   {r2}\n")

    print("Done! Much faster than the full agent.")
