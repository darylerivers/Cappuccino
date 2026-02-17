#!/usr/bin/env python3
"""
GLM-4 Agent Wrapper - Makes GLM-4 behave like an intelligent trading agent
Provides Claude Code-like functionality with structured prompts and context management
"""

import json
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


class GLMAgent:
    """
    Intelligent agent wrapper for GLM-4 with specialized trading analysis capabilities

    Features:
    - System prompt engineering for consistent behavior
    - Multi-turn conversation with context management
    - Structured JSON outputs
    - Specialized analysis methods (sentiment, risk, technical)
    - Token usage tracking
    """

    def __init__(
        self,
        model: str = "glm4",
        ollama_url: str = "http://localhost:11434",
        system_prompt_file: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000
    ):
        self.model = model
        self.ollama_url = ollama_url
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Load system prompt
        if system_prompt_file is None:
            system_prompt_file = Path(__file__).parent / "glm_agent_prompt.txt"

        with open(system_prompt_file, 'r') as f:
            self.system_prompt = f.read()

        # Conversation history for multi-turn context
        self.conversation_history: List[Dict[str, str]] = []

        # Token usage tracking
        self.total_tokens_used = 0
        self.total_requests = 0

    def _call_ollama(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        format_json: bool = False,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Internal method to call Ollama API with proper error handling
        """
        url = f"{self.ollama_url}/api/generate"

        # Build full prompt with system context
        full_prompt = f"{self.system_prompt}\n\n# Current Query\n{prompt}"

        # Add conversation history if available
        if self.conversation_history:
            context = "\n\n# Recent Context\n"
            for msg in self.conversation_history[-3:]:  # Last 3 turns
                context += f"{msg['role'].upper()}: {msg['content']}\n"
            full_prompt = f"{self.system_prompt}\n{context}\n# Current Query\n{prompt}"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": stream,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_predict": self.max_tokens
            }
        }

        if format_json:
            payload["format"] = "json"

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()

            # Track usage
            self.total_requests += 1
            if 'eval_count' in result:
                self.total_tokens_used += result['eval_count']

            return result

        except requests.exceptions.RequestException as e:
            return {"error": str(e), "response": None}

    def analyze_sentiment(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze market sentiment from news, social media, or market data

        Args:
            text: News headline, tweet, or market description
            context: Optional market data (price, volume, etc.)

        Returns:
            Structured sentiment analysis with confidence scores
        """
        prompt = f"Analyze market sentiment:\n\nText: {text}"

        if context:
            prompt += f"\n\nAdditional Context:\n{json.dumps(context, indent=2)}"

        prompt += "\n\nProvide analysis in JSON format."

        result = self._call_ollama(prompt, format_json=True)

        if result.get('error'):
            return {"error": result['error']}

        try:
            analysis = json.loads(result['response'])
            self.conversation_history.append({
                "role": "user",
                "content": f"Sentiment analysis: {text[:100]}"
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": result['response']
            })
            return analysis
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response", "raw": result['response']}

    def evaluate_trade_signal(
        self,
        signal: Dict[str, Any],
        market_data: Dict[str, Any],
        portfolio: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a DRL agent's trading signal for validity and risk

        Args:
            signal: Trading signal from DRL agent (action, position, confidence)
            market_data: Current market conditions
            portfolio: Current portfolio state

        Returns:
            Risk assessment and recommendation
        """
        prompt = f"""Evaluate this trading signal:

Signal: {json.dumps(signal, indent=2)}
Market Data: {json.dumps(market_data, indent=2)}
"""
        if portfolio:
            prompt += f"Portfolio: {json.dumps(portfolio, indent=2)}\n"

        prompt += "\nProvide risk assessment and recommendation in JSON format."

        result = self._call_ollama(prompt, format_json=True)

        if result.get('error'):
            return {"error": result['error']}

        try:
            return json.loads(result['response'])
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response", "raw": result['response']}

    def analyze_news_impact(
        self,
        headlines: List[str],
        tickers: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze multiple news headlines and predict market impact

        Args:
            headlines: List of recent news headlines
            tickers: Crypto tickers to analyze (BTC, ETH, etc.)

        Returns:
            Aggregated sentiment and predicted impact per ticker
        """
        prompt = f"""Analyze these news headlines and predict impact on {', '.join(tickers)}:

Headlines:
"""
        for i, headline in enumerate(headlines, 1):
            prompt += f"{i}. {headline}\n"

        prompt += f"\nProvide impact analysis for each ticker in JSON format."

        result = self._call_ollama(prompt, format_json=True, temperature=0.2)

        if result.get('error'):
            return {"error": result['error']}

        try:
            return json.loads(result['response'])
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response", "raw": result['response']}

    def risk_assessment(
        self,
        portfolio: Dict[str, Any],
        market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive risk assessment of current portfolio

        Args:
            portfolio: Current positions, PnL, exposure
            market_conditions: Volatility, trend, sentiment

        Returns:
            Risk analysis with suggested adjustments
        """
        prompt = f"""Perform comprehensive risk assessment:

Portfolio: {json.dumps(portfolio, indent=2)}
Market Conditions: {json.dumps(market_conditions, indent=2)}

Identify risks and suggest position adjustments in JSON format.
"""

        result = self._call_ollama(prompt, format_json=True, temperature=0.2)

        if result.get('error'):
            return {"error": result['error']}

        try:
            return json.loads(result['response'])
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response", "raw": result['response']}

    def chat(
        self,
        message: str,
        temperature: Optional[float] = None
    ) -> str:
        """
        Free-form chat with context retention (like Claude Code)

        Args:
            message: User message
            temperature: Optional temperature override

        Returns:
            Agent response as text
        """
        result = self._call_ollama(message, temperature=temperature)

        if result.get('error'):
            return f"Error: {result['error']}"

        response = result['response']

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": message})
        self.conversation_history.append({"role": "assistant", "content": response})

        # Trim history to last 10 turns (5 exchanges)
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

        return response

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

    def get_stats(self) -> Dict[str, Any]:
        """Get agent usage statistics"""
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens_used,
            "avg_tokens_per_request": (
                self.total_tokens_used / self.total_requests
                if self.total_requests > 0 else 0
            ),
            "conversation_turns": len(self.conversation_history)
        }


# Example usage
if __name__ == "__main__":
    # Initialize agent
    agent = GLMAgent()

    print("=== GLM-4 Trading Agent Test ===\n")

    # Test 1: Sentiment Analysis
    print("1. Sentiment Analysis:")
    sentiment = agent.analyze_sentiment(
        "Bitcoin surges 8% as major institutions announce crypto adoption plans"
    )
    print(json.dumps(sentiment, indent=2))
    print()

    # Test 2: Trade Signal Evaluation
    print("2. Trade Signal Evaluation:")
    signal = {
        "action": "buy",
        "ticker": "BTC",
        "position_size": 0.3,
        "confidence": 0.85,
        "reason": "strong momentum + positive sentiment"
    }
    market = {
        "price": 45000,
        "volume_24h": "high",
        "volatility": "medium",
        "trend": "upward"
    }
    evaluation = agent.evaluate_trade_signal(signal, market)
    print(json.dumps(evaluation, indent=2))
    print()

    # Test 3: Multi-turn chat
    print("3. Multi-turn Conversation:")
    response1 = agent.chat("What's your assessment of current BTC market conditions?")
    print(f"Agent: {response1[:200]}...")
    print()

    response2 = agent.chat("Given that, should we increase our position?")
    print(f"Agent: {response2[:200]}...")
    print()

    # Stats
    print("4. Agent Statistics:")
    print(json.dumps(agent.get_stats(), indent=2))
