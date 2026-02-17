"""
Trade Logger for AI Training Data Collection

Captures trade decisions with full context (market conditions, signals, news, etc.)
and outcomes for fine-tuning TinyLlama.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class TradeLogger:
    """Logs trades with context for AI training."""

    def __init__(self, log_dir: str = "ai_training/data"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Separate files for different data types
        self.trades_file = self.log_dir / "trades.jsonl"
        self.training_file = self.log_dir / "training_data.jsonl"

    def log_trade_decision(
        self,
        symbol: str,
        action: str,  # 'BUY', 'SELL', 'HOLD', 'SKIP'
        agent_signal: float,
        position_size: float,
        market_context: Dict,
        news_summary: Optional[Dict] = None,
        macro_context: Optional[Dict] = None,
        reasoning: Optional[str] = None
    ) -> str:
        """
        Log a trade decision with full context.

        Returns:
            trade_id for later outcome tracking
        """
        trade_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        trade_record = {
            "trade_id": trade_id,
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action,
            "agent_signal": agent_signal,
            "position_size": position_size,
            "market_context": market_context,
            "news_summary": news_summary or {},
            "macro_context": macro_context or {},
            "reasoning": reasoning,
            "outcome": None,  # Will be filled later
            "outcome_timestamp": None
        }

        # Append to trades log
        with open(self.trades_file, 'a') as f:
            f.write(json.dumps(trade_record) + '\n')

        return trade_id

    def log_trade_outcome(
        self,
        trade_id: str,
        profit_loss: float,
        profit_loss_pct: float,
        hold_duration_hours: float,
        exit_reason: str
    ):
        """
        Log the outcome of a trade.

        Updates the trade record and creates training example.
        """
        # Read all trades
        trades = []
        if self.trades_file.exists():
            with open(self.trades_file, 'r') as f:
                trades = [json.loads(line) for line in f]

        # Find and update the trade
        for trade in trades:
            if trade['trade_id'] == trade_id:
                trade['outcome'] = {
                    'profit_loss': profit_loss,
                    'profit_loss_pct': profit_loss_pct,
                    'hold_duration_hours': hold_duration_hours,
                    'exit_reason': exit_reason,
                    'success': profit_loss_pct > 0
                }
                trade['outcome_timestamp'] = datetime.now().isoformat()

                # Create training example
                self._create_training_example(trade)
                break

        # Write back updated trades
        with open(self.trades_file, 'w') as f:
            for trade in trades:
                f.write(json.dumps(trade) + '\n')

    def _create_training_example(self, trade: Dict):
        """
        Convert trade record into training format for fine-tuning.

        Format: Alpaca-style instruction tuning
        """
        # Build context string
        context_parts = []

        # Market context
        if trade['market_context']:
            ctx = trade['market_context']
            context_parts.append(f"Symbol: {trade['symbol']}")
            context_parts.append(f"Agent Signal: {ctx.get('signal', 'N/A')}")
            context_parts.append(f"Price: ${ctx.get('price', 'N/A')}")
            context_parts.append(f"Volume: {ctx.get('volume', 'N/A')}")

        # Macro context
        if trade['macro_context']:
            macro = trade['macro_context']
            context_parts.append(f"VIX: {macro.get('vix', 'N/A')}")
            context_parts.append(f"Market Regime: {macro.get('regime', 'N/A')}")
            context_parts.append(f"Position Sizing Multiplier: {macro.get('multiplier', 1.0):.0%}")

        # News context
        if trade['news_summary']:
            news = trade['news_summary']
            context_parts.append(f"News Sentiment: {news.get('recommendation', 'neutral')}")
            if news.get('bullish_count', 0) > 0:
                context_parts.append(f"Bullish Signals: {news['bullish_count']}")
            if news.get('bearish_count', 0) > 0:
                context_parts.append(f"Bearish Signals: {news['bearish_count']}")

        context = "\n".join(context_parts)

        # Build instruction
        instruction = f"Analyze this trading opportunity and provide a recommendation:"

        # Build input
        input_text = context

        # Build output (what the model should learn to say)
        outcome = trade['outcome']
        if outcome:
            if outcome['success']:
                output = f"RECOMMENDATION: {trade['action']}\n"
                output += f"CONFIDENCE: HIGH\n"
                output += f"REASONING: {trade['reasoning'] or 'Market conditions favorable'}\n"
                output += f"EXPECTED OUTCOME: This trade resulted in {outcome['profit_loss_pct']:.2f}% profit. "
                output += f"Similar patterns historically perform well."
            else:
                output = f"RECOMMENDATION: AVOID or REDUCE POSITION\n"
                output += f"CONFIDENCE: MEDIUM\n"
                output += f"REASONING: Historical pattern suggests risk. "
                output += f"This trade resulted in {outcome['profit_loss_pct']:.2f}% loss. "
                output += f"Key factors to watch: {trade['reasoning'] or 'market volatility'}"
        else:
            # No outcome yet - skip for now
            return

        # Create Alpaca-style training example
        training_example = {
            "instruction": instruction,
            "input": input_text,
            "output": output,
            "metadata": {
                "trade_id": trade['trade_id'],
                "symbol": trade['symbol'],
                "timestamp": trade['timestamp'],
                "outcome_pct": outcome['profit_loss_pct']
            }
        }

        # Append to training data
        with open(self.training_file, 'a') as f:
            f.write(json.dumps(training_example) + '\n')

    def get_training_stats(self) -> Dict:
        """Get statistics about collected training data."""
        stats = {
            "total_trades": 0,
            "trades_with_outcomes": 0,
            "training_examples": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "avg_profit_pct": 0.0
        }

        if not self.trades_file.exists():
            return stats

        with open(self.trades_file, 'r') as f:
            trades = [json.loads(line) for line in f]

        stats['total_trades'] = len(trades)

        trades_with_outcomes = [t for t in trades if t.get('outcome')]
        stats['trades_with_outcomes'] = len(trades_with_outcomes)

        if trades_with_outcomes:
            profits = [t['outcome']['profit_loss_pct'] for t in trades_with_outcomes]
            stats['winning_trades'] = sum(1 for p in profits if p > 0)
            stats['losing_trades'] = sum(1 for p in profits if p < 0)
            stats['avg_profit_pct'] = sum(profits) / len(profits)

        if self.training_file.exists():
            with open(self.training_file, 'r') as f:
                stats['training_examples'] = sum(1 for _ in f)

        return stats

    def export_for_training(self, output_file: str = "ai_training/data/dataset.json"):
        """
        Export training data in format ready for fine-tuning.

        Combines all training examples into a single JSON file.
        """
        if not self.training_file.exists():
            print("No training data available yet.")
            return

        with open(self.training_file, 'r') as f:
            examples = [json.loads(line) for line in f]

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(examples, f, indent=2)

        print(f"Exported {len(examples)} training examples to {output_file}")
        return output_file


# Singleton instance
_logger = None

def get_trade_logger() -> TradeLogger:
    """Get or create trade logger singleton."""
    global _logger
    if _logger is None:
        _logger = TradeLogger()
    return _logger
