"""
Pipeline Package - Automated Trading Pipeline
Orchestrates training → backtesting → CGE stress testing → paper trading → live trading
"""

from .state_manager import PipelineStateManager
from .gates import BacktestGate, CGEStressGate, PaperTradingGate
from .backtest_runner import BacktestRunner
from .cge_runner import CGEStressRunner
from .notifications import PipelineNotifier

__all__ = [
    'PipelineStateManager',
    'BacktestGate',
    'CGEStressGate',
    'PaperTradingGate',
    'BacktestRunner',
    'CGEStressRunner',
    'PipelineNotifier',
]
